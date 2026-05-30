package graph

import (
	"bufio"
	"crypto/sha1"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"sync"
	"time"
)

type Store struct {
	mu           sync.RWMutex
	dir          string
	eventPath    string
	snapshotPath string
	graph        *Graph
	lastSeq      int64
}

type snapshot struct {
	LastSeq int64  `json:"last_seq"`
	Graph   *Graph `json:"graph"`
}

func OpenStore(dir string) (*Store, error) {
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return nil, err
	}
	store := &Store{
		dir:          dir,
		eventPath:    filepath.Join(dir, "events.jsonl"),
		snapshotPath: filepath.Join(dir, "snapshot.json"),
		graph:        NewGraph(),
	}
	if err := store.load(); err != nil {
		return nil, err
	}
	return store, nil
}

func (s *Store) Graph() *Graph {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.graph.Clone()
}

func (s *Store) AddNode(node MemoryNode) (MemoryNode, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if node.Text == "" {
		return MemoryNode{}, errors.New("node text is required")
	}
	if node.ID == "" {
		node.ID = s.nextNodeID(node.Text)
	}
	if node.Timestamp.IsZero() {
		node.Timestamp = time.Now().UTC()
	}
	if node.Importance == 0 {
		node.Importance = 0.5
	}
	if len(node.Embedding) == 0 {
		node.Embedding = EmbedText(node.Text, DefaultEmbeddingDims)
	}
	if node.Metadata != nil {
		node.Metadata = cloneStringMap(node.Metadata)
	}
	event := GraphEvent{Type: EventNodeInsert, Node: &node}
	if err := s.appendLocked(event); err != nil {
		return MemoryNode{}, err
	}
	return node, nil
}

func (s *Store) UpsertEdge(edge MemoryEdge) (MemoryEdge, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if _, ok := s.graph.Nodes[edge.Source]; !ok {
		return MemoryEdge{}, fmt.Errorf("source node %q does not exist", edge.Source)
	}
	if _, ok := s.graph.Nodes[edge.Target]; !ok {
		return MemoryEdge{}, fmt.Errorf("target node %q does not exist", edge.Target)
	}
	edge = normalizeEdgeWithGraph(edge, s.graph)
	if existing, ok := s.graph.Edges[edge.ID]; ok {
		edge.UseCount += existing.UseCount
		edge.EvidenceCount += existing.EvidenceCount
		if edge.Confidence == 0 {
			edge.Confidence = existing.Confidence
		}
		if edge.ActivationMask == 0 {
			edge.ActivationMask = existing.ActivationMask
		}
		if edge.LastOutcome == "" {
			edge.LastOutcome = existing.LastOutcome
		}
		if edge.LastUsedAt.IsZero() {
			edge.LastUsedAt = existing.LastUsedAt
		}
		if existing.Protected {
			edge.Protected = true
		}
	}
	event := GraphEvent{Type: EventEdgeUpsert, Edge: &edge}
	if err := s.appendLocked(event); err != nil {
		return MemoryEdge{}, err
	}
	return edge, nil
}

func (s *Store) ApplyFeedback(feedback Feedback) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	event := GraphEvent{Type: EventFeedback, Feedback: &feedback}
	return s.appendLocked(event)
}

func (s *Store) DecayAndGC(factor float64, threshold float64) ([]string, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if factor <= 0 || factor > 1 {
		return nil, errors.New("decay factor must be in (0,1]")
	}
	if threshold < 0 {
		return nil, errors.New("gc threshold must be non-negative")
	}
	run := DecayRun{Factor: factor, GCThreshold: threshold}
	if err := s.appendLocked(GraphEvent{Type: EventDecay, Decay: &run}); err != nil {
		return nil, err
	}
	removed := collectGCEdges(s.graph, threshold)
	if len(removed) == 0 {
		return nil, nil
	}
	sort.Strings(removed)
	if err := s.appendLocked(GraphEvent{Type: EventGC, Removed: removed}); err != nil {
		return nil, err
	}
	return removed, nil
}

func (s *Store) SaveSnapshot() error {
	s.mu.RLock()
	snap := snapshot{LastSeq: s.lastSeq, Graph: s.graph.Clone()}
	s.mu.RUnlock()
	payload, err := json.MarshalIndent(snap, "", "  ")
	if err != nil {
		return err
	}
	tmp := s.snapshotPath + ".tmp"
	if err := os.WriteFile(tmp, payload, 0o644); err != nil {
		return err
	}
	return os.Rename(tmp, s.snapshotPath)
}

func (s *Store) load() error {
	if payload, err := os.ReadFile(s.snapshotPath); err == nil {
		var snap snapshot
		if err := json.Unmarshal(payload, &snap); err != nil {
			return err
		}
		if snap.Graph != nil {
			s.graph = snap.Graph
		}
		s.lastSeq = snap.LastSeq
	} else if !errors.Is(err, os.ErrNotExist) {
		return err
	}
	file, err := os.Open(s.eventPath)
	if errors.Is(err, os.ErrNotExist) {
		return nil
	}
	if err != nil {
		return err
	}
	defer file.Close()
	scanner := bufio.NewScanner(file)
	scanner.Buffer(make([]byte, 0, 64*1024), 1024*1024)
	for scanner.Scan() {
		var event GraphEvent
		if err := json.Unmarshal(scanner.Bytes(), &event); err != nil {
			return err
		}
		if event.Seq <= s.lastSeq {
			continue
		}
		s.apply(event)
		if event.Seq > s.lastSeq {
			s.lastSeq = event.Seq
		}
	}
	return scanner.Err()
}

func (s *Store) appendLocked(event GraphEvent) error {
	s.lastSeq++
	event.Seq = s.lastSeq
	if event.Timestamp.IsZero() {
		event.Timestamp = time.Now().UTC()
	}
	payload, err := json.Marshal(event)
	if err != nil {
		return err
	}
	file, err := os.OpenFile(s.eventPath, os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0o644)
	if err != nil {
		return err
	}
	if _, err := file.Write(append(payload, '\n')); err != nil {
		_ = file.Close()
		return err
	}
	if err := file.Close(); err != nil {
		return err
	}
	s.apply(event)
	return nil
}

func (s *Store) apply(event GraphEvent) {
	switch event.Type {
	case EventNodeInsert:
		if event.Node != nil {
			node := *event.Node
			node.Embedding = append([]float64(nil), node.Embedding...)
			if node.Metadata != nil {
				node.Metadata = cloneStringMap(node.Metadata)
			}
			s.graph.Nodes[node.ID] = node
		}
	case EventEdgeUpsert:
		if event.Edge != nil {
			edge := normalizeEdgeWithGraph(*event.Edge, s.graph)
			s.graph.Edges[edge.ID] = edge
		}
	case EventFeedback:
		if event.Feedback != nil {
			applyFeedback(s.graph, *event.Feedback)
		}
	case EventDecay:
		if event.Decay != nil {
			applyDecay(s.graph, event.Decay.Factor)
		}
	case EventGC:
		for _, id := range event.Removed {
			delete(s.graph.Edges, id)
		}
	}
}

func (s *Store) nextNodeID(text string) string {
	sum := sha1.Sum([]byte(text))
	base := "mem_" + hex.EncodeToString(sum[:])[:12]
	id := base
	for i := 2; ; i++ {
		if _, ok := s.graph.Nodes[id]; !ok {
			return id
		}
		id = fmt.Sprintf("%s_%d", base, i)
	}
}

func normalizeEdge(edge MemoryEdge) MemoryEdge {
	if edge.Type == "" {
		edge.Type = "related"
	}
	if edge.ID == "" {
		edge.ID = EdgeID(edge.Source, edge.Target, edge.Type)
	}
	if edge.Weight == 0 {
		edge.Weight = 0.5
	}
	if edge.DecayRate == 0 {
		edge.DecayRate = 0.02
	}
	if edge.Confidence == 0 {
		edge.Confidence = clamp(edge.Weight, 0.05, 1)
	}
	if edge.LastUsedAt.IsZero() {
		edge.LastUsedAt = time.Now().UTC()
	}
	return edge
}

func normalizeEdgeWithGraph(edge MemoryEdge, g *Graph) MemoryEdge {
	edge = normalizeEdge(edge)
	if edge.ActivationMask == 0 && g != nil {
		source, sourceOK := g.Nodes[edge.Source]
		target, targetOK := g.Nodes[edge.Target]
		if sourceOK && targetOK {
			edge.ActivationMask = ActivationMaskForEdge(source, target, edge.Type)
		}
	}
	return edge
}

func EdgeID(source string, target string, edgeType string) string {
	return source + "->" + target + ":" + edgeType
}

func applyFeedback(g *Graph, feedback Feedback) {
	now := time.Now().UTC()
	delta := -0.08
	switch feedback.Outcome {
	case "helpful":
		delta = 0.12
	case "corrected":
		delta = -0.18
	case "ignored":
		delta = -0.08
	}
	for _, id := range feedback.EdgeIDs {
		edge, ok := g.Edges[id]
		if !ok {
			continue
		}
		edge.Weight = clamp(edge.Weight+delta, 0, 1.5)
		edge.Confidence = clamp(edge.Confidence+delta/2, 0, 1)
		edge.LastOutcome = feedback.Outcome
		edge.LastUsedAt = now
		if delta > 0 {
			edge.UseCount++
			edge.EvidenceCount++
		}
		g.Edges[id] = edge
	}
	for _, id := range feedback.NodeIDs {
		node, ok := g.Nodes[id]
		if !ok {
			continue
		}
		node.Importance = clamp(node.Importance+delta/2, 0, 1)
		g.Nodes[id] = node
	}
}

func applyDecay(g *Graph, factor float64) {
	for id, edge := range g.Edges {
		if edge.Protected {
			continue
		}
		edge.Weight *= factor * (1 - edge.DecayRate)
		g.Edges[id] = edge
	}
}

func collectGCEdges(g *Graph, threshold float64) []string {
	var removed []string
	for id, edge := range g.Edges {
		if edge.Protected || edge.EvidenceCount >= 3 {
			continue
		}
		if edge.Weight < threshold {
			removed = append(removed, id)
		}
	}
	return removed
}

func clamp(value float64, minValue float64, maxValue float64) float64 {
	if value < minValue {
		return minValue
	}
	if value > maxValue {
		return maxValue
	}
	return value
}
