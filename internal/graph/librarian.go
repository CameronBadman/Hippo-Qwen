package graph

import (
	"sort"
	"time"
)

type LibrarianRequest struct {
	Anchor     MemoryNode   `json:"anchor"`
	Candidates []MemoryNode `json:"candidates"`
	Graph      *Graph       `json:"graph,omitempty"`
	Now        time.Time    `json:"now,omitempty"`
}

type EdgeAction struct {
	CandidateID     string  `json:"candidate_id"`
	ConnectScore    float64 `json:"connect_score"`
	EdgeType        string  `json:"edge_type"`
	Weight          float64 `json:"weight"`
	DecayRate       float64 `json:"decay_rate"`
	ImportanceDelta float64 `json:"importance_delta"`
}

type LibrarianResponse struct {
	Actions          []EdgeAction `json:"actions"`
	CreateNewCluster bool         `json:"create_new_cluster"`
	NewClusterScore  float64      `json:"new_cluster_score"`
}

type HeuristicLibrarian struct {
	Threshold float64
	MaxEdges  int
}

func NewHeuristicLibrarian() HeuristicLibrarian {
	return HeuristicLibrarian{Threshold: 0.28, MaxEdges: 6}
}

func (l HeuristicLibrarian) Place(req LibrarianRequest) LibrarianResponse {
	if l.Threshold == 0 {
		l.Threshold = 0.28
	}
	if l.MaxEdges == 0 {
		l.MaxEdges = 6
	}
	now := req.Now
	if now.IsZero() {
		now = time.Now().UTC()
	}
	actions := make([]EdgeAction, 0, len(req.Candidates))
	for _, candidate := range req.Candidates {
		cosine := Cosine(req.Anchor.Embedding, candidate.Embedding)
		jaccard := TokenJaccard(req.Anchor.Text+" "+req.Anchor.Summary, candidate.Text+" "+candidate.Summary)
		meta := metadataScore(req.Anchor.Metadata, candidate.Metadata)
		cluster := clusterScore(req.Anchor, candidate)
		recency := recencyScore(now, candidate.Timestamp)
		graphBonus := graphProximityScore(req.Graph, req.Anchor.ID, candidate.ID)
		score := 0.35*cosine + 0.20*jaccard + 0.20*meta + 0.15*cluster + 0.05*recency + 0.05*graphBonus
		if score < l.Threshold {
			continue
		}
		edgeType := inferEdgeType(req.Anchor, candidate, jaccard, meta)
		actions = append(actions, EdgeAction{
			CandidateID:     candidate.ID,
			ConnectScore:    score,
			EdgeType:        edgeType,
			Weight:          clamp(0.2+score, 0.05, 1.2),
			DecayRate:       decayForEdge(edgeType),
			ImportanceDelta: clamp((score-0.5)*0.08, -0.04, 0.08),
		})
	}
	sort.Slice(actions, func(i, j int) bool {
		if actions[i].ConnectScore == actions[j].ConnectScore {
			return actions[i].CandidateID < actions[j].CandidateID
		}
		return actions[i].ConnectScore > actions[j].ConnectScore
	})
	if len(actions) > l.MaxEdges {
		actions = actions[:l.MaxEdges]
	}
	return LibrarianResponse{
		Actions:          actions,
		CreateNewCluster: len(actions) == 0,
		NewClusterScore:  1 - bestActionScore(actions),
	}
}

func (l HeuristicLibrarian) ActionsToEdges(anchor MemoryNode, response LibrarianResponse) []MemoryEdge {
	edges := make([]MemoryEdge, 0, len(response.Actions)*2)
	for _, action := range response.Actions {
		forward := MemoryEdge{
			Source:    anchor.ID,
			Target:    action.CandidateID,
			Type:      action.EdgeType,
			Weight:    action.Weight,
			DecayRate: action.DecayRate,
		}
		edges = append(edges, normalizeEdge(forward))
		if action.EdgeType != "temporal_next" {
			backward := forward
			backward.Source, backward.Target = forward.Target, forward.Source
			backward.Weight *= 0.85
			edges = append(edges, normalizeEdge(backward))
		}
	}
	return edges
}

func metadataScore(a map[string]string, b map[string]string) float64 {
	if len(a) == 0 || len(b) == 0 {
		return 0
	}
	var matches int
	var total int
	for key, left := range a {
		total++
		if right, ok := b[key]; ok && right == left {
			matches++
		}
	}
	if total == 0 {
		return 0
	}
	return float64(matches) / float64(total)
}

func clusterScore(anchor MemoryNode, candidate MemoryNode) float64 {
	if anchor.Cluster == "" || candidate.Cluster == "" {
		return 0
	}
	if anchor.Cluster == candidate.Cluster {
		return 1
	}
	return 0
}

func recencyScore(now time.Time, timestamp time.Time) float64 {
	if timestamp.IsZero() {
		return 0
	}
	days := now.Sub(timestamp).Hours() / 24
	if days < 0 {
		days = 0
	}
	if days > 30 {
		return 0
	}
	return 1 - days/30
}

func graphProximityScore(g *Graph, anchorID string, candidateID string) float64 {
	if g == nil {
		return 0
	}
	if _, ok := g.Edges[EdgeID(anchorID, candidateID, "used_with")]; ok {
		return 1
	}
	var outgoing int
	for _, edge := range g.Edges {
		if edge.Source == candidateID {
			outgoing++
		}
	}
	if outgoing > 6 {
		return 0.3
	}
	if outgoing > 0 {
		return 0.15
	}
	return 0
}

func inferEdgeType(anchor MemoryNode, candidate MemoryNode, jaccard float64, meta float64) string {
	if anchor.Cluster != "" && anchor.Cluster == candidate.Cluster {
		return "same_cluster"
	}
	if meta > 0.5 {
		return "same_context"
	}
	if anchor.Timestamp.After(candidate.Timestamp) {
		age := anchor.Timestamp.Sub(candidate.Timestamp)
		if age > 0 && age < 2*time.Hour {
			return "temporal_next"
		}
	}
	text := anchor.Text + " " + candidate.Text
	if containsAny(text, []string{"prefer", "preference", "like", "dislike", "always", "never"}) {
		return "preference"
	}
	if containsAny(text, []string{"correct", "instead", "actually", "wrong"}) {
		return "correction"
	}
	if jaccard > 0.18 {
		return "same_topic"
	}
	return "used_with"
}

func containsAny(text string, needles []string) bool {
	for _, needle := range needles {
		if tokenPattern.MatchString(needle) && TokenJaccard(text, needle) > 0 {
			return true
		}
	}
	return false
}

func decayForEdge(edgeType string) float64 {
	switch edgeType {
	case "preference", "correction":
		return 0.005
	case "same_cluster", "same_context":
		return 0.01
	case "temporal_next":
		return 0.03
	default:
		return 0.02
	}
}

func bestActionScore(actions []EdgeAction) float64 {
	if len(actions) == 0 {
		return 0
	}
	return actions[0].ConnectScore
}
