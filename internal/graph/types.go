package graph

import "time"

const (
	EventNodeInsert = "node_insert"
	EventEdgeUpsert = "edge_upsert"
	EventFeedback   = "feedback"
	EventDecay      = "decay"
	EventGC         = "gc"
)

type MemoryNode struct {
	ID         string            `json:"id"`
	Text       string            `json:"text"`
	Summary    string            `json:"summary,omitempty"`
	Embedding  []float64         `json:"embedding"`
	Timestamp  time.Time         `json:"timestamp"`
	Source     string            `json:"source,omitempty"`
	Importance float64           `json:"importance"`
	Cluster    string            `json:"cluster,omitempty"`
	Metadata   map[string]string `json:"metadata,omitempty"`
}

type MemoryEdge struct {
	ID            string    `json:"id"`
	Source        string    `json:"source"`
	Target        string    `json:"target"`
	Type          string    `json:"type"`
	Weight        float64   `json:"weight"`
	DecayRate     float64   `json:"decay_rate"`
	UseCount      int       `json:"use_count"`
	LastUsedAt    time.Time `json:"last_used_at"`
	EvidenceCount int       `json:"evidence_count"`
	Protected     bool      `json:"protected"`
}

type GraphEvent struct {
	Seq       int64       `json:"seq"`
	Type      string      `json:"type"`
	Timestamp time.Time   `json:"timestamp"`
	Node      *MemoryNode `json:"node,omitempty"`
	Edge      *MemoryEdge `json:"edge,omitempty"`
	Feedback  *Feedback   `json:"feedback,omitempty"`
	Decay     *DecayRun   `json:"decay,omitempty"`
	Removed   []string    `json:"removed,omitempty"`
}

type Feedback struct {
	Outcome string   `json:"outcome"`
	NodeIDs []string `json:"node_ids,omitempty"`
	EdgeIDs []string `json:"edge_ids,omitempty"`
	Note    string   `json:"note,omitempty"`
}

type DecayRun struct {
	Factor      float64 `json:"factor"`
	GCThreshold float64 `json:"gc_threshold"`
}

type Graph struct {
	Nodes map[string]MemoryNode `json:"nodes"`
	Edges map[string]MemoryEdge `json:"edges"`
}

func NewGraph() *Graph {
	return &Graph{
		Nodes: make(map[string]MemoryNode),
		Edges: make(map[string]MemoryEdge),
	}
}

func (g *Graph) Clone() *Graph {
	out := NewGraph()
	for id, node := range g.Nodes {
		node.Embedding = append([]float64(nil), node.Embedding...)
		if node.Metadata != nil {
			node.Metadata = cloneStringMap(node.Metadata)
		}
		out.Nodes[id] = node
	}
	for id, edge := range g.Edges {
		out.Edges[id] = edge
	}
	return out
}

func cloneStringMap(in map[string]string) map[string]string {
	if in == nil {
		return nil
	}
	out := make(map[string]string, len(in))
	for key, value := range in {
		out[key] = value
	}
	return out
}
