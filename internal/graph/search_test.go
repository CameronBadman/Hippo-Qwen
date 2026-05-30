package graph

import "testing"

func TestSearchUsesGraphTraversalForConnectedContext(t *testing.T) {
	g := NewGraph()
	queryLike := MemoryNode{
		ID:         "n1",
		Text:       "User asked about headache medicine options.",
		Embedding:  EmbedText("headache medicine options", DefaultEmbeddingDims),
		Importance: 0.4,
	}
	critical := MemoryNode{
		ID:         "n2",
		Text:       "User's doctor said to avoid ibuprofen because of stomach bleeding risk.",
		Embedding:  EmbedText("doctor stomach bleeding risk", DefaultEmbeddingDims),
		Importance: 0.9,
	}
	unrelated := MemoryNode{
		ID:         "n3",
		Text:       "User likes mechanical keyboards with tactile switches.",
		Embedding:  EmbedText("mechanical keyboards tactile switches", DefaultEmbeddingDims),
		Importance: 0.3,
	}
	g.Nodes[queryLike.ID] = queryLike
	g.Nodes[critical.ID] = critical
	g.Nodes[unrelated.ID] = unrelated
	edge := normalizeEdge(MemoryEdge{
		Source: queryLike.ID,
		Target: critical.ID,
		Type:   "used_with",
		Weight: 1.1,
	})
	g.Edges[edge.ID] = edge
	idx := NewLinearIndex(DefaultEmbeddingDims)
	if err := idx.Rebuild(g.Nodes); err != nil {
		t.Fatal(err)
	}

	resp := Search(g, idx, SearchRequest{Query: "what should I take for a headache?", Limit: 3, Budget: 1000})
	if !containsNode(resp.GraphResults, critical.ID) {
		t.Fatalf("graph results did not include critical connected node: %#v", ids(resp.GraphResults))
	}
	if len(resp.ContextEdgeIDs) == 0 || resp.ContextEdgeIDs[0] != edge.ID {
		t.Fatalf("context edge ids = %#v, want %q", resp.ContextEdgeIDs, edge.ID)
	}
}

func containsNode(results []SearchResult, id string) bool {
	for _, result := range results {
		if result.Node.ID == id {
			return true
		}
	}
	return false
}

func ids(results []SearchResult) []string {
	out := make([]string, 0, len(results))
	for _, result := range results {
		out = append(out, result.Node.ID)
	}
	return out
}
