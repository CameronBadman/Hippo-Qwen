package graph

import "testing"

func TestActivationMaskOverlap(t *testing.T) {
	left := ActivationMaskForText("hippograph graph memory runtime")
	right := ActivationMaskForText("graph memory search")
	other := ActivationMaskForText("cooking recipe dinner")

	if left == 0 || right == 0 {
		t.Fatalf("expected non-empty activation masks")
	}
	if ActivationOverlap(left, right) <= ActivationOverlap(left, other) {
		t.Fatalf("related masks should overlap more than unrelated masks")
	}
}

func TestActionsIncludeCompactRoutingFeatures(t *testing.T) {
	anchor := MemoryNode{
		ID:        "a",
		Text:      "User prefers concise Go answers for HippoGraph.",
		Embedding: EmbedText("concise Go answers HippoGraph", DefaultEmbeddingDims),
		Cluster:   "hippograph",
		Metadata:  map[string]string{"project": "hippograph"},
	}
	candidate := MemoryNode{
		ID:        "b",
		Text:      "HippoGraph runtime uses a Go graph store.",
		Embedding: EmbedText("Go graph store HippoGraph", DefaultEmbeddingDims),
		Cluster:   "hippograph",
		Metadata:  map[string]string{"project": "hippograph"},
	}
	response := NewHeuristicLibrarian().Place(LibrarianRequest{
		Anchor:     anchor,
		Candidates: []MemoryNode{candidate},
	})
	if len(response.Actions) != 1 {
		t.Fatalf("actions = %d, want 1", len(response.Actions))
	}
	action := response.Actions[0]
	if action.Confidence == 0 {
		t.Fatalf("expected action confidence")
	}
	if action.ActivationMask == 0 {
		t.Fatalf("expected action activation mask")
	}
	edges := NewHeuristicLibrarian().ActionsToEdges(anchor, response)
	if len(edges) == 0 || edges[0].ActivationMask == 0 {
		t.Fatalf("expected edge activation mask")
	}
}
