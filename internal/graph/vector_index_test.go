package graph

import "testing"

func TestLinearIndexSearchAndRebuild(t *testing.T) {
	nodes := map[string]MemoryNode{
		"a": {ID: "a", Text: "alpha", Embedding: EmbedText("alpha beta", DefaultEmbeddingDims)},
		"b": {ID: "b", Text: "gamma", Embedding: EmbedText("gamma delta", DefaultEmbeddingDims)},
	}
	idx := NewLinearIndex(DefaultEmbeddingDims)
	if err := idx.Rebuild(nodes); err != nil {
		t.Fatal(err)
	}
	if idx.Len() != 2 {
		t.Fatalf("len = %d, want 2", idx.Len())
	}
	hits := idx.Search(EmbedText("alpha", DefaultEmbeddingDims), 1)
	if len(hits) != 1 {
		t.Fatalf("hits = %d, want 1", len(hits))
	}
	if hits[0].NodeID != "a" {
		t.Fatalf("top hit = %q, want a", hits[0].NodeID)
	}

	node := MemoryNode{ID: "c", Text: "alpha project", Embedding: EmbedText("alpha project", DefaultEmbeddingDims)}
	if err := idx.Insert(node); err != nil {
		t.Fatal(err)
	}
	if idx.Len() != 3 {
		t.Fatalf("len after insert = %d, want 3", idx.Len())
	}
}
