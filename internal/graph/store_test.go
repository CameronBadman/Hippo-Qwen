package graph

import (
	"testing"
)

func TestStoreReplaySnapshotAndFeedback(t *testing.T) {
	dir := t.TempDir()
	store, err := OpenStore(dir)
	if err != nil {
		t.Fatal(err)
	}
	first, err := store.AddNode(MemoryNode{Text: "User prefers concise Go implementations.", Importance: 0.7})
	if err != nil {
		t.Fatal(err)
	}
	second, err := store.AddNode(MemoryNode{Text: "HippoGraph runtime is implemented in Go.", Importance: 0.6})
	if err != nil {
		t.Fatal(err)
	}
	edge, err := store.UpsertEdge(MemoryEdge{
		Source: first.ID,
		Target: second.ID,
		Type:   "same_context",
		Weight: 0.4,
	})
	if err != nil {
		t.Fatal(err)
	}
	if err := store.ApplyFeedback(Feedback{Outcome: "helpful", NodeIDs: []string{second.ID}, EdgeIDs: []string{edge.ID}}); err != nil {
		t.Fatal(err)
	}
	if err := store.SaveSnapshot(); err != nil {
		t.Fatal(err)
	}

	reopened, err := OpenStore(dir)
	if err != nil {
		t.Fatal(err)
	}
	g := reopened.Graph()
	if len(g.Nodes) != 2 {
		t.Fatalf("nodes = %d, want 2", len(g.Nodes))
	}
	got := g.Edges[edge.ID]
	if got.Weight <= edge.Weight {
		t.Fatalf("feedback did not strengthen edge: got %.3f <= %.3f", got.Weight, edge.Weight)
	}
	if g.Nodes[second.ID].Importance <= second.Importance {
		t.Fatalf("feedback did not raise node importance")
	}
}

func TestDecayAndGCRemovesWeakUnprotectedEdges(t *testing.T) {
	store, err := OpenStore(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	a, _ := store.AddNode(MemoryNode{Text: "A"})
	b, _ := store.AddNode(MemoryNode{Text: "B"})
	edge, err := store.UpsertEdge(MemoryEdge{Source: a.ID, Target: b.ID, Type: "used_with", Weight: 0.04, DecayRate: 0.1})
	if err != nil {
		t.Fatal(err)
	}
	removed, err := store.DecayAndGC(0.9, 0.05)
	if err != nil {
		t.Fatal(err)
	}
	if len(removed) != 1 || removed[0] != edge.ID {
		t.Fatalf("removed = %#v, want %q", removed, edge.ID)
	}
	if _, ok := store.Graph().Edges[edge.ID]; ok {
		t.Fatalf("edge %q still present after GC", edge.ID)
	}
}
