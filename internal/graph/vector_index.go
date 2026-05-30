package graph

import (
	"errors"
	"sort"
	"sync"
)

type VectorHit struct {
	NodeID string  `json:"node_id"`
	Score  float64 `json:"score"`
}

type VectorIndex interface {
	Insert(node MemoryNode) error
	Search(embedding []float64, k int) []VectorHit
	Rebuild(nodes map[string]MemoryNode) error
	Len() int
}

type LinearIndex struct {
	mu    sync.RWMutex
	nodes map[string][]float64
	dims  int
}

func NewLinearIndex(dims int) *LinearIndex {
	if dims <= 0 {
		dims = DefaultEmbeddingDims
	}
	return &LinearIndex{
		nodes: make(map[string][]float64),
		dims:  dims,
	}
}

func (idx *LinearIndex) Insert(node MemoryNode) error {
	if node.ID == "" {
		return errors.New("node id is required")
	}
	if len(node.Embedding) != idx.dims {
		return errors.New("embedding dimension mismatch")
	}
	idx.mu.Lock()
	defer idx.mu.Unlock()
	idx.nodes[node.ID] = append([]float64(nil), node.Embedding...)
	return nil
}

func (idx *LinearIndex) Search(embedding []float64, k int) []VectorHit {
	idx.mu.RLock()
	defer idx.mu.RUnlock()
	if k <= 0 {
		k = len(idx.nodes)
	}
	hits := make([]VectorHit, 0, len(idx.nodes))
	for id, nodeEmbedding := range idx.nodes {
		hits = append(hits, VectorHit{
			NodeID: id,
			Score:  Cosine(embedding, nodeEmbedding),
		})
	}
	sort.Slice(hits, func(i, j int) bool {
		if hits[i].Score == hits[j].Score {
			return hits[i].NodeID < hits[j].NodeID
		}
		return hits[i].Score > hits[j].Score
	})
	if len(hits) > k {
		hits = hits[:k]
	}
	return hits
}

func (idx *LinearIndex) Rebuild(nodes map[string]MemoryNode) error {
	idx.mu.Lock()
	defer idx.mu.Unlock()
	idx.nodes = make(map[string][]float64, len(nodes))
	for id, node := range nodes {
		if len(node.Embedding) != idx.dims {
			return errors.New("embedding dimension mismatch")
		}
		idx.nodes[id] = append([]float64(nil), node.Embedding...)
	}
	return nil
}

func (idx *LinearIndex) Len() int {
	idx.mu.RLock()
	defer idx.mu.RUnlock()
	return len(idx.nodes)
}

type HNSWIndex struct{}

func NewHNSWIndex() *HNSWIndex {
	return &HNSWIndex{}
}

func (idx *HNSWIndex) Insert(MemoryNode) error {
	return errors.New("hnsw index is not implemented yet")
}

func (idx *HNSWIndex) Search([]float64, int) []VectorHit {
	return nil
}

func (idx *HNSWIndex) Rebuild(map[string]MemoryNode) error {
	return errors.New("hnsw index is not implemented yet")
}

func (idx *HNSWIndex) Len() int {
	return 0
}
