package graph

import "sort"

type SearchRequest struct {
	Query  string `json:"query"`
	Limit  int    `json:"limit,omitempty"`
	Budget int    `json:"budget,omitempty"`
}

type SearchResult struct {
	Node        MemoryNode `json:"node"`
	Score       float64    `json:"score"`
	VectorScore float64    `json:"vector_score"`
	GraphScore  float64    `json:"graph_score"`
	Path        []string   `json:"path,omitempty"`
	EdgeIDs     []string   `json:"edge_ids,omitempty"`
}

type SearchResponse struct {
	Query          string         `json:"query"`
	VectorResults  []SearchResult `json:"vector_results"`
	GraphResults   []SearchResult `json:"graph_results"`
	Context        string         `json:"context"`
	ContextNodeIDs []string       `json:"context_node_ids"`
	ContextEdgeIDs []string       `json:"context_edge_ids"`
}

func Search(g *Graph, idx VectorIndex, req SearchRequest) SearchResponse {
	if req.Limit <= 0 {
		req.Limit = 10
	}
	if req.Budget <= 0 {
		req.Budget = 2400
	}
	queryEmbedding := EmbedText(req.Query, DefaultEmbeddingDims)
	vector := vectorRank(g, idx, queryEmbedding, maxInt(req.Limit, 12))
	seeds := vector
	if len(seeds) > 8 {
		seeds = seeds[:8]
	}
	graphResults := graphRank(g, seeds)
	if len(vector) > req.Limit {
		vector = vector[:req.Limit]
	}
	if len(graphResults) > req.Limit {
		graphResults = graphResults[:req.Limit]
	}
	context, nodeIDs, edgeIDs := packContext(graphResults, req.Budget)
	return SearchResponse{
		Query:          req.Query,
		VectorResults:  vector,
		GraphResults:   graphResults,
		Context:        context,
		ContextNodeIDs: nodeIDs,
		ContextEdgeIDs: edgeIDs,
	}
}

func TopCandidates(g *Graph, idx VectorIndex, embedding []float64, excludeID string, limit int) []MemoryNode {
	ranked := vectorRank(g, idx, embedding, limit+1)
	out := make([]MemoryNode, 0, limit)
	for _, result := range ranked {
		if result.Node.ID == excludeID {
			continue
		}
		out = append(out, result.Node)
		if len(out) >= limit {
			break
		}
	}
	return out
}

func vectorRank(g *Graph, idx VectorIndex, queryEmbedding []float64, limit int) []SearchResult {
	hits := idx.Search(queryEmbedding, limit)
	results := make([]SearchResult, 0, len(hits))
	for _, hit := range hits {
		node, ok := g.Nodes[hit.NodeID]
		if !ok {
			continue
		}
		results = append(results, SearchResult{
			Node:        node,
			Score:       hit.Score,
			VectorScore: hit.Score,
		})
	}
	sortResults(results)
	return results
}

func graphRank(g *Graph, seeds []SearchResult) []SearchResult {
	byNode := make(map[string]SearchResult)
	for _, seed := range seeds {
		seed.GraphScore = seed.VectorScore * 0.35
		seed.Score = seed.VectorScore + seed.GraphScore + seed.Node.Importance*0.05
		byNode[seed.Node.ID] = seed
	}
	for _, seed := range seeds {
		expandFrom(g, byNode, seed, 2)
	}
	results := make([]SearchResult, 0, len(byNode))
	for _, result := range byNode {
		results = append(results, result)
	}
	sortResults(results)
	return results
}

func expandFrom(g *Graph, byNode map[string]SearchResult, seed SearchResult, depth int) {
	frontier := []SearchResult{seed}
	for step := 0; step < depth; step++ {
		var next []SearchResult
		for _, current := range frontier {
			for _, edge := range g.Edges {
				if edge.Source != current.Node.ID {
					continue
				}
				target, ok := g.Nodes[edge.Target]
				if !ok {
					continue
				}
				if target.ID == seed.Node.ID || pathContains(current.Path, target.ID) {
					continue
				}
				graphScore := current.GraphScore + edge.Weight*edgeTypeBoost(edge.Type)/(float64(step)+1.5)
				score := seed.VectorScore*0.55 + graphScore + target.Importance*0.08
				path := append(append([]string(nil), current.Path...), current.Node.ID, target.ID)
				edgeIDs := append(append([]string(nil), current.EdgeIDs...), edge.ID)
				existing, seen := byNode[target.ID]
				if !seen || score > existing.Score {
					result := SearchResult{
						Node:        target,
						Score:       score,
						VectorScore: existing.VectorScore,
						GraphScore:  graphScore,
						Path:        compactPath(path),
						EdgeIDs:     edgeIDs,
					}
					byNode[target.ID] = result
					next = append(next, result)
				}
			}
		}
		frontier = next
	}
}

func pathContains(path []string, id string) bool {
	for _, item := range path {
		if item == id {
			return true
		}
	}
	return false
}

func edgeTypeBoost(edgeType string) float64 {
	switch edgeType {
	case "preference", "correction":
		return 1.25
	case "same_context", "same_cluster":
		return 1.1
	case "temporal_next":
		return 0.85
	default:
		return 1
	}
}

func packContext(results []SearchResult, budget int) (string, []string, []string) {
	var context string
	var nodeIDs []string
	edgeSeen := map[string]bool{}
	var edgeIDs []string
	for _, result := range results {
		text := "- " + result.Node.Text + "\n"
		if len(context)+len(text) > budget {
			break
		}
		context += text
		nodeIDs = append(nodeIDs, result.Node.ID)
		for _, id := range result.EdgeIDs {
			if !edgeSeen[id] {
				edgeSeen[id] = true
				edgeIDs = append(edgeIDs, id)
			}
		}
	}
	return context, nodeIDs, edgeIDs
}

func sortResults(results []SearchResult) {
	sortSearchResults(results)
}

func sortSearchResults(results []SearchResult) {
	sort.Slice(results, func(i, j int) bool {
		if results[i].Score == results[j].Score {
			return results[i].Node.ID < results[j].Node.ID
		}
		return results[i].Score > results[j].Score
	})
}

func compactPath(path []string) []string {
	if len(path) == 0 {
		return nil
	}
	out := make([]string, 0, len(path))
	last := ""
	for _, id := range path {
		if id == "" || id == last {
			continue
		}
		out = append(out, id)
		last = id
	}
	return out
}

func maxInt(a int, b int) int {
	if a > b {
		return a
	}
	return b
}
