package graph

import "hash/fnv"

func ActivationMaskForText(text string) uint64 {
	var mask uint64
	for token := range tokenSet(text) {
		hash := fnv.New64a()
		_, _ = hash.Write([]byte(token))
		bit := hash.Sum64() % 64
		mask |= uint64(1) << bit
	}
	return mask
}

func ActivationMaskForEdge(source MemoryNode, target MemoryNode, edgeType string) uint64 {
	text := source.Text + " " + source.Summary + " " + source.Cluster + " " + target.Text + " " + target.Summary + " " + target.Cluster + " " + edgeType
	for key, value := range source.Metadata {
		text += " " + key + " " + value
	}
	for key, value := range target.Metadata {
		text += " " + key + " " + value
	}
	return ActivationMaskForText(text)
}

func ActivationOverlap(a uint64, b uint64) float64 {
	if a == 0 || b == 0 {
		return 0
	}
	intersection := popcount64(a & b)
	union := popcount64(a | b)
	if union == 0 {
		return 0
	}
	return float64(intersection) / float64(union)
}

func popcount64(value uint64) int {
	var count int
	for value != 0 {
		value &= value - 1
		count++
	}
	return count
}
