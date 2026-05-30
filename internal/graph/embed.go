package graph

import (
	"hash/fnv"
	"math"
	"regexp"
	"strings"
)

const DefaultEmbeddingDims = 64

var tokenPattern = regexp.MustCompile(`[a-z0-9]+`)

func EmbedText(text string, dims int) []float64 {
	if dims <= 0 {
		dims = DefaultEmbeddingDims
	}
	vec := make([]float64, dims)
	tokens := tokenPattern.FindAllString(strings.ToLower(text), -1)
	for _, token := range tokens {
		hash := fnv.New64a()
		_, _ = hash.Write([]byte(token))
		sum := hash.Sum64()
		idx := int(sum % uint64(dims))
		sign := 1.0
		if (sum>>8)&1 == 1 {
			sign = -1
		}
		vec[idx] += sign
	}
	normalize(vec)
	return vec
}

func Cosine(a []float64, b []float64) float64 {
	if len(a) == 0 || len(a) != len(b) {
		return 0
	}
	var dot float64
	for i := range a {
		dot += a[i] * b[i]
	}
	return dot
}

func TokenJaccard(a string, b string) float64 {
	left := tokenSet(a)
	right := tokenSet(b)
	if len(left) == 0 || len(right) == 0 {
		return 0
	}
	var intersection int
	for token := range left {
		if right[token] {
			intersection++
		}
	}
	union := len(left) + len(right) - intersection
	if union == 0 {
		return 0
	}
	return float64(intersection) / float64(union)
}

func tokenSet(text string) map[string]bool {
	tokens := tokenPattern.FindAllString(strings.ToLower(text), -1)
	out := make(map[string]bool, len(tokens))
	for _, token := range tokens {
		if len(token) > 1 {
			out[token] = true
		}
	}
	return out
}

func normalize(vec []float64) {
	var sum float64
	for _, value := range vec {
		sum += value * value
	}
	norm := math.Sqrt(sum)
	if norm == 0 {
		return
	}
	for i := range vec {
		vec[i] /= norm
	}
}
