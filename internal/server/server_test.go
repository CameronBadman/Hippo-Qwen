package server

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"hippograph/internal/graph"
)

func TestRememberSearchAndFeedbackAPI(t *testing.T) {
	store, err := graph.OpenStore(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	app, err := New(Config{Store: store})
	if err != nil {
		t.Fatal(err)
	}
	handler := app.Handler()

	postJSON(t, handler, "/memories", map[string]any{
		"text":       "User prefers short answers for Go runtime work.",
		"importance": 0.8,
		"metadata":   map[string]string{"project": "hippograph"},
	}, http.StatusOK)
	var second rememberResponse
	postJSONInto(t, handler, "/memories", map[string]any{
		"text":       "HippoGraph stores memory as weighted graph edges.",
		"importance": 0.7,
		"metadata":   map[string]string{"project": "hippograph"},
	}, http.StatusOK, &second)
	if len(second.Edges) == 0 {
		t.Fatalf("expected heuristic librarian to create same-project edges")
	}

	var search graph.SearchResponse
	postJSONInto(t, handler, "/search", map[string]any{
		"query":  "what does the user prefer for Go runtime work?",
		"limit":  5,
		"budget": 900,
	}, http.StatusOK, &search)
	if len(search.GraphResults) == 0 {
		t.Fatalf("expected graph results")
	}
	postJSON(t, handler, "/feedback", map[string]any{
		"outcome":  "helpful",
		"node_ids": search.ContextNodeIDs,
		"edge_ids": search.ContextEdgeIDs,
	}, http.StatusOK)

	req := httptest.NewRequest(http.MethodGet, "/tools/list", nil)
	rec := httptest.NewRecorder()
	handler.ServeHTTP(rec, req)
	if rec.Code != http.StatusOK {
		t.Fatalf("tools/list status = %d", rec.Code)
	}
}

func postJSON(t *testing.T, handler http.Handler, path string, payload any, wantStatus int) {
	t.Helper()
	postJSONInto(t, handler, path, payload, wantStatus, nil)
}

func postJSONInto(t *testing.T, handler http.Handler, path string, payload any, wantStatus int, out any) {
	t.Helper()
	body, err := json.Marshal(payload)
	if err != nil {
		t.Fatal(err)
	}
	req := httptest.NewRequest(http.MethodPost, path, bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()
	handler.ServeHTTP(rec, req)
	if rec.Code != wantStatus {
		t.Fatalf("%s status = %d, want %d, body=%s", path, rec.Code, wantStatus, rec.Body.String())
	}
	if out != nil {
		if err := json.Unmarshal(rec.Body.Bytes(), out); err != nil {
			t.Fatal(err)
		}
	}
}
