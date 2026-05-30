package server

import (
	"encoding/json"
	"errors"
	"log"
	"net/http"
	"strings"
	"time"

	"hippograph/internal/graph"
)

type Server struct {
	store     *graph.Store
	librarian graph.HeuristicLibrarian
	webDir    string
}

type Config struct {
	Store  *graph.Store
	WebDir string
}

func New(config Config) (*Server, error) {
	if config.Store == nil {
		return nil, errors.New("store is required")
	}
	return &Server{
		store:     config.Store,
		librarian: graph.NewHeuristicLibrarian(),
		webDir:    config.WebDir,
	}, nil
}

func (s *Server) Handler() http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("GET /healthz", s.handleHealth)
	mux.HandleFunc("GET /api/healthz", s.handleHealth)
	mux.HandleFunc("POST /memories", s.handleRemember)
	mux.HandleFunc("POST /api/memories", s.handleRemember)
	mux.HandleFunc("POST /search", s.handleSearch)
	mux.HandleFunc("POST /api/search", s.handleSearch)
	mux.HandleFunc("POST /feedback", s.handleFeedback)
	mux.HandleFunc("POST /api/feedback", s.handleFeedback)
	mux.HandleFunc("POST /maintenance/decay", s.handleDecay)
	mux.HandleFunc("POST /api/maintenance/decay", s.handleDecay)
	mux.HandleFunc("GET /graph", s.handleGraph)
	mux.HandleFunc("GET /api/graph", s.handleGraph)
	mux.HandleFunc("GET /tools/list", s.handleToolList)
	mux.HandleFunc("GET /api/tools/list", s.handleToolList)
	mux.HandleFunc("POST /tools/call", s.handleToolCall)
	mux.HandleFunc("POST /api/tools/call", s.handleToolCall)
	if s.webDir != "" {
		mux.Handle("/", http.FileServer(http.Dir(s.webDir)))
	}
	return withLogging(withJSON(mux))
}

type rememberRequest struct {
	ID         string            `json:"id,omitempty"`
	Text       string            `json:"text"`
	Summary    string            `json:"summary,omitempty"`
	Source     string            `json:"source,omitempty"`
	Importance float64           `json:"importance,omitempty"`
	Cluster    string            `json:"cluster,omitempty"`
	Metadata   map[string]string `json:"metadata,omitempty"`
}

type rememberResponse struct {
	Node       graph.MemoryNode        `json:"node"`
	Edges      []graph.MemoryEdge      `json:"edges"`
	Librarian  graph.LibrarianResponse `json:"librarian"`
	Candidates []graph.MemoryNode      `json:"candidates"`
}

func (s *Server) handleHealth(w http.ResponseWriter, _ *http.Request) {
	writeJSON(w, http.StatusOK, map[string]string{"status": "ok"})
}

func (s *Server) handleRemember(w http.ResponseWriter, r *http.Request) {
	var req rememberRequest
	if err := decodeJSON(r, &req); err != nil {
		writeError(w, http.StatusBadRequest, err)
		return
	}
	if strings.TrimSpace(req.Text) == "" {
		writeError(w, http.StatusBadRequest, errors.New("text is required"))
		return
	}
	before := s.store.Graph()
	node := graph.MemoryNode{
		ID:         req.ID,
		Text:       strings.TrimSpace(req.Text),
		Summary:    strings.TrimSpace(req.Summary),
		Embedding:  graph.EmbedText(req.Text+" "+req.Summary, graph.DefaultEmbeddingDims),
		Timestamp:  time.Now().UTC(),
		Source:     req.Source,
		Importance: req.Importance,
		Cluster:    defaultCluster(req.Cluster, req.Metadata),
		Metadata:   req.Metadata,
	}
	stored, err := s.store.AddNode(node)
	if err != nil {
		writeError(w, http.StatusBadRequest, err)
		return
	}
	candidates := graph.TopCandidates(before, stored.Embedding, stored.ID, 32)
	librarianResponse := s.librarian.Place(graph.LibrarianRequest{
		Anchor:     stored,
		Candidates: candidates,
		Graph:      before,
		Now:        time.Now().UTC(),
	})
	edges := s.librarian.ActionsToEdges(stored, librarianResponse)
	created := make([]graph.MemoryEdge, 0, len(edges))
	for _, edge := range edges {
		upserted, err := s.store.UpsertEdge(edge)
		if err != nil {
			writeError(w, http.StatusInternalServerError, err)
			return
		}
		created = append(created, upserted)
	}
	_ = s.store.SaveSnapshot()
	writeJSON(w, http.StatusOK, rememberResponse{
		Node:       stored,
		Edges:      created,
		Librarian:  librarianResponse,
		Candidates: candidates,
	})
}

func (s *Server) handleSearch(w http.ResponseWriter, r *http.Request) {
	var req graph.SearchRequest
	if err := decodeJSON(r, &req); err != nil {
		writeError(w, http.StatusBadRequest, err)
		return
	}
	if strings.TrimSpace(req.Query) == "" {
		writeError(w, http.StatusBadRequest, errors.New("query is required"))
		return
	}
	resp := graph.Search(s.store.Graph(), req)
	writeJSON(w, http.StatusOK, resp)
}

func (s *Server) handleFeedback(w http.ResponseWriter, r *http.Request) {
	var feedback graph.Feedback
	if err := decodeJSON(r, &feedback); err != nil {
		writeError(w, http.StatusBadRequest, err)
		return
	}
	if feedback.Outcome == "" {
		writeError(w, http.StatusBadRequest, errors.New("outcome is required"))
		return
	}
	if err := s.store.ApplyFeedback(feedback); err != nil {
		writeError(w, http.StatusBadRequest, err)
		return
	}
	_ = s.store.SaveSnapshot()
	writeJSON(w, http.StatusOK, map[string]any{"ok": true})
}

type decayRequest struct {
	Factor      float64 `json:"factor,omitempty"`
	GCThreshold float64 `json:"gc_threshold,omitempty"`
}

func (s *Server) handleDecay(w http.ResponseWriter, r *http.Request) {
	var req decayRequest
	if err := decodeJSON(r, &req); err != nil {
		writeError(w, http.StatusBadRequest, err)
		return
	}
	if req.Factor == 0 {
		req.Factor = 0.98
	}
	if req.GCThreshold == 0 {
		req.GCThreshold = 0.05
	}
	removed, err := s.store.DecayAndGC(req.Factor, req.GCThreshold)
	if err != nil {
		writeError(w, http.StatusBadRequest, err)
		return
	}
	_ = s.store.SaveSnapshot()
	writeJSON(w, http.StatusOK, map[string]any{"removed": removed, "removed_count": len(removed)})
}

func (s *Server) handleGraph(w http.ResponseWriter, _ *http.Request) {
	writeJSON(w, http.StatusOK, s.store.Graph())
}

type toolDefinition struct {
	Name        string         `json:"name"`
	Description string         `json:"description"`
	Schema      map[string]any `json:"schema"`
}

func (s *Server) handleToolList(w http.ResponseWriter, _ *http.Request) {
	writeJSON(w, http.StatusOK, map[string]any{"tools": []toolDefinition{
		{Name: "remember", Description: "Store a memory and place it in the graph."},
		{Name: "search_memory", Description: "Search graph memory and return compact context."},
		{Name: "explain_memory_path", Description: "Return graph nodes and edges for visualization."},
		{Name: "give_feedback", Description: "Update edge weights from helpful, ignored, or corrected feedback."},
	}})
}

type toolCallRequest struct {
	Tool      string          `json:"tool"`
	Arguments json.RawMessage `json:"arguments"`
}

func (s *Server) handleToolCall(w http.ResponseWriter, r *http.Request) {
	var call toolCallRequest
	if err := decodeJSON(r, &call); err != nil {
		writeError(w, http.StatusBadRequest, err)
		return
	}
	switch call.Tool {
	case "remember":
		var req rememberRequest
		if err := json.Unmarshal(call.Arguments, &req); err != nil {
			writeError(w, http.StatusBadRequest, err)
			return
		}
		r2 := r.Clone(r.Context())
		r2.Body = bodyFromJSON(req)
		s.handleRemember(w, r2)
	case "search_memory":
		var req graph.SearchRequest
		if err := json.Unmarshal(call.Arguments, &req); err != nil {
			writeError(w, http.StatusBadRequest, err)
			return
		}
		r2 := r.Clone(r.Context())
		r2.Body = bodyFromJSON(req)
		s.handleSearch(w, r2)
	case "explain_memory_path":
		s.handleGraph(w, r)
	case "give_feedback":
		var req graph.Feedback
		if err := json.Unmarshal(call.Arguments, &req); err != nil {
			writeError(w, http.StatusBadRequest, err)
			return
		}
		r2 := r.Clone(r.Context())
		r2.Body = bodyFromJSON(req)
		s.handleFeedback(w, r2)
	default:
		writeError(w, http.StatusBadRequest, errors.New("unknown tool"))
	}
}

func defaultCluster(cluster string, metadata map[string]string) string {
	if cluster != "" {
		return cluster
	}
	for _, key := range []string{"project", "topic", "user"} {
		if metadata != nil && metadata[key] != "" {
			return metadata[key]
		}
	}
	return ""
}

func decodeJSON(r *http.Request, dst any) error {
	defer r.Body.Close()
	decoder := json.NewDecoder(r.Body)
	decoder.DisallowUnknownFields()
	return decoder.Decode(dst)
}

func writeJSON(w http.ResponseWriter, status int, payload any) {
	w.WriteHeader(status)
	if err := json.NewEncoder(w).Encode(payload); err != nil {
		log.Printf("write json: %v", err)
	}
}

func writeError(w http.ResponseWriter, status int, err error) {
	writeJSON(w, status, map[string]string{"error": err.Error()})
}

func withJSON(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Access-Control-Allow-Origin", "*")
		w.Header().Set("Access-Control-Allow-Headers", "Content-Type")
		w.Header().Set("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
		if r.Method == http.MethodOptions {
			w.WriteHeader(http.StatusNoContent)
			return
		}
		if strings.HasPrefix(r.URL.Path, "/") && !strings.Contains(r.URL.Path, ".") {
			w.Header().Set("Content-Type", "application/json")
		}
		next.ServeHTTP(w, r)
	})
}

func withLogging(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		start := time.Now()
		next.ServeHTTP(w, r)
		log.Printf("%s %s %s", r.Method, r.URL.Path, time.Since(start).Round(time.Millisecond))
	})
}
