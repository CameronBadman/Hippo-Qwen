package main

import (
	"flag"
	"log"
	"net/http"
	"strings"

	"hippograph/internal/graph"
	"hippograph/internal/server"
)

func main() {
	addr := flag.String("addr", ":8080", "HTTP listen address")
	dataDir := flag.String("data-dir", "data/hippograph", "graph storage directory")
	webDir := flag.String("web-dir", "web", "static web directory")
	flag.Parse()

	store, err := graph.OpenStore(*dataDir)
	if err != nil {
		log.Fatalf("open store: %v", err)
	}
	app, err := server.New(server.Config{Store: store, WebDir: *webDir})
	if err != nil {
		log.Fatalf("server: %v", err)
	}
	log.Printf("HippoGraph listening on %s", displayURL(*addr))
	if err := http.ListenAndServe(*addr, app.Handler()); err != nil {
		log.Fatal(err)
	}
}

func displayURL(addr string) string {
	if strings.HasPrefix(addr, ":") {
		return "http://localhost" + addr
	}
	return "http://" + addr
}
