package server

import (
	"bytes"
	"encoding/json"
	"io"
)

func bodyFromJSON(value any) io.ReadCloser {
	payload, _ := json.Marshal(value)
	return io.NopCloser(bytes.NewReader(payload))
}
