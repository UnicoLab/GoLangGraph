// Copyright (c) 2024 GoLangGraph Team
//
// Licensed under the MIT License. See LICENSE file in the project root for full license information.
//
// Package: GoLangGraph - A powerful Go framework for building AI agent workflows

// Copyright (c) 2024 GoLangGraph Team
//
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

package server

import (
	"context"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/UnicoLab/GoLangGraph/pkg/core"
	"github.com/gorilla/websocket"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// wsTestServer starts a real HTTP server so WebSocket upgrades exercise the
// full path, not just the handler.
func wsTestServer(t *testing.T, mutate func(*ServerConfig)) (*Server, *httptest.Server) {
	t.Helper()
	s := newTestServer(t, mutate)
	ts := httptest.NewServer(s.router)
	t.Cleanup(ts.Close)
	return s, ts
}

func wsURL(ts *httptest.Server, path string) string {
	return "ws" + strings.TrimPrefix(ts.URL, "http") + path
}

func dialWS(t *testing.T, ts *httptest.Server, path string, headers http.Header) *websocket.Conn {
	t.Helper()
	conn, resp, err := websocket.DefaultDialer.Dial(wsURL(ts, path), headers)
	if err != nil {
		if resp != nil {
			t.Fatalf("dial %s: %v (status %s)", path, err, resp.Status)
		}
		t.Fatalf("dial %s: %v", path, err)
	}
	t.Cleanup(func() { _ = conn.Close() })
	return conn
}

// readMessages collects frames until a terminal type arrives or the deadline hits.
func readMessages(t *testing.T, conn *websocket.Conn, terminal map[string]bool, timeout time.Duration) []map[string]interface{} {
	t.Helper()
	require.NoError(t, conn.SetReadDeadline(time.Now().Add(timeout)))

	var messages []map[string]interface{}
	for {
		var msg map[string]interface{}
		if err := conn.ReadJSON(&msg); err != nil {
			return messages
		}
		messages = append(messages, msg)
		if kind, _ := msg["type"].(string); terminal[kind] {
			return messages
		}
	}
}

func TestWebSocket_GraphExecutionStreamsEveryStep(t *testing.T) {
	_, ts := wsTestServer(t, nil)
	conn := dialWS(t, ts, "/api/v1/ws/graphs/demo/stream", nil)

	require.NoError(t, conn.WriteJSON(map[string]interface{}{
		"type": "execute", "input": "world",
	}))

	messages := readMessages(t, conn, map[string]bool{"complete": true, "error": true}, 10*time.Second)
	require.NotEmpty(t, messages)

	var kinds []string
	var steps []string
	var final map[string]interface{}
	for _, m := range messages {
		kind, _ := m["type"].(string)
		kinds = append(kinds, kind)
		switch kind {
		case "step":
			step := m["step"].(map[string]interface{})
			steps = append(steps, step["node_id"].(string))
		case "complete":
			final, _ = m["state"].(map[string]interface{})
		}
	}

	assert.Equal(t, "start", kinds[0])
	assert.Equal(t, "complete", kinds[len(kinds)-1],
		"a run must finish with a terminal frame, got %v", kinds)
	assert.Equal(t, []string{"greet", "finish"}, steps,
		"every executed node must be streamed, in order")
	require.NotNil(t, final)
	assert.Equal(t, "hello world", final["greeting"])
}

func TestWebSocket_UnknownGraphReportsError(t *testing.T) {
	_, ts := wsTestServer(t, nil)
	conn := dialWS(t, ts, "/api/v1/ws/graphs/nope/stream", nil)

	require.NoError(t, conn.WriteJSON(map[string]interface{}{"type": "execute"}))
	messages := readMessages(t, conn, map[string]bool{"error": true}, 5*time.Second)
	require.NotEmpty(t, messages)
	assert.Equal(t, "error", messages[len(messages)-1]["type"])
	assert.Contains(t, messages[len(messages)-1]["error"], "not found")
}

func TestWebSocket_UnknownMessageTypeIsReported(t *testing.T) {
	_, ts := wsTestServer(t, nil)
	conn := dialWS(t, ts, "/api/v1/ws/graphs/demo/stream", nil)

	require.NoError(t, conn.WriteJSON(map[string]interface{}{"type": "not-a-command"}))
	messages := readMessages(t, conn, map[string]bool{"error": true}, 5*time.Second)
	require.NotEmpty(t, messages)
	assert.Contains(t, messages[0]["error"], "unknown message type")
}

func TestWebSocket_PingPong(t *testing.T) {
	_, ts := wsTestServer(t, nil)
	conn := dialWS(t, ts, "/api/v1/ws/graphs/demo/stream", nil)

	require.NoError(t, conn.WriteJSON(map[string]interface{}{"type": "ping"}))
	messages := readMessages(t, conn, map[string]bool{"pong": true}, 5*time.Second)
	require.NotEmpty(t, messages)
	assert.Equal(t, "pong", messages[0]["type"])
}

// Malformed frames must not kill the server or leak goroutines.
func TestWebSocket_MalformedFramesAreSurvived(t *testing.T) {
	_, ts := wsTestServer(t, nil)
	conn := dialWS(t, ts, "/api/v1/ws/graphs/demo/stream", nil)

	// A single unusable frame must be answered with an error, not a disconnect.
	require.NoError(t, conn.WriteMessage(websocket.TextMessage, []byte("{not json")))
	messages := readMessages(t, conn, map[string]bool{"error": true}, 5*time.Second)
	require.NotEmpty(t, messages)
	assert.Contains(t, messages[0]["error"], "invalid message")

	// A field of the wrong type is equally recoverable.
	require.NoError(t, conn.WriteJSON(map[string]interface{}{"type": "execute", "input": 12345}))
	messages = readMessages(t, conn, map[string]bool{"error": true}, 5*time.Second)
	require.NotEmpty(t, messages)

	// The same session still works afterwards.
	require.NoError(t, conn.WriteJSON(map[string]interface{}{"type": "ping"}))
	messages = readMessages(t, conn, map[string]bool{"pong": true}, 5*time.Second)
	require.NotEmpty(t, messages, "a malformed frame must not end the session")
}

// The origin allowlist must apply to WebSocket upgrades.
func TestWebSocket_RejectsDisallowedOrigin(t *testing.T) {
	_, ts := wsTestServer(t, func(c *ServerConfig) {
		c.Security.AllowedOrigins = []string{"https://studio.example.com"}
	})

	headers := http.Header{}
	headers.Set("Origin", "https://evil.example.com")
	_, resp, err := websocket.DefaultDialer.Dial(wsURL(ts, "/api/v1/ws/graphs/demo/stream"), headers)
	require.Error(t, err, "a disallowed origin must not complete the upgrade")
	if resp != nil {
		assert.Equal(t, http.StatusForbidden, resp.StatusCode)
	}
}

// Several clients may watch the same graph; one disconnecting must not evict
// the others from the connection registry.
func TestWebSocket_MultipleClientsPerGraph(t *testing.T) {
	s, ts := wsTestServer(t, nil)

	first := dialWS(t, ts, "/api/v1/ws/graphs/demo/stream", nil)
	second := dialWS(t, ts, "/api/v1/ws/graphs/demo/stream", nil)

	// Both must be registered.
	require.Eventually(t, func() bool { return s.wsConnectionCount("demo") == 2 },
		5*time.Second, 20*time.Millisecond, "both clients must be tracked")

	require.NoError(t, first.Close())
	require.Eventually(t, func() bool { return s.wsConnectionCount("demo") == 1 },
		5*time.Second, 20*time.Millisecond, "closing one client must not drop the other")

	// The survivor still works.
	require.NoError(t, second.WriteJSON(map[string]interface{}{"type": "ping"}))
	messages := readMessages(t, second, map[string]bool{"pong": true}, 5*time.Second)
	require.NotEmpty(t, messages)
}

// Concurrent executions on one connection must not interleave frames. Without
// serialized writes gorilla/websocket corrupts the stream.
func TestWebSocket_ConcurrentExecutionsDoNotCorruptStream(t *testing.T) {
	s, ts := wsTestServer(t, nil)

	// A graph slow enough that several runs overlap.
	slow := core.NewGraph("slow-stream")
	for i := 0; i < 6; i++ {
		id := fmt.Sprintf("n%d", i)
		slow.AddNode(id, id, func(ctx context.Context, st *core.BaseState) (*core.BaseState, error) {
			time.Sleep(5 * time.Millisecond)
			return st, nil
		})
		if i > 0 {
			slow.AddEdge(fmt.Sprintf("n%d", i-1), id, nil)
		}
	}
	require.NoError(t, slow.SetStartNode("n0"))
	require.NoError(t, slow.AddEndNode("n5"))
	s.GraphManager().Register("slow-stream", slow)

	conn := dialWS(t, ts, "/api/v1/ws/graphs/slow-stream/stream", nil)

	const runs = 4
	for i := 0; i < runs; i++ {
		require.NoError(t, conn.WriteJSON(map[string]interface{}{"type": "execute", "input": fmt.Sprintf("run-%d", i)}))
	}

	require.NoError(t, conn.SetReadDeadline(time.Now().Add(30*time.Second)))
	completes := 0
	frames := 0
	for completes < runs {
		var msg map[string]interface{}
		// A corrupted stream surfaces here as a JSON decode failure.
		require.NoError(t, conn.ReadJSON(&msg), "stream corrupted after %d frames", frames)
		frames++
		if kind, _ := msg["type"].(string); kind == "complete" {
			completes++
		}
	}
	assert.Equal(t, runs, completes)
}

// Closing the connection must cancel the run instead of leaving it going.
func TestWebSocket_DisconnectCancelsExecution(t *testing.T) {
	s, ts := wsTestServer(t, nil)

	started := make(chan struct{})
	finished := make(chan error, 1)
	var once sync.Once

	blocking := core.NewGraph("blocking")
	blocking.AddNode("wait", "Wait", func(ctx context.Context, st *core.BaseState) (*core.BaseState, error) {
		once.Do(func() { close(started) })
		<-ctx.Done()
		finished <- ctx.Err()
		return nil, ctx.Err()
	})
	require.NoError(t, blocking.SetStartNode("wait"))
	s.GraphManager().Register("blocking", blocking)

	conn := dialWS(t, ts, "/api/v1/ws/graphs/blocking/stream", nil)
	require.NoError(t, conn.WriteJSON(map[string]interface{}{"type": "execute"}))

	select {
	case <-started:
	case <-time.After(10 * time.Second):
		t.Fatal("graph never started")
	}

	require.NoError(t, conn.Close())

	select {
	case err := <-finished:
		assert.ErrorIs(t, err, context.Canceled,
			"a disconnected client must cancel its run rather than leaving it running")
	case <-time.After(15 * time.Second):
		t.Fatal("run was not canceled when the client disconnected")
	}
}

// Server shutdown must close live WebSocket connections rather than hanging.
func TestWebSocket_ShutdownClosesConnections(t *testing.T) {
	s := newTestServer(t, nil)
	ts := httptest.NewServer(s.router)
	defer ts.Close()

	conn := dialWS(t, ts, "/api/v1/ws/graphs/demo/stream", nil)
	require.NoError(t, conn.WriteJSON(map[string]interface{}{"type": "ping"}))
	require.NotEmpty(t, readMessages(t, conn, map[string]bool{"pong": true}, 5*time.Second))

	require.Eventually(t, func() bool { return s.wsConnectionCount("demo") == 1 },
		5*time.Second, 20*time.Millisecond)

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	require.NoError(t, s.Stop(ctx))

	assert.Zero(t, s.wsConnectionCount("demo"), "shutdown must release tracked connections")
}
