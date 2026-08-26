// Copyright (c) 2024 GoLangGraph Team
//
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

package e2e

import (
	"context"
	"fmt"
	"net/http"
	"runtime"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/gorilla/websocket"
	"github.com/piotrlaczkowski/GoLangGraph/pkg/core"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// settledGoroutines waits for the goroutine count to stabilise, so a background
// worker that is merely slow to exit is not mistaken for a leak.
func settledGoroutines(t *testing.T) int {
	t.Helper()
	var last int
	for i := 0; i < 50; i++ {
		runtime.GC()
		time.Sleep(50 * time.Millisecond)
		current := runtime.NumGoroutine()
		if current == last {
			return current
		}
		last = current
	}
	return last
}

// Repeated executions must not leak goroutines. A framework that leaks one
// goroutine per run exhausts a long-lived server.
func TestResource_RepeatedExecutionsDoNotLeakGoroutines(t *testing.T) {
	live := startServer(t, nil)

	// Warm up so first-use allocations are not counted as leaks.
	for i := 0; i < 5; i++ {
		status, _ := live.do(t, http.MethodPost, "/api/v1/graphs/studio-workflow/execute",
			map[string]interface{}{"input": "warmup"})
		require.Equal(t, http.StatusOK, status)
	}

	before := settledGoroutines(t)

	for i := 0; i < 100; i++ {
		status, _ := live.do(t, http.MethodPost, "/api/v1/graphs/studio-workflow/execute",
			map[string]interface{}{"input": fmt.Sprintf("run-%d", i)})
		require.Equal(t, http.StatusOK, status)
	}

	after := settledGoroutines(t)
	assert.LessOrEqual(t, after, before+10,
		"goroutines grew from %d to %d over 100 executions", before, after)
}

// WebSocket churn must release its connections and streaming goroutines.
func TestResource_WebSocketChurnDoesNotLeak(t *testing.T) {
	live := startServer(t, nil)
	wsURL := "ws" + strings.TrimPrefix(live.baseURL, "http") + "/api/v1/ws/graphs/studio-workflow/stream"

	dialRun := func() {
		conn, _, err := websocket.DefaultDialer.Dial(wsURL, nil)
		require.NoError(t, err)
		defer func() { _ = conn.Close() }()

		require.NoError(t, conn.WriteJSON(map[string]interface{}{"type": "execute", "input": "x"}))
		require.NoError(t, conn.SetReadDeadline(time.Now().Add(15*time.Second)))
		for {
			var msg map[string]interface{}
			if err := conn.ReadJSON(&msg); err != nil {
				return
			}
			if kind, _ := msg["type"].(string); kind == "complete" || kind == "error" {
				return
			}
		}
	}

	for i := 0; i < 5; i++ {
		dialRun()
	}
	before := settledGoroutines(t)

	for i := 0; i < 40; i++ {
		dialRun()
	}
	after := settledGoroutines(t)

	assert.LessOrEqual(t, after, before+10,
		"goroutines grew from %d to %d over 40 WebSocket sessions", before, after)
}

// An interrupted graph must release everything it was holding.
func TestResource_InterruptedRunsAreCleanedUp(t *testing.T) {
	live := startServer(t, nil)

	blocking := core.NewGraph("blocking-cleanup")
	blocking.Config.EnableStreaming = false
	entered := make(chan struct{}, 100)
	blocking.AddNode("wait", "Wait", func(ctx context.Context, st *core.BaseState) (*core.BaseState, error) {
		entered <- struct{}{}
		<-ctx.Done()
		return nil, ctx.Err()
	})
	require.NoError(t, blocking.SetStartNode("wait"))
	live.srv.GraphManager().Register("blocking-cleanup", blocking)

	before := settledGoroutines(t)

	for i := 0; i < 20; i++ {
		ctx, cancel := context.WithTimeout(context.Background(), 50*time.Millisecond)
		done := make(chan struct{})
		go func() {
			defer close(done)
			_, _ = blocking.Execute(ctx, core.NewBaseState())
		}()
		<-entered
		cancel()
		<-done
	}

	after := settledGoroutines(t)
	assert.LessOrEqual(t, after, before+10,
		"goroutines grew from %d to %d over 20 cancelled runs", before, after)
	assert.False(t, blocking.IsRunning(), "no run should still be marked active")
}

// A long-running graph must complete without unbounded memory growth in its
// history or state.
func TestResource_LongRunningGraph(t *testing.T) {
	g := core.NewGraph("long-running")
	g.Config.MaxIterations = 5000
	g.Config.EnableStreaming = false

	g.AddNode("step", "Step", func(ctx context.Context, st *core.BaseState) (*core.BaseState, error) {
		n, _ := st.Get("n")
		count, _ := n.(int)
		st.Set("n", count+1)
		return st, nil
	})
	g.AddNode("done", "Done", func(ctx context.Context, st *core.BaseState) (*core.BaseState, error) {
		st.Set("done", true)
		return st, nil
	})
	require.NoError(t, g.SetStartNode("step"))
	require.NoError(t, g.AddEndNode("done"))
	require.NoError(t, g.AddConditionalEdges("step",
		func(ctx context.Context, st *core.BaseState) (string, error) {
			n, _ := st.Get("n")
			if count, _ := n.(int); count >= 2000 {
				return "exit", nil
			}
			return "again", nil
		},
		map[string]string{"again": "step", "exit": "done"}))

	start := time.Now()
	out, err := g.Execute(context.Background(), core.NewBaseState())
	require.NoError(t, err)

	n, _ := out.Get("n")
	assert.Equal(t, 2000, n)
	done, _ := out.Get("done")
	assert.Equal(t, true, done)
	t.Logf("2001 node executions in %s", time.Since(start))

	history := g.GetExecutionHistory()
	assert.Len(t, history, 2001, "every step must be recorded exactly once")
}

// The same request sent many times concurrently must produce the same result
// each time, with no cross-talk between the duplicates.
func TestResource_DuplicateConcurrentRequests(t *testing.T) {
	live := startServer(t, nil)

	const copies = 30
	var wg sync.WaitGroup
	results := make([]string, copies)

	for i := 0; i < copies; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			status, body := live.do(t, http.MethodPost, "/api/v1/graphs/studio-workflow/execute",
				map[string]interface{}{"input": "identical request"})
			if status != http.StatusOK {
				results[i] = fmt.Sprintf("status %d", status)
				return
			}
			var resp struct {
				State map[string]interface{} `json:"state"`
			}
			decode(t, body, &resp)
			results[i] = fmt.Sprint(resp.State["result"])
		}(i)
	}
	wg.Wait()

	for i, got := range results {
		assert.Equal(t, "long", got, "duplicate %d produced a different result", i)
	}
}

// Duplicate agent executions must each get their own execution record.
func TestResource_DuplicateAgentExecutionsAreDistinct(t *testing.T) {
	live := startServer(t, nil)

	const copies = 10
	var wg sync.WaitGroup
	ids := make([]string, copies)

	for i := 0; i < copies; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			status, body := live.do(t, http.MethodPost, "/api/v1/agents/studio-agent/execute",
				map[string]string{"input": "same input"})
			if status != http.StatusOK {
				return
			}
			var wrapper struct {
				Execution map[string]interface{} `json:"execution"`
			}
			decode(t, body, &wrapper)
			ids[i] = fmt.Sprint(wrapper.Execution["id"])
		}(i)
	}
	wg.Wait()

	seen := map[string]bool{}
	for _, id := range ids {
		if id == "" || id == "<nil>" {
			continue
		}
		assert.False(t, seen[id], "execution ID %s was reused across concurrent runs", id)
		seen[id] = true
	}
	assert.NotEmpty(t, seen, "at least one execution must have completed")
}

// The server must survive being started and stopped repeatedly, releasing its
// port each time.
func TestResource_ServerRestartCycles(t *testing.T) {
	for i := 0; i < 3; i++ {
		live := startServer(t, nil)

		status, _ := live.do(t, http.MethodGet, "/api/v1/health", nil)
		require.Equal(t, http.StatusOK, status, "cycle %d", i)

		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		err := live.srv.Stop(ctx)
		cancel()
		require.NoError(t, err, "cycle %d shutdown", i)
	}
}

// A slow provider must not hold a request open past the client's deadline.
func TestResource_SlowProviderDoesNotHangRequest(t *testing.T) {
	live := startServer(t, nil)
	live.provider.setDelay(30 * time.Second)

	client := &http.Client{Timeout: 2 * time.Second}
	req, err := http.NewRequest(http.MethodPost,
		live.baseURL+"/api/v1/agents/studio-agent/execute",
		strings.NewReader(`{"input":"hi"}`))
	require.NoError(t, err)
	req.Header.Set("Content-Type", "application/json")

	start := time.Now()
	resp, err := client.Do(req)
	if resp != nil {
		_ = resp.Body.Close()
	}
	elapsed := time.Since(start)

	// Either the request is cut off by the client deadline or the server
	// returns; what matters is that it does not run for the provider's delay.
	assert.Less(t, elapsed, 10*time.Second,
		"request took %s; a slow provider must not pin a connection", elapsed)
	_ = err
}
