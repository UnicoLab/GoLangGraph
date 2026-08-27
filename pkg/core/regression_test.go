package core

// Regression tests for defects found in the original execution engine. Each
// test corresponds to a bug that was reproduced against the previous
// implementation before being fixed:
//
//   - AddConditionalEdges was recorded but never consulted by Execute.
//   - Clone panicked on any struct with unexported fields (time.Time).
//   - A node returning a nil state panicked while holding the graph read lock,
//     deadlocking every subsequent operation on the graph.
//   - Interrupt after Close panicked by sending on a closed channel.
//   - Context cancellation lost the underlying cause.
//   - FromJSON left nil maps that panicked on the next write.
//   - Concurrent Execute calls shared mutable run state and cross-talked.

import (
	"context"
	"errors"
	"fmt"
	"testing"
	"time"
)

// BUG 1: AddConditionalEdges is completely ignored by Execute().
func TestRegression_ConditionalEdgesIgnoredByExecute(t *testing.T) {
	g := NewGraph("cond")
	g.Config.RetryAttempts = 0
	g.AddNode("start", "start", func(ctx context.Context, s *BaseState) (*BaseState, error) {
		s.Set("route", "b")
		return s, nil
	})
	g.AddNode("a", "a", func(ctx context.Context, s *BaseState) (*BaseState, error) {
		s.Set("visited", "a")
		return s, nil
	})
	g.AddNode("b", "b", func(ctx context.Context, s *BaseState) (*BaseState, error) {
		s.Set("visited", "b")
		return s, nil
	})
	_ = g.SetStartNode("start")
	_ = g.AddEndNode("a")
	_ = g.AddEndNode("b")
	if err := g.AddConditionalEdges("start", func(ctx context.Context, s *BaseState) (string, error) {
		v, _ := s.Get("route")
		return v.(string), nil
	}, map[string]string{"a": "a", "b": "b"}); err != nil {
		t.Fatalf("AddConditionalEdges: %v", err)
	}
	out, err := g.Execute(context.Background(), NewBaseState())
	t.Logf("err=%v", err)
	if err != nil {
		t.Fatalf("conditional routing not honored by Execute: %v", err)
	}
	v, _ := out.Get("visited")
	if v != "b" {
		t.Fatalf("expected to visit b, got %v", v)
	}
}

// BUG 2: deepCopy panics on structs with unexported fields (e.g. time.Time).
func TestRegression_CloneWithTimeTime(t *testing.T) {
	s := NewBaseState()
	s.Set("ts", time.Now())
	defer func() {
		if r := recover(); r != nil {
			t.Fatalf("Clone panicked on time.Time: %v", r)
		}
	}()
	c := s.Clone()
	if _, ok := c.Get("ts"); !ok {
		t.Fatal("ts missing after clone")
	}
}

// BUG 3: node returning nil state nil-derefs on the next iteration.
func TestRegression_NilStateFromNode(t *testing.T) {
	g := NewGraph("nilstate")
	g.Config.RetryAttempts = 0
	g.AddNode("a", "a", func(ctx context.Context, s *BaseState) (*BaseState, error) {
		return nil, nil
	})
	g.AddNode("b", "b", func(ctx context.Context, s *BaseState) (*BaseState, error) {
		return s, nil
	})
	g.AddEdge("a", "b", nil)
	_ = g.SetStartNode("a")
	_ = g.AddEndNode("b")
	defer func() {
		if r := recover(); r != nil {
			t.Fatalf("nil state panics engine: %v", r)
		}
	}()
	_, err := g.Execute(context.Background(), NewBaseState())
	t.Logf("err=%v", err)
}

// BUG 4: Interrupt() after Close() panics (send on closed channel).
func TestRegression_InterruptAfterClose(t *testing.T) {
	g := NewGraph("closed")
	defer func() {
		if r := recover(); r != nil {
			t.Fatalf("Interrupt after Close panics: %v", r)
		}
	}()
	g.Close()
	g.Interrupt()
}

// BUG 5: context.Canceled is not preserved through Execute.
func TestRegression_ContextErrorNotWrapped(t *testing.T) {
	g := NewGraph("cancel")
	g.Config.RetryAttempts = 0
	g.AddNode("a", "a", func(ctx context.Context, s *BaseState) (*BaseState, error) {
		return s, nil
	})
	g.AddEdge("a", "a", nil)
	_ = g.SetStartNode("a")
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	_, err := g.Execute(ctx, NewBaseState())
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("errors.Is(err, context.Canceled) is false; err=%v", err)
	}
}

// BUG 6: FromJSON with empty object yields nil map -> panic on Set.
func TestRegression_FromJSONNilMap(t *testing.T) {
	s := NewBaseState()
	if err := s.FromJSON([]byte(`{}`)); err != nil {
		t.Fatalf("FromJSON: %v", err)
	}
	defer func() {
		if r := recover(); r != nil {
			t.Fatalf("Set after FromJSON({}) panics: %v", r)
		}
	}()
	s.Set("k", "v")
}

// BUG 7: concurrent Execute on one graph corrupts shared execution state.
func TestRegression_ConcurrentExecute(t *testing.T) {
	g := NewGraph("conc")
	g.Config.RetryAttempts = 0
	g.Config.EnableStreaming = false
	g.AddNode("a", "a", func(ctx context.Context, s *BaseState) (*BaseState, error) {
		v, _ := s.Get("in")
		s.Set("out", v)
		time.Sleep(2 * time.Millisecond)
		return s, nil
	})
	_ = g.SetStartNode("a")
	_ = g.AddEndNode("a")

	errCh := make(chan error, 8)
	for i := 0; i < 8; i++ {
		go func(i int) {
			st := NewBaseState()
			st.Set("in", i)
			out, err := g.Execute(context.Background(), st)
			if err != nil {
				errCh <- err
				return
			}
			got, _ := out.Get("out")
			if got != i {
				errCh <- fmt.Errorf("cross-talk: sent %d got %v", i, got)
				return
			}
			errCh <- nil
		}(i)
	}
	for i := 0; i < 8; i++ {
		if err := <-errCh; err != nil {
			t.Fatalf("concurrent Execute is unsafe: %v", err)
		}
	}
}
