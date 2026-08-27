// Copyright (c) 2024 GoLangGraph Team
//
// Licensed under the MIT License. See LICENSE file in the project root for full license information.
//
// Package: GoLangGraph - A powerful Go framework for building AI agent workflows

// Copyright (c) 2024 GoLangGraph Team
//
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

package conformance

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/UnicoLab/GoLangGraph/pkg/core"
	"github.com/UnicoLab/GoLangGraph/pkg/llm"
	"github.com/UnicoLab/GoLangGraph/pkg/persistence"
	"github.com/UnicoLab/GoLangGraph/pkg/server"
	"github.com/UnicoLab/GoLangGraph/pkg/tools"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// The examples in docs/PRODUCTION.md are executed here so the documentation
// cannot drift from the API it describes.

func TestDocumented_SecurityConfig(t *testing.T) {
	cfg := server.DefaultServerConfig()
	cfg.Security = &server.SecurityConfig{
		RequireAuth:     true,
		APIKeys:         []string{"a-key"},
		AllowedOrigins:  []string{"https://studio.example.com"},
		MaxRequestBytes: 4 << 20,
		PublicPaths:     []string{"/api/v1/health"},
	}
	srv := server.NewServer(cfg)
	require.NotNil(t, srv)
	assert.NotNil(t, srv.GraphManager())
}

func TestDocumented_ToolSecurityPolicy(t *testing.T) {
	dir := t.TempDir()

	policy := tools.DefaultSecurityPolicy()
	policy.SetAllowedRoots([]string{dir})
	policy.MaxOutputBytes = 256 << 10
	policy.AllowedCommands = []string{"ls", "wc"}
	policy.AllowedHosts = []string{"api.example.com"}

	read := tools.NewFileReadTool()
	read.SetSecurityPolicy(policy)

	target := filepath.Join(dir, "note.txt")
	require.NoError(t, os.WriteFile(target, []byte("documented"), 0o600))

	out, err := read.Execute(context.Background(), `{"file_path":"`+target+`"}`)
	require.NoError(t, err)
	assert.Contains(t, out, "documented")

	// The documented claim: "find" is not in the default allowlist.
	assert.NotContains(t, tools.DefaultSecurityPolicy().AllowedCommands, "find")
	assert.NotContains(t, tools.DefaultSecurityPolicy().AllowedCommands, "cat")
	assert.NotContains(t, tools.DefaultSecurityPolicy().AllowedCommands, "grep")
}

func TestDocumented_DurableExecutionAndResume(t *testing.T) {
	ctx := context.Background()
	threadID := "documented-thread"

	checkpointer := persistence.NewFileCheckpointer(filepath.Join(t.TempDir(), "checkpoints"))
	saver := persistence.NewCheckpointSaver(checkpointer)

	build := func() *core.Graph {
		g := core.NewGraph("documented")
		g.WithCheckpointer(saver, threadID)
		g.AddNode("first", "First", setNode("first", true))
		g.AddNode("second", "Second", setNode("second", true))
		g.AddEdge("first", "second", nil)
		require.NoError(t, g.SetStartNode("first"))
		require.NoError(t, g.AddEndNode("second"))
		return g
	}

	// A run that stops after the first node.
	partial := build()
	partial.Config.InterruptAfter = []string{"first"}
	_, err := partial.Execute(ctx, core.NewBaseState())
	require.Error(t, err)

	// The documented resume recipe.
	latest, err := persistence.Latest(ctx, checkpointer, threadID)
	require.NoError(t, err)
	require.NotNil(t, latest)
	assert.Equal(t, "first", latest.NodeID)

	resumed := build()
	next, err := resumed.GetNextNodes(ctx, latest.NodeID, latest.State)
	require.NoError(t, err)
	require.NotEmpty(t, next)

	final, err := resumed.ExecuteWithOptions(ctx, latest.State, &core.ExecuteOptions{
		ThreadID:  threadID,
		StartNode: next[0],
	})
	require.NoError(t, err)

	first, ok := final.Get("first")
	require.True(t, ok, "work from before the restart must survive")
	assert.Equal(t, true, first)
	second, ok := final.Get("second")
	require.True(t, ok)
	assert.Equal(t, true, second)
}

func TestDocumented_HumanInTheLoop(t *testing.T) {
	g := core.NewGraph("documented-hitl")
	g.Config.InterruptBefore = []string{"apply_changes"}
	g.AddNode("propose", "Propose", setNode("amount", 100))
	g.AddNode("apply_changes", "Apply", func(ctx context.Context, s *core.BaseState) (*core.BaseState, error) {
		v, _ := s.Get("amount")
		s.Set("applied", v)
		return s, nil
	})
	g.AddEdge("propose", "apply_changes", nil)
	require.NoError(t, g.SetStartNode("propose"))
	require.NoError(t, g.AddEndNode("apply_changes"))

	_, err := g.Execute(context.Background(), core.NewBaseState())

	var interrupt *core.InterruptError
	require.True(t, errors.As(err, &interrupt))

	interrupt.State.Set("amount", 25)
	g.Config.InterruptBefore = nil

	final, err := g.Resume(context.Background(), interrupt)
	require.NoError(t, err)
	applied, _ := final.Get("applied")
	assert.Equal(t, 25, applied)
}

func TestDocumented_RetryPolicy(t *testing.T) {
	attempts := 0
	g := core.NewGraph("documented-retry")
	node := g.AddNode("fetch", "Fetch", func(ctx context.Context, s *core.BaseState) (*core.BaseState, error) {
		attempts++
		if attempts < 3 {
			return nil, llm.ErrProviderUnavailable
		}
		s.Set("fetched", true)
		return s, nil
	})
	node.Retry = &core.RetryPolicy{
		MaxAttempts: 3,
		Delay:       time.Millisecond,
		Backoff:     2,
		RetryIf:     func(err error) bool { return errors.Is(err, llm.ErrProviderUnavailable) },
	}
	require.NoError(t, g.SetStartNode("fetch"))
	require.NoError(t, g.AddEndNode("fetch"))

	out, err := g.Execute(context.Background(), core.NewBaseState())
	require.NoError(t, err)
	fetched, _ := out.Get("fetched")
	assert.Equal(t, true, fetched)
	assert.Equal(t, 3, attempts)
}

// Every sentinel the documentation lists must exist and be distinct.
func TestDocumented_ErrorSentinels(t *testing.T) {
	sentinels := []error{
		core.ErrGraphInvalid, core.ErrRecursionLimit, core.ErrInterrupted,
		core.ErrNodePanic, core.ErrNoRoute, core.ErrGraphClosed,
		llm.ErrProviderUnavailable, llm.ErrRateLimited,
		llm.ErrProviderAuth, llm.ErrProviderRequest,
	}

	for i, a := range sentinels {
		require.Error(t, a)
		for j, b := range sentinels {
			if i == j {
				continue
			}
			assert.False(t, errors.Is(a, b),
				"sentinels %v and %v must be distinguishable", a, b)
		}
	}
}
