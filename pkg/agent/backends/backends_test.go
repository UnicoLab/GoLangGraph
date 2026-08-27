// Copyright (c) 2024 GoLangGraph Team
//
// Licensed under the MIT License. See LICENSE file in the project root for full license information.
//
// Package: GoLangGraph - A powerful Go framework for building AI agent workflows

package backends

import (
	"context"
	"errors"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/UnicoLab/GoLangGraph/pkg/persistence"
)

func TestStoreBackend_PersistentFilesystemOperations(t *testing.T) {
	ctx := context.Background()
	store := persistence.NewInMemoryStore()
	backend := NewStoreBackend(store, "alice")

	for path, content := range map[string]string{
		"docs/report.txt":       "first needle\nsecond needle",
		"docs/nested/child.txt": "child needle",
		"notes/todo.md":         "remember this",
		"documents/ignored.txt": "outside docs",
	} {
		result, err := backend.Write(ctx, path, content)
		require.NoError(t, err)
		require.Empty(t, result.Error)
	}

	read, err := backend.Read(ctx, "docs/report.txt", 1, 1)
	require.NoError(t, err)
	assert.Contains(t, read, "     2\tsecond needle")
	assert.NotContains(t, read, "first needle")
	_, err = backend.Read(ctx, "docs/report.txt", -1, 0)
	require.Error(t, err)

	infos, err := backend.List(ctx, "docs")
	require.NoError(t, err)
	require.Len(t, infos, 1, "List is non-recursive, matching the state backend")
	assert.Equal(t, "docs/report.txt", infos[0].Path)
	assert.Positive(t, infos[0].Size)
	assert.False(t, infos[0].ModifiedAt.IsZero())

	all, err := backend.List(ctx, "/")
	require.NoError(t, err)
	assert.Equal(t, []string{"docs/nested/child.txt", "docs/report.txt", "documents/ignored.txt", "notes/todo.md"}, fileInfoPaths(all))

	globbed, err := backend.Glob(ctx, "*.txt", "docs")
	require.NoError(t, err)
	assert.Equal(t, []string{"docs/nested/child.txt", "docs/report.txt"}, globbed)
	_, err = backend.Glob(ctx, "[", "docs")
	require.Error(t, err)

	grep, err := backend.Grep(ctx, "needle", "docs", "*.txt")
	require.NoError(t, err)
	assert.Equal(t, []GrepMatch{
		{Path: "docs/nested/child.txt", LineNumber: 1, Line: "child needle"},
		{Path: "docs/report.txt", LineNumber: 1, Line: "first needle"},
		{Path: "docs/report.txt", LineNumber: 2, Line: "second needle"},
	}, grep)
	_, err = backend.Grep(ctx, "needle", "docs", "[")
	require.Error(t, err)

	edited, err := backend.Edit(ctx, "docs/report.txt", "needle", "match", true)
	require.NoError(t, err)
	require.Empty(t, edited.Error)
	assert.Equal(t, 2, edited.Occurrences)
	updated, err := backend.Read(ctx, "docs/report.txt", 0, 0)
	require.NoError(t, err)
	assert.Contains(t, updated, "match")
	assert.NotContains(t, updated, "needle")

	otherNamespace := NewStoreBackend(store, "bob")
	_, err = otherNamespace.Read(ctx, "docs/report.txt", 0, 0)
	require.Error(t, err, "namespaces must prevent one user reading another user's files")
}

func TestStoreBackend_ReportsStoreFailures(t *testing.T) {
	backend := NewStoreBackend(failingStore{err: errors.New("store unavailable")}, "test")
	_, err := backend.List(context.Background(), "/")
	require.Error(t, err)
	assert.Contains(t, err.Error(), "store unavailable")

	result, err := backend.Write(context.Background(), "a.txt", "content")
	require.NoError(t, err)
	assert.Contains(t, result.Error, "store unavailable")

	nilBackend := NewStoreBackend(nil, "test")
	_, err = nilBackend.Read(context.Background(), "a.txt", 0, 0)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "not configured")
}

func TestStateBackend_RejectsInvalidReadAndGrepFilters(t *testing.T) {
	state := map[string]interface{}{
		"files": map[string]*FileData{"docs/a.txt": {Content: []string{"needle"}}},
	}
	backend := NewStateBackend(func() map[string]interface{} { return state })
	_, err := backend.Read(context.Background(), "docs/a.txt", -1, 0)
	require.Error(t, err)
	_, err = backend.Grep(context.Background(), "needle", "docs", "[")
	require.Error(t, err)
}

func TestStateBackend_FilesystemOperationsAreDeterministic(t *testing.T) {
	ctx := context.Background()
	state := map[string]interface{}{
		"files": map[string]*FileData{
			"docs/report.txt":       {Content: []string{"alpha needle", "beta needle"}, CreatedAt: "2026-01-01T00:00:00Z", ModifiedAt: "2026-01-02T00:00:00Z"},
			"docs/nested/child.txt": {Content: []string{"child needle"}, CreatedAt: "2026-01-01T00:00:00Z", ModifiedAt: "2026-01-02T00:00:00Z"},
			"documents/other.txt":   {Content: []string{"not under docs"}, CreatedAt: "2026-01-01T00:00:00Z", ModifiedAt: "2026-01-02T00:00:00Z"},
			"empty.txt":             {Content: nil, CreatedAt: "2026-01-01T00:00:00Z", ModifiedAt: "2026-01-02T00:00:00Z"},
		},
	}
	backend := NewStateBackend(func() map[string]interface{} { return state })

	write, err := backend.Write(ctx, "new.txt", "one\ntwo")
	require.NoError(t, err)
	assert.Equal(t, []string{"one", "two"}, write.FilesUpdate["new.txt"].Content)

	read, err := backend.Read(ctx, "docs/report.txt", 1, 1)
	require.NoError(t, err)
	assert.Contains(t, read, "     2\tbeta needle")
	_, err = backend.Read(ctx, "docs/report.txt", 5, 0)
	require.Error(t, err)
	empty, err := backend.Read(ctx, "empty.txt", 0, 0)
	require.NoError(t, err)
	assert.Contains(t, empty, "empty contents")

	edited, err := backend.Edit(ctx, "docs/report.txt", "needle", "match", false)
	require.NoError(t, err)
	assert.Contains(t, edited.Error, "replace_all=true")
	edited, err = backend.Edit(ctx, "docs/report.txt", "needle", "match", true)
	require.NoError(t, err)
	assert.Empty(t, edited.Error)
	assert.Equal(t, 2, edited.Occurrences)
	missing, err := backend.Edit(ctx, "absent.txt", "a", "b", false)
	require.NoError(t, err)
	assert.Contains(t, missing.Error, "not found")

	infos, err := backend.List(ctx, "docs")
	require.NoError(t, err)
	assert.Equal(t, []string{"docs/report.txt"}, fileInfoPaths(infos))

	globbed, err := backend.Glob(ctx, "*.txt", "docs")
	require.NoError(t, err)
	assert.Equal(t, []string{"docs/nested/child.txt", "docs/report.txt"}, globbed)
	_, err = backend.Glob(ctx, "[", "docs")
	require.Error(t, err)

	grep, err := backend.Grep(ctx, "needle", "docs", "*.txt")
	require.NoError(t, err)
	assert.Equal(t, []GrepMatch{
		{Path: "docs/nested/child.txt", LineNumber: 1, Line: "child needle"},
		{Path: "docs/report.txt", LineNumber: 1, Line: "alpha needle"},
		{Path: "docs/report.txt", LineNumber: 2, Line: "beta needle"},
	}, grep)
}

func TestCompositeBackend_RoutesEveryFilesystemOperation(t *testing.T) {
	defaultBackend := &backendStub{name: "default"}
	memoryBackend := &backendStub{name: "memory"}
	privateBackend := &backendStub{name: "private"}
	backend := NewCompositeBackend(defaultBackend)
	backend.AddRoute("mem", memoryBackend)
	backend.AddRoute("mem/private", privateBackend)

	assert.Same(t, privateBackend, backend.getBackend("mem/private/secret.txt"))
	assert.Same(t, memoryBackend, backend.getBackend("mem/public.txt"))
	assert.Same(t, defaultBackend, backend.getBackend("workspace/readme.md"))

	read, err := backend.Read(context.Background(), "mem/private/secret.txt", 0, 0)
	require.NoError(t, err)
	assert.Equal(t, "private:mem/private/secret.txt", read)
	write, err := backend.Write(context.Background(), "mem/public.txt", "hello")
	require.NoError(t, err)
	assert.Equal(t, "memory", write.Path)
	edit, err := backend.Edit(context.Background(), "workspace/a.txt", "a", "b", false)
	require.NoError(t, err)
	assert.Equal(t, "default", edit.Path)
	listed, err := backend.List(context.Background(), "mem")
	require.NoError(t, err)
	assert.Equal(t, "memory", listed[0].Path)
	globbed, err := backend.Glob(context.Background(), "*.txt", "mem/private")
	require.NoError(t, err)
	assert.Equal(t, []string{"private"}, globbed)
	grepped, err := backend.Grep(context.Background(), "needle", "workspace", "")
	require.NoError(t, err)
	assert.Equal(t, "default", grepped[0].Path)

	assert.False(t, backend.SupportsExecution())
	execution, err := backend.Execute(context.Background(), "echo hello")
	require.NoError(t, err)
	assert.Equal(t, 1, execution.ExitCode)
	assert.Contains(t, execution.Output, "not supported")
}

func TestCompositeBackend_UsesExecutableDefault(t *testing.T) {
	backend := NewCompositeBackend(&executableBackend{backendStub: backendStub{name: "sandbox"}})
	assert.True(t, backend.SupportsExecution())
	result, err := backend.Execute(context.Background(), "echo hello")
	require.NoError(t, err)
	assert.Equal(t, 0, result.ExitCode)
	assert.Equal(t, "executed: echo hello", result.Output)
}

func fileInfoPaths(infos []FileInfo) []string {
	paths := make([]string, 0, len(infos))
	for _, info := range infos {
		paths = append(paths, info.Path)
	}
	return paths
}

type failingStore struct{ err error }

func (s failingStore) Get(context.Context, string) (string, error) { return "", s.err }
func (s failingStore) Set(context.Context, string, string) error   { return s.err }
func (s failingStore) Delete(context.Context, string) error        { return s.err }
func (s failingStore) Has(context.Context, string) (bool, error)   { return false, s.err }
func (s failingStore) List(context.Context, string) ([]string, error) {
	return nil, s.err
}

var _ persistence.Store = failingStore{}

type backendStub struct{ name string }

func (b *backendStub) Read(_ context.Context, path string, _ int, _ int) (string, error) {
	return b.name + ":" + path, nil
}
func (b *backendStub) Write(context.Context, string, string) (*WriteResult, error) {
	return &WriteResult{Path: b.name}, nil
}
func (b *backendStub) Edit(context.Context, string, string, string, bool) (*EditResult, error) {
	return &EditResult{Path: b.name}, nil
}
func (b *backendStub) List(context.Context, string) ([]FileInfo, error) {
	return []FileInfo{{Path: b.name}}, nil
}
func (b *backendStub) Glob(context.Context, string, string) ([]string, error) {
	return []string{b.name}, nil
}
func (b *backendStub) Grep(context.Context, string, string, string) ([]GrepMatch, error) {
	return []GrepMatch{{Path: b.name}}, nil
}

type executableBackend struct{ backendStub }

func (b *executableBackend) Execute(_ context.Context, command string) (*ExecuteResult, error) {
	return &ExecuteResult{Output: "executed: " + command, ExitCode: 0}, nil
}
