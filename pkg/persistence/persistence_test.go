// Copyright (c) 2024 GoLangGraph Team
//
// Licensed under the MIT License. See LICENSE file in the project root for full license information.
//
// Package: GoLangGraph - A powerful Go framework for building AI agent workflows

package persistence

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/UnicoLab/GoLangGraph/pkg/core"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func stateWith(kv map[string]interface{}) *core.BaseState {
	s := core.NewBaseState()
	for k, v := range kv {
		s.Set(k, v)
	}
	return s
}

// A checkpoint must survive JSON encoding. BaseState keeps its data in
// unexported fields, so without a marshaller every persisted checkpoint and
// every API response carrying a state is silently empty.
func TestCheckpoint_JSONRoundTripPreservesState(t *testing.T) {
	cp := &Checkpoint{
		ID: "c1", ThreadID: "t1", NodeID: "n1", StepID: 3,
		CreatedAt: time.Now().UTC().Truncate(time.Second),
		State:     stateWith(map[string]interface{}{"counter": 42, "messages": []interface{}{"hi"}}),
		Metadata:  map[string]interface{}{"source": "test"},
	}

	data, err := json.Marshal(cp)
	require.NoError(t, err)
	assert.NotContains(t, string(data), `"state":{}`, "state must not serialize as an empty object")

	var back Checkpoint
	require.NoError(t, json.Unmarshal(data, &back))
	require.NotNil(t, back.State)

	counter, ok := back.State.Get("counter")
	require.True(t, ok, "state was lost across the round trip")
	assert.EqualValues(t, 42, counter)
	assert.Equal(t, 3, back.StepID)
	assert.Equal(t, "n1", back.NodeID)
}

// FileCheckpointer previously had stub IO: Save reported success and wrote
// nothing at all.
func TestFileCheckpointer_ActuallyWritesToDisk(t *testing.T) {
	base := filepath.Join(t.TempDir(), "store")
	cp := NewFileCheckpointer(base)
	ctx := context.Background()

	require.NoError(t, cp.Save(ctx, &Checkpoint{
		ID: "cp-1", ThreadID: "t1", NodeID: "start", StepID: 0,
		State: stateWith(map[string]interface{}{"v": "persisted"}), CreatedAt: time.Now(),
	}))

	path := filepath.Join(base, "t1", "cp-1.json")
	info, err := os.Stat(path)
	require.NoError(t, err, "no checkpoint file was written")
	assert.Greater(t, info.Size(), int64(0))

	raw, err := os.ReadFile(path)
	require.NoError(t, err)
	assert.Contains(t, string(raw), "persisted")

	loaded, err := cp.Load(ctx, "t1", "cp-1")
	require.NoError(t, err)
	v, ok := loaded.State.Get("v")
	require.True(t, ok)
	assert.Equal(t, "persisted", v)
}

// A checkpoint holds whatever the graph put in state — conversation history,
// tool results, credentials a node read — so the file must not be readable by
// anyone but the user the process runs as. It was written 0640.
func TestFileCheckpointer_WritesPrivateFiles(t *testing.T) {
	base := filepath.Join(t.TempDir(), "store")
	cp := NewFileCheckpointer(base)

	require.NoError(t, cp.Save(context.Background(), &Checkpoint{
		ID: "cp-1", ThreadID: "t1", NodeID: "start", StepID: 0,
		State: stateWith(map[string]interface{}{"api_key": "secret"}), CreatedAt: time.Now(),
	}))

	info, err := os.Stat(filepath.Join(base, "t1", "cp-1.json"))
	require.NoError(t, err)
	assert.Equal(t, os.FileMode(0o600), info.Mode().Perm(),
		"checkpoint state must not be group- or world-readable")
}

// A checkpoint survives a process restart: a fresh checkpointer over the same
// directory sees everything the previous one wrote.
func TestFileCheckpointer_SurvivesRestart(t *testing.T) {
	base := filepath.Join(t.TempDir(), "store")
	ctx := context.Background()

	first := NewFileCheckpointer(base)
	for i := 0; i < 3; i++ {
		require.NoError(t, first.Save(ctx, &Checkpoint{
			ID: fmt.Sprintf("cp-%d", i), ThreadID: "t", StepID: i,
			State: stateWith(map[string]interface{}{"step": i}), CreatedAt: time.Now(),
		}))
	}
	require.NoError(t, first.Close())

	// A new process attaches to the same directory.
	second := NewFileCheckpointer(base)
	metas, err := second.List(ctx, "t")
	require.NoError(t, err)
	assert.Len(t, metas, 3, "checkpoints must outlive the process that wrote them")

	latest, err := Latest(ctx, second, "t")
	require.NoError(t, err)
	require.NotNil(t, latest)
	step, _ := latest.State.Get("step")
	assert.EqualValues(t, 2, step)
}

// Writes are atomic: a reader must never observe a half-written checkpoint.
func TestFileCheckpointer_NoPartialFilesRemain(t *testing.T) {
	base := filepath.Join(t.TempDir(), "store")
	cp := NewFileCheckpointer(base)
	ctx := context.Background()

	big := strings.Repeat("payload", 10000)
	require.NoError(t, cp.Save(ctx, &Checkpoint{
		ID: "cp", ThreadID: "t", State: stateWith(map[string]interface{}{"blob": big}), CreatedAt: time.Now(),
	}))

	entries, err := os.ReadDir(filepath.Join(base, "t"))
	require.NoError(t, err)
	for _, e := range entries {
		assert.False(t, strings.HasSuffix(e.Name(), ".tmp"),
			"temporary file %s was left behind", e.Name())
	}
	assert.Len(t, entries, 1)
}

func TestFileCheckpointer_RejectsUnsafeIdentifiers(t *testing.T) {
	cp := NewFileCheckpointer(filepath.Join(t.TempDir(), "store"))
	ctx := context.Background()

	for _, bad := range []string{"../escape", "a/b", "..", "", "x\x00y", strings.Repeat("a", 300)} {
		err := cp.Save(ctx, &Checkpoint{ID: "ok", ThreadID: bad, State: core.NewBaseState(), CreatedAt: time.Now()})
		assert.Error(t, err, "thread ID %q must be rejected", bad)

		_, err = cp.Load(ctx, bad, "ok")
		assert.Error(t, err, "loading thread ID %q must be rejected", bad)
	}
}

func TestMemoryCheckpointer_IsolatesStoredState(t *testing.T) {
	cp := NewMemoryCheckpointer()
	ctx := context.Background()

	original := stateWith(map[string]interface{}{"v": 1})
	require.NoError(t, cp.Save(ctx, &Checkpoint{ID: "cp", ThreadID: "t", State: original, CreatedAt: time.Now()}))

	// Mutating the caller's state must not change what was stored.
	original.Set("v", 999)

	loaded, err := cp.Load(ctx, "t", "cp")
	require.NoError(t, err)
	v, _ := loaded.State.Get("v")
	assert.EqualValues(t, 1, v, "stored checkpoints must not alias caller state")

	// Mutating a loaded copy must not change the store either.
	loaded.State.Set("v", 555)
	again, err := cp.Load(ctx, "t", "cp")
	require.NoError(t, err)
	v, _ = again.State.Get("v")
	assert.EqualValues(t, 1, v)
}

// Every backend must behave the same under concurrent use.
func TestCheckpointers_ConcurrentAccess(t *testing.T) {
	backends := map[string]Checkpointer{
		"memory": NewMemoryCheckpointer(),
		"file":   NewFileCheckpointer(filepath.Join(t.TempDir(), "store")),
	}

	for name, cp := range backends {
		t.Run(name, func(t *testing.T) {
			ctx := context.Background()
			var wg sync.WaitGroup
			errs := make([]error, 24)

			for i := 0; i < 24; i++ {
				wg.Add(1)
				go func(i int) {
					defer wg.Done()
					thread := fmt.Sprintf("t%d", i%4)
					id := fmt.Sprintf("cp-%d", i)
					if err := cp.Save(ctx, &Checkpoint{
						ID: id, ThreadID: thread, StepID: i,
						State: stateWith(map[string]interface{}{"i": i}), CreatedAt: time.Now(),
					}); err != nil {
						errs[i] = err
						return
					}
					if _, err := cp.Load(ctx, thread, id); err != nil {
						errs[i] = err
						return
					}
					if _, err := cp.List(ctx, thread); err != nil {
						errs[i] = err
					}
				}(i)
			}
			wg.Wait()
			for i, err := range errs {
				assert.NoError(t, err, "worker %d", i)
			}
		})
	}
}

// A corrupt file must be reported on load and skipped on list, never panic.
func TestFileCheckpointer_CorruptedDataIsHandled(t *testing.T) {
	base := filepath.Join(t.TempDir(), "store")
	cp := NewFileCheckpointer(base)
	ctx := context.Background()

	require.NoError(t, cp.Save(ctx, &Checkpoint{
		ID: "good", ThreadID: "t", State: stateWith(map[string]interface{}{"v": 1}), CreatedAt: time.Now(),
	}))
	require.NoError(t, cp.Save(ctx, &Checkpoint{
		ID: "bad", ThreadID: "t", State: stateWith(map[string]interface{}{"v": 2}), CreatedAt: time.Now(),
	}))

	require.NoError(t, os.WriteFile(filepath.Join(base, "t", "bad.json"), []byte("\x00\x01 not json"), 0o600))

	_, err := cp.Load(ctx, "t", "bad")
	assert.Error(t, err, "a corrupt checkpoint must be reported")

	metas, err := cp.List(ctx, "t")
	require.NoError(t, err, "one corrupt file must not fail the whole listing")
	require.Len(t, metas, 1)
	assert.Equal(t, "good", metas[0].ID)
}

func TestCheckpointSaver_PersistsPerStep(t *testing.T) {
	cp := NewMemoryCheckpointer()
	saver := NewCheckpointSaver(cp)
	ctx := context.Background()

	for step, node := range []string{"a", "b", "c"} {
		require.NoError(t, saver.SaveState(ctx, "thread", node, step,
			stateWith(map[string]interface{}{"step": step})))
	}

	metas, err := cp.List(ctx, "thread")
	require.NoError(t, err)
	assert.Len(t, metas, 3)

	latest, err := Latest(ctx, cp, "thread")
	require.NoError(t, err)
	assert.Equal(t, "c", latest.NodeID)
	assert.Equal(t, 2, latest.StepID)
}

func TestCheckpointSaver_RejectsNilState(t *testing.T) {
	saver := NewCheckpointSaver(NewMemoryCheckpointer())
	err := saver.SaveState(context.Background(), "t", "n", 0, nil)
	assert.Error(t, err)
}

func TestCheckpointSaver_NilCheckpointerIsNoOp(t *testing.T) {
	saver := NewCheckpointSaver(nil)
	assert.NoError(t, saver.SaveState(context.Background(), "t", "n", 0, core.NewBaseState()))
}

func TestLatest_EmptyThread(t *testing.T) {
	latest, err := Latest(context.Background(), NewMemoryCheckpointer(), "nothing-here")
	require.NoError(t, err)
	assert.Nil(t, latest, "an empty thread has no latest checkpoint, and that is not an error")
}

// ---------------------------------------------------------------------------
// Database layer robustness
// ---------------------------------------------------------------------------

// fakeConnection implements DatabaseConnection without a real database, which
// is exactly the case that used to panic on an unchecked type assertion.
type fakeConnection struct {
	execErr error
	pinged  bool
}

func (f *fakeConnection) Connect() error        { return nil }
func (f *fakeConnection) Ping() error           { f.pinged = true; return nil }
func (f *fakeConnection) Close() error          { return nil }
func (f *fakeConnection) GetType() DatabaseType { return DatabaseType("fake") }
func (f *fakeConnection) GetConfig() *DatabaseConfig {
	return &DatabaseConfig{Type: DatabaseType("fake")}
}
func (f *fakeConnection) ExecuteQuery(ctx context.Context, query string, args ...interface{}) error {
	return f.execErr
}
func (f *fakeConnection) QueryRow(ctx context.Context, query string, args ...interface{}) interface{} {
	return nil
}
func (f *fakeConnection) QueryRows(ctx context.Context, query string, args ...interface{}) (interface{}, error) {
	return nil, nil
}

func TestSessionManager_NonSQLConnectionReturnsErrorNotPanic(t *testing.T) {
	sm := NewSessionManager(&fakeConnection{})
	ctx := context.Background()

	defer func() {
		if r := recover(); r != nil {
			t.Fatalf("a non-SQL DatabaseConnection must not panic: %v", r)
		}
	}()

	_, err := sm.GetSession(ctx, "s1")
	assert.Error(t, err)

	_, err = sm.GetThread(ctx, "t1")
	assert.Error(t, err)
}

func TestSessionManager_NilConnectionReturnsError(t *testing.T) {
	sm := NewSessionManager(nil)
	ctx := context.Background()

	defer func() {
		if r := recover(); r != nil {
			t.Fatalf("a nil connection must not panic: %v", r)
		}
	}()

	_, err := sm.GetSession(ctx, "s1")
	assert.Error(t, err)
}

func TestPostgresConnection_UnopenedConnectionErrors(t *testing.T) {
	conn := &PostgresConnection{}
	ctx := context.Background()

	defer func() {
		if r := recover(); r != nil {
			t.Fatalf("an unopened connection must not panic: %v", r)
		}
	}()

	assert.Error(t, conn.Ping())
	assert.Error(t, conn.ExecuteQuery(ctx, "SELECT 1"))
	_, err := conn.QueryRows(ctx, "SELECT 1")
	assert.Error(t, err)
	assert.Nil(t, conn.QueryRow(ctx, "SELECT 1"))
}
