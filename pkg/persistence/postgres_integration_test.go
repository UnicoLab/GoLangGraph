// Copyright (c) 2024 GoLangGraph Team
//
// Licensed under the MIT License. See LICENSE file in the project root for full license information.
//
// Package: GoLangGraph - A powerful Go framework for building AI agent workflows

// Integration tests that run PostgresCheckpointer against a REAL PostgreSQL
// server. Nothing here is mocked: the queries, the argument binding, the row
// scanning and the schema are all exercised end to end.
//
// Discovery order:
//  1. POSTGRES_TEST_DSN, if set. An explicitly configured server that cannot be
//     reached is a hard failure -- the operator asked for these tests to run.
//  2. Otherwise a couple of conventional local DSNs. If none answer, every test
//     in this file skips with an explanation, so CI without a database stays
//     green.
//
// Each test runs inside its own PostgreSQL schema so tests cannot see one
// another's rows and a failure leaves nothing behind.

package persistence

import (
	"context"
	"database/sql"
	"fmt"
	"net/url"
	"os"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/lib/pq"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/piotrlaczkowski/GoLangGraph/pkg/core"
)

// candidatePostgresDSNs are tried in order when POSTGRES_TEST_DSN is unset.
// They cover the two usual local setups: password auth with the conventional
// "postgres" password, and trust auth where no password is needed.
var candidatePostgresDSNs = []string{
	"postgres://postgres:postgres@127.0.0.1:5432/postgres?sslmode=disable",
	"postgres://postgres@127.0.0.1:5432/postgres?sslmode=disable",
}

var (
	pgDiscoverOnce sync.Once
	pgBaseConfig   *DatabaseConfig
	pgDiscoverErr  error
	pgExplicitDSN  bool
	pgHasVector    bool
)

// parsePostgresDSN turns a postgres:// URL into the DatabaseConfig the
// checkpointer expects. The checkpointer builds its own libpq DSN from these
// fields, so we cannot simply pass the URL through.
func parsePostgresDSN(dsn string) (*DatabaseConfig, error) {
	u, err := url.Parse(dsn)
	if err != nil {
		return nil, fmt.Errorf("invalid DSN %q: %w", dsn, err)
	}

	port := 5432
	if p := u.Port(); p != "" {
		port, err = strconv.Atoi(p)
		if err != nil {
			return nil, fmt.Errorf("invalid port in DSN %q: %w", dsn, err)
		}
	}

	password, _ := u.User.Password()
	cfg := NewPostgresConfig(
		u.Hostname(),
		port,
		strings.TrimPrefix(u.Path, "/"),
		u.User.Username(),
		password,
	)
	if mode := u.Query().Get("sslmode"); mode != "" {
		cfg.SSLMode = mode
	}
	return cfg, nil
}

// discoverPostgres finds a usable server exactly once per test binary.
func discoverPostgres() (*DatabaseConfig, error) {
	pgDiscoverOnce.Do(func() {
		dsns := candidatePostgresDSNs
		if explicit := os.Getenv("POSTGRES_TEST_DSN"); explicit != "" {
			dsns = []string{explicit}
			pgExplicitDSN = true
		}

		var lastErr error
		for _, dsn := range dsns {
			cfg, err := parsePostgresDSN(dsn)
			if err != nil {
				lastErr = err
				continue
			}
			conn, err := NewPostgresConnection(cfg)
			if err != nil {
				lastErr = err
				continue
			}

			// Record whether pgvector is usable so the vector tests can skip
			// precisely rather than failing on a server without the extension.
			var one int
			row, rerr := asSQLRow(conn.QueryRow(context.Background(),
				`SELECT 1 FROM pg_extension WHERE extname = 'vector'`))
			if rerr == nil && row.Scan(&one) == nil {
				pgHasVector = true
			}

			_ = conn.Close()
			pgBaseConfig = cfg
			return
		}
		pgDiscoverErr = lastErr
	})
	return pgBaseConfig, pgDiscoverErr
}

// requirePostgres returns a base config, skipping the test when no server is
// reachable. An explicitly requested server that is down fails instead.
func requirePostgres(t *testing.T) *DatabaseConfig {
	t.Helper()

	cfg, err := discoverPostgres()
	if cfg == nil {
		if pgExplicitDSN {
			t.Fatalf("POSTGRES_TEST_DSN is set but the server is unreachable: %v", err)
		}
		t.Skipf("no local PostgreSQL reachable (tried %v; last error: %v). "+
			"Set POSTGRES_TEST_DSN to run these integration tests.", candidatePostgresDSNs, err)
	}

	clone := *cfg
	return &clone
}

// newPostgresSchema gives a test its own PostgreSQL schema, dropped on cleanup.
//
// Isolation matters here because the checkpointer creates fixed table names
// (threads, checkpoints, documents...) and one test's rows would otherwise show
// up in another's List. It also means each test starts against a genuinely
// empty schema, which is what exercises initSchema.
func newPostgresSchema(t *testing.T, mutate func(*DatabaseConfig)) *DatabaseConfig {
	t.Helper()

	base := requirePostgres(t)
	admin, err := NewPostgresConnection(base)
	require.NoError(t, err, "connect to PostgreSQL")

	schema := fmt.Sprintf("gltest_%d_%d", time.Now().UnixNano(), atomic.AddUint64(&schemaCounter, 1))
	ctx := context.Background()
	require.NoError(t, admin.ExecuteQuery(ctx, "CREATE SCHEMA "+pq.QuoteIdentifier(schema)))

	t.Cleanup(func() {
		if err := admin.ExecuteQuery(context.Background(),
			"DROP SCHEMA "+pq.QuoteIdentifier(schema)+" CASCADE"); err != nil {
			t.Logf("failed to drop schema %s: %v", schema, err)
		}
		if err := admin.Close(); err != nil {
			t.Logf("failed to close admin connection: %v", err)
		}
	})

	cfg := *base
	// "public" stays on the path so extension-provided types such as pgvector's
	// "vector" resolve; the test schema comes first so CREATE TABLE lands there.
	cfg.ConnectionParams = map[string]string{"search_path": schema + ",public"}
	if mutate != nil {
		mutate(&cfg)
	}
	return &cfg
}

var schemaCounter uint64

// newPostgresCheckpointer builds a checkpointer on a private schema.
func newPostgresCheckpointer(t *testing.T, mutate func(*DatabaseConfig)) *PostgresCheckpointer {
	t.Helper()

	cfg := newPostgresSchema(t, mutate)
	cp, err := NewPostgresCheckpointer(cfg)
	require.NoError(t, err, "create PostgresCheckpointer")
	t.Cleanup(func() {
		if err := cp.Close(); err != nil {
			t.Logf("failed to close checkpointer: %v", err)
		}
	})
	return cp
}

// richState builds a state covering the value shapes that actually flow through
// a graph: strings, numbers, booleans, nils, slices and nested maps. Anything
// that silently drops data on the way to storage shows up here.
func richState() *core.BaseState {
	st := core.NewBaseState()
	st.Set("greeting", "hello world")
	st.Set("count", 42)
	st.Set("ratio", 3.5)
	st.Set("enabled", true)
	st.Set("missing", nil)
	st.Set("tags", []interface{}{"a", "b", "c"})
	st.Set("nested", map[string]interface{}{
		"inner":  "value",
		"deep":   map[string]interface{}{"x": 1.0},
		"list":   []interface{}{1.0, 2.0},
		"quoted": `he said "hi"; DROP TABLE checkpoints;--`,
	})
	st.SetMetadata("origin", "integration-test")
	return st
}

// assertRichState checks a state survived a full store/load cycle.
//
// This is the assertion the whole exercise exists for: BaseState keeps its data
// in unexported fields, so before it grew MarshalJSON/UnmarshalJSON every
// checkpoint serialised as "{}" and lost everything. A round trip that returns
// the same values is the proof that no longer happens through PostgreSQL.
func assertRichState(t *testing.T, got *core.BaseState) {
	t.Helper()
	require.NotNil(t, got, "state must not be nil after load")

	all := got.GetAll()
	assert.Equal(t, "hello world", all["greeting"])
	// JSON has one number type, so integers come back as float64.
	assert.EqualValues(t, 42, all["count"])
	assert.EqualValues(t, 3.5, all["ratio"])
	assert.Equal(t, true, all["enabled"])
	assert.Nil(t, all["missing"])
	assert.Equal(t, []interface{}{"a", "b", "c"}, all["tags"])

	nested, ok := all["nested"].(map[string]interface{})
	require.True(t, ok, "nested value should decode as a map, got %T", all["nested"])
	assert.Equal(t, "value", nested["inner"])
	assert.Equal(t, `he said "hi"; DROP TABLE checkpoints;--`, nested["quoted"],
		"quotes and SQL metacharacters must survive verbatim")
	deep, ok := nested["deep"].(map[string]interface{})
	require.True(t, ok, "deeply nested map should survive")
	assert.EqualValues(t, 1, deep["x"])

	origin, found := got.GetMetadata("origin")
	assert.True(t, found, "state metadata must survive the round trip")
	assert.Equal(t, "integration-test", origin)
}

func newCheckpoint(threadID, id string, step int, state *core.BaseState) *Checkpoint {
	return &Checkpoint{
		ID:       id,
		ThreadID: threadID,
		State:    state,
		Metadata: map[string]interface{}{"node": "n" + strconv.Itoa(step), "step": step},
		// Postgres stores microsecond precision; truncating keeps equality
		// assertions honest rather than comparing against rounded values.
		CreatedAt: time.Now().UTC().Truncate(time.Microsecond),
		NodeID:    "node-" + strconv.Itoa(step),
		StepID:    step,
	}
}

// --- core round trip ------------------------------------------------------

func TestPostgresCheckpointer_SaveLoadRoundTripPreservesState(t *testing.T) {
	cp := newPostgresCheckpointer(t, nil)
	ctx := context.Background()

	want := newCheckpoint("thread-round-trip", "ckpt-1", 1, richState())
	require.NoError(t, cp.Save(ctx, want))

	got, err := cp.Load(ctx, want.ThreadID, want.ID)
	require.NoError(t, err)

	assert.Equal(t, want.ID, got.ID)
	assert.Equal(t, want.ThreadID, got.ThreadID)
	assert.Equal(t, want.NodeID, got.NodeID)
	assert.Equal(t, want.StepID, got.StepID)
	assert.WithinDuration(t, want.CreatedAt, got.CreatedAt, time.Millisecond)
	assert.Equal(t, "n1", got.Metadata["node"])
	assertRichState(t, got.State)
}

// TestPostgresCheckpointer_StoresRealJSONNotEmptyObject guards the specific
// regression that motivated these tests: state used to reach the database as
// literal "{}". Reading the column back as text proves the payload really
// contains the data, independent of how Load decodes it.
func TestPostgresCheckpointer_StoresRealJSONNotEmptyObject(t *testing.T) {
	cp := newPostgresCheckpointer(t, nil)
	ctx := context.Background()

	ck := newCheckpoint("thread-json", "ckpt-json", 1, richState())
	require.NoError(t, cp.Save(ctx, ck))

	row, err := asSQLRow(cp.conn.QueryRow(ctx,
		`SELECT state_data::text FROM checkpoints WHERE id = $1`, ck.ID))
	require.NoError(t, err)

	var raw string
	require.NoError(t, row.Scan(&raw))

	assert.NotEqual(t, "{}", raw, "state was serialised as an empty object -- all state lost")
	assert.Contains(t, raw, "hello world")
	assert.Contains(t, raw, "\"count\"")
	assert.Contains(t, raw, "integration-test", "state metadata must be persisted too")
}

// TestPostgresCheckpointer_SaveCreatesThreadRow covers the defect that made the
// PostgreSQL backend unusable: checkpoints.thread_id has a FOREIGN KEY to
// threads(id), but the Checkpointer interface offers no way to create a thread,
// so every first Save failed with a foreign key violation.
func TestPostgresCheckpointer_SaveCreatesThreadRow(t *testing.T) {
	cp := newPostgresCheckpointer(t, nil)
	ctx := context.Background()

	ck := newCheckpoint("brand-new-thread", "ckpt-fk", 1, core.NewBaseState())
	require.NoError(t, cp.Save(ctx, ck),
		"saving to a thread that was never registered must succeed, as it does for the memory and file backends")

	sm := NewSessionManager(cp.conn)
	thread, err := sm.GetThread(ctx, ck.ThreadID)
	require.NoError(t, err, "Save should have registered the parent thread")
	assert.Equal(t, ck.ThreadID, thread.ID)
}

func TestPostgresCheckpointer_SaveIsIdempotentUpsert(t *testing.T) {
	cp := newPostgresCheckpointer(t, nil)
	ctx := context.Background()

	first := core.NewBaseState()
	first.Set("version", "one")
	ck := newCheckpoint("thread-upsert", "ckpt-same-id", 1, first)
	require.NoError(t, cp.Save(ctx, ck))

	second := core.NewBaseState()
	second.Set("version", "two")
	ck.State = second
	ck.NodeID = "updated"
	require.NoError(t, cp.Save(ctx, ck), "re-saving the same ID must upsert, not conflict")

	got, err := cp.Load(ctx, ck.ThreadID, ck.ID)
	require.NoError(t, err)
	assert.Equal(t, "two", got.State.GetAll()["version"], "upsert must overwrite the stored state")
	assert.Equal(t, "updated", got.NodeID)

	list, err := cp.List(ctx, ck.ThreadID)
	require.NoError(t, err)
	assert.Len(t, list, 1, "upsert must not create a duplicate row")
}

// --- isolation ------------------------------------------------------------

func TestPostgresCheckpointer_ThreadIsolation(t *testing.T) {
	cp := newPostgresCheckpointer(t, nil)
	ctx := context.Background()

	a := core.NewBaseState()
	a.Set("owner", "alice")
	b := core.NewBaseState()
	b.Set("owner", "bob")

	require.NoError(t, cp.Save(ctx, newCheckpoint("thread-a", "shared-id", 1, a)))
	require.NoError(t, cp.Save(ctx, newCheckpoint("thread-b", "other-id", 1, b)))

	gotA, err := cp.Load(ctx, "thread-a", "shared-id")
	require.NoError(t, err)
	assert.Equal(t, "alice", gotA.State.GetAll()["owner"])

	// A checkpoint ID that exists, but under a different thread, must not be
	// readable from this thread.
	_, err = cp.Load(ctx, "thread-b", "shared-id")
	assert.Error(t, err, "loading another thread's checkpoint ID must fail")

	listA, err := cp.List(ctx, "thread-a")
	require.NoError(t, err)
	require.Len(t, listA, 1)
	assert.Equal(t, "shared-id", listA[0].ID)
}

// TestPostgresCheckpointer_IdentifiersWithSQLMetacharacters feeds hostile
// identifiers through every query. All statements use bound parameters, so
// these must be stored and matched verbatim rather than being interpreted.
func TestPostgresCheckpointer_IdentifiersWithSQLMetacharacters(t *testing.T) {
	cp := newPostgresCheckpointer(t, nil)
	ctx := context.Background()

	hostile := `thread'; DROP TABLE checkpoints; --`
	ckID := `id" OR "1"="1`

	st := core.NewBaseState()
	st.Set("safe", true)
	require.NoError(t, cp.Save(ctx, newCheckpoint(hostile, ckID, 1, st)))

	got, err := cp.Load(ctx, hostile, ckID)
	require.NoError(t, err, "identifiers must round trip verbatim")
	assert.Equal(t, hostile, got.ThreadID)
	assert.Equal(t, ckID, got.ID)

	// The table is obviously still there if this works.
	list, err := cp.List(ctx, hostile)
	require.NoError(t, err)
	assert.Len(t, list, 1)

	// And the injected predicate must not have widened the match.
	_, err = cp.Load(ctx, hostile, "no-such-checkpoint")
	assert.Error(t, err)
}

// --- listing and deletion -------------------------------------------------

func TestPostgresCheckpointer_ListReturnsEveryCheckpoint(t *testing.T) {
	cp := newPostgresCheckpointer(t, nil)
	ctx := context.Background()

	const n = 5
	for i := 0; i < n; i++ {
		st := core.NewBaseState()
		st.Set("i", i)
		require.NoError(t, cp.Save(ctx, newCheckpoint("thread-list", fmt.Sprintf("ckpt-%d", i), i, st)))
	}

	list, err := cp.List(ctx, "thread-list")
	require.NoError(t, err)
	require.Len(t, list, n)

	seen := map[string]bool{}
	for _, m := range list {
		seen[m.ID] = true
		assert.Equal(t, "thread-list", m.ThreadID)
		assert.NotNil(t, m.Metadata, "metadata must never be nil after List")
	}
	assert.Len(t, seen, n, "every checkpoint must appear exactly once")

	empty, err := cp.List(ctx, "thread-that-does-not-exist")
	require.NoError(t, err, "listing an unknown thread is not an error")
	assert.Empty(t, empty)
}

func TestPostgresCheckpointer_Delete(t *testing.T) {
	cp := newPostgresCheckpointer(t, nil)
	ctx := context.Background()

	require.NoError(t, cp.Save(ctx, newCheckpoint("thread-del", "ckpt-del", 1, core.NewBaseState())))
	require.NoError(t, cp.Delete(ctx, "thread-del", "ckpt-del"))

	_, err := cp.Load(ctx, "thread-del", "ckpt-del")
	assert.Error(t, err, "deleted checkpoint must be gone")

	list, err := cp.List(ctx, "thread-del")
	require.NoError(t, err)
	assert.Empty(t, list)
}

// TestPostgresCheckpointer_DeleteMissingReportsNotFound covers a defect where
// deleting a checkpoint that did not exist reported success, so a typo'd ID
// looked like a completed deletion. The memory and file backends both error.
func TestPostgresCheckpointer_DeleteMissingReportsNotFound(t *testing.T) {
	cp := newPostgresCheckpointer(t, nil)
	ctx := context.Background()

	require.NoError(t, cp.Save(ctx, newCheckpoint("thread-del2", "real", 1, core.NewBaseState())))

	err := cp.Delete(ctx, "thread-del2", "never-existed")
	assert.Error(t, err, "deleting a missing checkpoint must not report success")
	assert.Contains(t, err.Error(), "not found")

	// Deleting under the wrong thread must not delete the real row either.
	assert.Error(t, cp.Delete(ctx, "some-other-thread", "real"))
	_, err = cp.Load(ctx, "thread-del2", "real")
	assert.NoError(t, err, "a mis-targeted delete must not remove the real checkpoint")
}

func TestPostgresCheckpointer_LoadMissingIsAnError(t *testing.T) {
	cp := newPostgresCheckpointer(t, nil)
	ctx := context.Background()

	_, err := cp.Load(ctx, "no-thread", "no-checkpoint")
	require.Error(t, err)
	assert.Contains(t, err.Error(), "not found")
	assert.NotErrorIs(t, err, sql.ErrNoRows, "the raw driver sentinel should be translated")
}

// --- damaged and legacy rows ----------------------------------------------

// TestPostgresCheckpointer_TolerantOfNullColumns covers rows written by an older
// release or another tool. metadata, node_id and step_id are all nullable in the
// schema, and scanning a NULL used to abort Load *and* List -- and a failing
// List breaks Latest(), i.e. resuming the thread at all.
func TestPostgresCheckpointer_TolerantOfNullColumns(t *testing.T) {
	cp := newPostgresCheckpointer(t, nil)
	ctx := context.Background()

	ck := newCheckpoint("thread-null", "ckpt-null", 7, richState())
	require.NoError(t, cp.Save(ctx, ck))

	require.NoError(t, cp.conn.ExecuteQuery(ctx,
		`UPDATE checkpoints SET metadata = NULL, node_id = NULL, step_id = NULL WHERE id = $1`, ck.ID))

	got, err := cp.Load(ctx, ck.ThreadID, ck.ID)
	require.NoError(t, err, "a row with NULL metadata/node_id/step_id must still load")
	assert.NotNil(t, got.Metadata, "NULL metadata must decode to an empty map, not nil")
	assert.Empty(t, got.Metadata)
	assert.Equal(t, "", got.NodeID)
	assert.Equal(t, 0, got.StepID)
	assertRichState(t, got.State)

	list, err := cp.List(ctx, ck.ThreadID)
	require.NoError(t, err, "List must survive NULL columns as well")
	require.Len(t, list, 1)
	assert.NotNil(t, list[0].Metadata)
}

// TestPostgresCheckpointer_CorruptStateIsReported checks that unreadable data is
// surfaced rather than silently returning an empty state, which would look like
// a successful resume from nothing.
func TestPostgresCheckpointer_CorruptStateIsReported(t *testing.T) {
	cp := newPostgresCheckpointer(t, nil)
	ctx := context.Background()

	ck := newCheckpoint("thread-corrupt", "ckpt-corrupt", 1, richState())
	require.NoError(t, cp.Save(ctx, ck))

	// A JSON scalar where an object is expected: valid JSONB, wrong shape.
	require.NoError(t, cp.conn.ExecuteQuery(ctx,
		`UPDATE checkpoints SET state_data = '"not-an-object"'::jsonb WHERE id = $1`, ck.ID))

	_, err := cp.Load(ctx, ck.ThreadID, ck.ID)
	require.Error(t, err, "corrupt state must be reported, not silently dropped")
	assert.Contains(t, err.Error(), "unmarshal state")
}

// --- concurrency ----------------------------------------------------------

// TestPostgresCheckpointer_ConcurrentAccess drives real concurrent traffic
// through one pool. Run under -race it also covers the checkpointer's own
// shared state.
func TestPostgresCheckpointer_ConcurrentAccess(t *testing.T) {
	cp := newPostgresCheckpointer(t, nil)
	ctx := context.Background()

	const workers = 8
	const perWorker = 5

	var wg sync.WaitGroup
	errCh := make(chan error, workers*perWorker*2)

	for w := 0; w < workers; w++ {
		wg.Add(1)
		go func(w int) {
			defer wg.Done()
			threadID := fmt.Sprintf("concurrent-%d", w)
			for i := 0; i < perWorker; i++ {
				st := core.NewBaseState()
				st.Set("worker", w)
				st.Set("index", i)
				ck := newCheckpoint(threadID, fmt.Sprintf("ckpt-%d-%d", w, i), i, st)
				if err := cp.Save(ctx, ck); err != nil {
					errCh <- fmt.Errorf("save: %w", err)
					continue
				}
				got, err := cp.Load(ctx, threadID, ck.ID)
				if err != nil {
					errCh <- fmt.Errorf("load: %w", err)
					continue
				}
				if got.State.GetAll()["worker"] != float64(w) {
					errCh <- fmt.Errorf("worker %d read back %v", w, got.State.GetAll()["worker"])
				}
			}
		}(w)
	}

	wg.Wait()
	close(errCh)
	for err := range errCh {
		t.Error(err)
	}

	// Every worker's thread must hold exactly its own checkpoints.
	for w := 0; w < workers; w++ {
		list, err := cp.List(ctx, fmt.Sprintf("concurrent-%d", w))
		require.NoError(t, err)
		assert.Len(t, list, perWorker, "thread %d lost or gained checkpoints", w)
	}
}

// --- context propagation --------------------------------------------------

// TestPostgresCheckpointer_HonoursContextCancellation proves ctx actually
// reaches the driver rather than being accepted and ignored.
func TestPostgresCheckpointer_HonoursContextCancellation(t *testing.T) {
	cp := newPostgresCheckpointer(t, nil)

	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	err := cp.Save(ctx, newCheckpoint("thread-ctx", "ckpt-ctx", 1, core.NewBaseState()))
	require.Error(t, err)
	assert.ErrorIs(t, err, context.Canceled)

	_, err = cp.List(ctx, "thread-ctx")
	require.Error(t, err)
	assert.ErrorIs(t, err, context.Canceled)

	_, err = cp.Load(ctx, "thread-ctx", "ckpt-ctx")
	require.Error(t, err)
	assert.ErrorIs(t, err, context.Canceled)

	err = cp.Delete(ctx, "thread-ctx", "ckpt-ctx")
	require.Error(t, err)
	assert.ErrorIs(t, err, context.Canceled)
}

// --- integration with the graph-facing helpers ----------------------------

// TestPostgresCheckpointer_SaverAndLatest exercises the path a running graph
// actually takes: CheckpointSaver writes a checkpoint per step, Latest resumes
// from the newest one.
func TestPostgresCheckpointer_SaverAndLatest(t *testing.T) {
	cp := newPostgresCheckpointer(t, nil)
	ctx := context.Background()

	saver := NewCheckpointSaver(cp)
	threadID := "thread-saver"

	for step := 1; step <= 4; step++ {
		st := core.NewBaseState()
		st.Set("step", step)
		st.Set("payload", fmt.Sprintf("after-node-%d", step))
		require.NoError(t, saver.SaveState(ctx, threadID, fmt.Sprintf("node-%d", step), step, st))
	}

	latest, err := Latest(ctx, cp, threadID)
	require.NoError(t, err)
	require.NotNil(t, latest, "a thread with checkpoints must have a latest one")
	assert.EqualValues(t, 4, latest.State.GetAll()["step"], "Latest must return the highest step")
	assert.Equal(t, "after-node-4", latest.State.GetAll()["payload"])

	list, err := cp.List(ctx, threadID)
	require.NoError(t, err)
	assert.Len(t, list, 4, "one checkpoint per step")

	none, err := Latest(ctx, cp, "thread-with-nothing")
	require.NoError(t, err)
	assert.Nil(t, none)
}

// --- RAG document storage -------------------------------------------------

// TestPostgresCheckpointer_DocumentRoundTrip covers the non-vector RAG path.
// SaveDocument passed its metadata map straight to database/sql, which rejects
// maps -- so this method failed 100% of the time before the fix.
func TestPostgresCheckpointer_DocumentRoundTrip(t *testing.T) {
	cp := newPostgresCheckpointer(t, func(c *DatabaseConfig) { c.EnableRAG = true })
	ctx := context.Background()

	threadID := "thread-docs"
	doc := &Document{
		ID:        "doc-1",
		ThreadID:  threadID,
		Content:   "the quick brown fox",
		Metadata:  map[string]interface{}{"source": "unit-test", "page": 3.0},
		CreatedAt: time.Now().UTC().Truncate(time.Microsecond),
		UpdatedAt: time.Now().UTC().Truncate(time.Microsecond),
	}
	require.NoError(t, cp.SaveDocument(ctx, doc), "SaveDocument must accept a metadata map")

	docs, err := cp.SearchDocuments(ctx, threadID, nil, 10)
	require.NoError(t, err)
	require.Len(t, docs, 1)
	assert.Equal(t, "doc-1", docs[0].ID)
	assert.Equal(t, "the quick brown fox", docs[0].Content)
	assert.Equal(t, "unit-test", docs[0].Metadata["source"], "document metadata must round trip")
	assert.EqualValues(t, 3, docs[0].Metadata["page"])

	// Upsert on the same ID.
	doc.Content = "updated content"
	require.NoError(t, cp.SaveDocument(ctx, doc))
	docs, err = cp.SearchDocuments(ctx, threadID, nil, 10)
	require.NoError(t, err)
	require.Len(t, docs, 1, "re-saving the same document ID must upsert")
	assert.Equal(t, "updated content", docs[0].Content)

	// Documents from another thread must not leak into this one.
	other := *doc
	other.ID = "doc-other"
	other.ThreadID = "different-thread"
	require.NoError(t, cp.SaveDocument(ctx, &other))
	docs, err = cp.SearchDocuments(ctx, threadID, nil, 10)
	require.NoError(t, err)
	assert.Len(t, docs, 1, "SearchDocuments must be scoped to its thread")
}

func TestPostgresCheckpointer_DocumentNullMetadata(t *testing.T) {
	cp := newPostgresCheckpointer(t, func(c *DatabaseConfig) { c.EnableRAG = true })
	ctx := context.Background()

	// Register the thread, then insert a row with NULL metadata the way an
	// external writer would.
	require.NoError(t, cp.Save(ctx, newCheckpoint("thread-doc-null", "c", 1, core.NewBaseState())))
	require.NoError(t, cp.conn.ExecuteQuery(ctx,
		`INSERT INTO documents (id, thread_id, content, metadata, created_at, updated_at)
		 VALUES ('legacy-doc', $1, 'content', NULL, NOW(), NOW())`, "thread-doc-null"))

	docs, err := cp.SearchDocuments(ctx, "thread-doc-null", nil, 10)
	require.NoError(t, err, "a document with NULL metadata must not break the search")
	require.Len(t, docs, 1)
	assert.NotNil(t, docs[0].Metadata)
	assert.Empty(t, docs[0].Metadata)
}

func TestPostgresCheckpointer_RAGDisabledIsRejected(t *testing.T) {
	cp := newPostgresCheckpointer(t, nil) // EnableRAG defaults to false
	ctx := context.Background()

	err := cp.SaveDocument(ctx, &Document{ID: "d", ThreadID: "t", Content: "c"})
	require.Error(t, err)
	assert.Contains(t, err.Error(), "RAG is not enabled")

	_, err = cp.SearchDocuments(ctx, "t", nil, 5)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "RAG is not enabled")
}

// --- pgvector -------------------------------------------------------------

// TestPgVectorCheckpointer_EmbeddingRoundTrip covers the vector path against a
// real pgvector install. Two separate defects lived here: []float64 is not a
// valid driver argument (so both writes and vector searches always failed), and
// the scanned embedding was thrown away behind a "handle conversion if needed"
// comment, so every document read back had a nil Embedding.
func TestPgVectorCheckpointer_EmbeddingRoundTrip(t *testing.T) {
	requirePostgres(t)
	if !pgHasVector {
		t.Skip("pgvector extension is not installed in the test database; " +
			"run CREATE EXTENSION vector to enable this test")
	}

	cp := newPostgresCheckpointer(t, func(c *DatabaseConfig) {
		c.Type = DatabaseTypePgVector
		c.EnableRAG = true
		c.VectorDimension = 3
	})
	ctx := context.Background()

	threadID := "thread-vec"
	near := &Document{
		ID: "near", ThreadID: threadID, Content: "close match",
		Metadata: map[string]interface{}{"kind": "near"},
		// Deliberately non-integral so a lossy encoder would show up.
		Embedding: []float64{1.0, 0.25, -0.5},
		CreatedAt: time.Now().UTC(), UpdatedAt: time.Now().UTC(),
	}
	far := &Document{
		ID: "far", ThreadID: threadID, Content: "distant match",
		Metadata:  map[string]interface{}{"kind": "far"},
		Embedding: []float64{-9, -9, -9},
		CreatedAt: time.Now().UTC(), UpdatedAt: time.Now().UTC(),
	}
	require.NoError(t, cp.SaveDocument(ctx, near), "SaveDocument must accept a []float64 embedding")
	require.NoError(t, cp.SaveDocument(ctx, far))

	docs, err := cp.SearchDocuments(ctx, threadID, []float64{1.0, 0.25, -0.5}, 10)
	require.NoError(t, err, "vector similarity search must accept a []float64 query")
	require.Len(t, docs, 2)

	// Nearest first: this is what makes it a similarity search rather than an
	// arbitrary listing.
	assert.Equal(t, "near", docs[0].ID, "results must be ordered by vector distance")
	assert.Equal(t, "far", docs[1].ID)

	assert.Equal(t, []float64{1.0, 0.25, -0.5}, docs[0].Embedding,
		"the stored embedding must be decoded back, not discarded")
	assert.Equal(t, "near", docs[0].Metadata["kind"])
}

// --- SessionManager against a real server ---------------------------------

func TestSessionManager_ThreadAndSessionRoundTrip(t *testing.T) {
	cp := newPostgresCheckpointer(t, nil)
	ctx := context.Background()
	sm := NewSessionManager(cp.conn)

	thread := &Thread{
		ID:        "sm-thread",
		Name:      "conversation",
		Metadata:  map[string]interface{}{"locale": "en"},
		CreatedAt: time.Now().UTC().Truncate(time.Microsecond),
		UpdatedAt: time.Now().UTC().Truncate(time.Microsecond),
	}
	require.NoError(t, sm.CreateThread(ctx, thread))

	gotThread, err := sm.GetThread(ctx, thread.ID)
	require.NoError(t, err)
	assert.Equal(t, thread.Name, gotThread.Name)
	assert.Equal(t, "en", gotThread.Metadata["locale"])

	expires := time.Now().UTC().Add(time.Hour).Truncate(time.Microsecond)
	session := &Session{
		ID:        "sm-session",
		ThreadID:  thread.ID,
		UserID:    "user-1",
		Metadata:  map[string]interface{}{"agent": "test"},
		CreatedAt: time.Now().UTC().Truncate(time.Microsecond),
		ExpiresAt: &expires,
	}
	require.NoError(t, sm.CreateSession(ctx, session))

	gotSession, err := sm.GetSession(ctx, session.ID)
	require.NoError(t, err)
	assert.Equal(t, "user-1", gotSession.UserID)
	assert.Equal(t, thread.ID, gotSession.ThreadID)
	assert.Equal(t, "test", gotSession.Metadata["agent"])
	require.NotNil(t, gotSession.ExpiresAt)
	assert.WithinDuration(t, expires, *gotSession.ExpiresAt, time.Millisecond)

	_, err = sm.GetThread(ctx, "missing")
	assert.Error(t, err)
	_, err = sm.GetSession(ctx, "missing")
	assert.Error(t, err)
}

// --- connection handling --------------------------------------------------

// TestPostgresConnection_RejectsInvalidMaxLifetime covers a silent failure: an
// unparseable duration used to be swallowed, leaving connections with no
// lifetime cap instead of the configured one.
func TestPostgresConnection_RejectsInvalidMaxLifetime(t *testing.T) {
	cfg := requirePostgres(t)
	cfg.MaxLifetime = "5 minutes" // not a Go duration

	_, err := NewPostgresConnection(cfg)
	require.Error(t, err, "an unparseable max_lifetime must be reported, not ignored")
	assert.Contains(t, err.Error(), "max_lifetime")
}

func TestPostgresConnection_ExecAndTransaction(t *testing.T) {
	cfg := newPostgresSchema(t, nil)
	conn, err := NewPostgresConnection(cfg)
	require.NoError(t, err)
	defer func() { require.NoError(t, conn.Close()) }()

	ctx := context.Background()
	require.NoError(t, conn.ExecuteQuery(ctx, `CREATE TABLE t (id int primary key)`))

	res, err := conn.Exec(ctx, `INSERT INTO t (id) VALUES (1), (2)`)
	require.NoError(t, err)
	affected, err := res.RowsAffected()
	require.NoError(t, err)
	assert.EqualValues(t, 2, affected)

	// A transaction that returns an error must leave nothing behind.
	wantErr := fmt.Errorf("deliberate failure")
	err = conn.WithTx(ctx, func(tx *sql.Tx) error {
		if _, e := tx.ExecContext(ctx, `INSERT INTO t (id) VALUES (3)`); e != nil {
			return e
		}
		return wantErr
	})
	assert.ErrorIs(t, err, wantErr)

	row, err := asSQLRow(conn.QueryRow(ctx, `SELECT count(*) FROM t`))
	require.NoError(t, err)
	var count int
	require.NoError(t, row.Scan(&count))
	assert.Equal(t, 2, count, "a rolled back transaction must not persist its rows")
}

func TestDatabaseConnectionManager_RealConnectionLifecycle(t *testing.T) {
	cfg := newPostgresSchema(t, nil)
	mgr := NewDatabaseConnectionManager()

	require.NoError(t, mgr.AddConnection("primary", cfg))

	conn, err := mgr.GetConnection("primary")
	require.NoError(t, err)
	require.NoError(t, conn.Ping())
	assert.Equal(t, DatabaseTypePostgres, conn.GetType())

	// Re-adding under the same name must close the previous pool rather than
	// leaking it, and must hand back the new one.
	require.NoError(t, mgr.AddConnection("primary", cfg))
	replaced, err := mgr.GetConnection("primary")
	require.NoError(t, err)
	assert.NotSame(t, conn, replaced, "re-adding a name must install a new connection")
	assert.Error(t, conn.Ping(), "the replaced connection must have been closed")

	require.NoError(t, mgr.CloseAll())
	// CloseAll clears the registry, so a second call is a safe no-op rather
	// than a double close.
	require.NoError(t, mgr.CloseAll())
	_, err = mgr.GetConnection("primary")
	assert.Error(t, err)
}

// TestDatabaseConnectionManager_ConcurrentUse would crash the process with a
// "concurrent map read and map write" fatal error before the manager grew a
// mutex. Run under -race it also reports the data race itself.
func TestDatabaseConnectionManager_ConcurrentUse(t *testing.T) {
	cfg := newPostgresSchema(t, nil)
	mgr := NewDatabaseConnectionManager()
	t.Cleanup(func() {
		if err := mgr.CloseAll(); err != nil {
			t.Logf("CloseAll: %v", err)
		}
	})

	var wg sync.WaitGroup
	for i := 0; i < 4; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			name := fmt.Sprintf("conn-%d", i)
			if err := mgr.AddConnection(name, cfg); err != nil {
				t.Errorf("AddConnection: %v", err)
				return
			}
			if _, err := mgr.GetConnection(name); err != nil {
				t.Errorf("GetConnection: %v", err)
			}
		}(i)
	}
	for i := 0; i < 4; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			// Misses are expected; the point is that a concurrent read against
			// a concurrent write must not crash.
			_, _ = mgr.GetConnection("conn-0")
		}()
	}
	wg.Wait()
}
