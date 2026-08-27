// Copyright (c) 2024 GoLangGraph Team
//
// Licensed under the MIT License. See LICENSE file in the project root for full license information.
//
// Package: GoLangGraph - A powerful Go framework for building AI agent workflows

// Driver-level tests for PostgresCheckpointer.
//
// IMPORTANT: the type registered here is a database/sql DRIVER DOUBLE, not a
// real PostgreSQL server. It speaks the driver.Driver/Conn/Stmt/Rows protocol,
// so the checkpointer's own query construction, argument binding, row scanning
// and error handling all run for real -- but no SQL is parsed or executed.
// Behavioral coverage against a genuine server lives in
// postgres_integration_test.go; this file exists for the failure modes a real
// server will not produce on demand:
//
//   - a result set that fails partway through iteration, which is what
//     rows.Err() is for;
//   - a RowsAffected() that reports an error;
//   - the exact SQL text and bound parameters the checkpointer sends.
//
// These tests need no database and always run.

package persistence

import (
	"context"
	"database/sql"
	"database/sql/driver"
	"errors"
	"fmt"
	"io"
	"sync"
	"testing"
	"time"

	"github.com/sirupsen/logrus"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

const scriptDriverName = "golanggraph_script_test"

func init() {
	sql.Register(scriptDriverName, scriptDriver{})
}

// recordedCall captures one statement the checkpointer sent.
type recordedCall struct {
	query string
	args  []driver.NamedValue
}

// script tells the double how to answer. One script is registered per test
// under a unique DSN, which is how a connection finds its instructions.
type script struct {
	mu sync.Mutex

	columns []string
	rows    [][]driver.Value

	// failAfter rows have been handed over, Next returns failErr instead of
	// io.EOF. A negative value means the result set ends cleanly.
	failAfter int
	failErr   error

	queryErr error

	execAffected int64
	execErr      error // returned by RowsAffected, not by Exec itself

	calls []recordedCall
}

func (s *script) record(query string, args []driver.NamedValue) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.calls = append(s.calls, recordedCall{query: query, args: args})
}

func (s *script) recorded() []recordedCall {
	s.mu.Lock()
	defer s.mu.Unlock()
	out := make([]recordedCall, len(s.calls))
	copy(out, s.calls)
	return out
}

var (
	scriptsMu sync.Mutex
	scripts   = map[string]*script{}
	scriptSeq int
)

// newScriptedCheckpointer wires a real PostgresCheckpointer on top of the
// double. It deliberately bypasses NewPostgresCheckpointer, whose Connect hard-
// codes the "postgres" driver name and would also try to create the schema.
// Everything below the connection -- the checkpointer's own logic -- is real.
func newScriptedCheckpointer(t *testing.T, s *script, cfg *DatabaseConfig) *PostgresCheckpointer {
	t.Helper()

	scriptsMu.Lock()
	scriptSeq++
	dsn := fmt.Sprintf("script-%d", scriptSeq)
	scripts[dsn] = s
	scriptsMu.Unlock()

	t.Cleanup(func() {
		scriptsMu.Lock()
		delete(scripts, dsn)
		scriptsMu.Unlock()
	})

	db, err := sql.Open(scriptDriverName, dsn)
	require.NoError(t, err)
	t.Cleanup(func() { _ = db.Close() })

	if cfg == nil {
		cfg = NewPostgresConfig("fake", 5432, "fake", "fake", "fake")
	}

	return &PostgresCheckpointer{
		conn:   &PostgresConnection{db: db, config: cfg, logger: logrus.New()},
		config: cfg,
		logger: logrus.New(),
	}
}

// --- the double -----------------------------------------------------------

type scriptDriver struct{}

func (scriptDriver) Open(dsn string) (driver.Conn, error) {
	scriptsMu.Lock()
	s, ok := scripts[dsn]
	scriptsMu.Unlock()
	if !ok {
		return nil, fmt.Errorf("no script registered for %q", dsn)
	}
	return &scriptConn{s: s}, nil
}

type scriptConn struct{ s *script }

// Implementing QueryerContext and ExecerContext lets database/sql hand us the
// statement and its bound arguments directly, which is what makes the
// assertions on generated SQL possible.
var (
	_ driver.QueryerContext = (*scriptConn)(nil)
	_ driver.ExecerContext  = (*scriptConn)(nil)
)

func (c *scriptConn) QueryContext(ctx context.Context, query string, args []driver.NamedValue) (driver.Rows, error) {
	c.s.record(query, args)
	if c.s.queryErr != nil {
		return nil, c.s.queryErr
	}
	return &scriptRows{s: c.s}, nil
}

func (c *scriptConn) ExecContext(ctx context.Context, query string, args []driver.NamedValue) (driver.Result, error) {
	c.s.record(query, args)
	return scriptResult{affected: c.s.execAffected, err: c.s.execErr}, nil
}

func (c *scriptConn) Prepare(query string) (driver.Stmt, error) {
	return &scriptStmt{c: c, query: query}, nil
}
func (c *scriptConn) Close() error              { return nil }
func (c *scriptConn) Begin() (driver.Tx, error) { return scriptTx{}, nil }

type scriptTx struct{}

func (scriptTx) Commit() error   { return nil }
func (scriptTx) Rollback() error { return nil }

type scriptStmt struct {
	c     *scriptConn
	query string
}

func (s *scriptStmt) Close() error  { return nil }
func (s *scriptStmt) NumInput() int { return -1 } // -1 disables arity checking
func (s *scriptStmt) Exec(args []driver.Value) (driver.Result, error) {
	return scriptResult{affected: s.c.s.execAffected, err: s.c.s.execErr}, nil
}
func (s *scriptStmt) Query(args []driver.Value) (driver.Rows, error) {
	return &scriptRows{s: s.c.s}, nil
}

type scriptResult struct {
	affected int64
	err      error
}

func (r scriptResult) LastInsertId() (int64, error) { return 0, nil }
func (r scriptResult) RowsAffected() (int64, error) { return r.affected, r.err }

type scriptRows struct {
	s   *script
	pos int
}

func (r *scriptRows) Columns() []string { return r.s.columns }
func (r *scriptRows) Close() error      { return nil }

func (r *scriptRows) Next(dest []driver.Value) error {
	if r.s.failAfter >= 0 && r.pos >= r.s.failAfter {
		// A non-EOF error here is exactly what a connection dropping mid-result
		// looks like to database/sql: Rows.Next reports false and the reason is
		// only available from Rows.Err.
		return r.s.failErr
	}
	if r.pos >= len(r.s.rows) {
		return io.EOF
	}
	copy(dest, r.s.rows[r.pos])
	r.pos++
	return nil
}

// checkpointRow builds one row shaped like the List query's SELECT list:
// id, thread_id, metadata, created_at, node_id, step_id.
func checkpointRow(id, threadID string, nodeID interface{}, stepID interface{}) []driver.Value {
	return []driver.Value{
		id,
		threadID,
		[]byte(`{"k":"v"}`),
		time.Date(2024, 1, 1, 12, 0, 0, 0, time.UTC),
		nodeID,
		stepID,
	}
}

var checkpointColumns = []string{"id", "thread_id", "metadata", "created_at", "node_id", "step_id"}

// --- rows.Err() -----------------------------------------------------------

// TestPostgresCheckpointer_ListSurfacesIterationError covers a silent
// truncation bug: List iterated with `for rows.Next()` and returned without ever
// calling rows.Err(). A connection that dropped after two of five rows produced
// a two-element slice and a nil error, so the caller could not tell a partial
// answer from a complete one -- and Latest() would happily "resume" from a
// checkpoint that was not actually the newest.
func TestPostgresCheckpointer_ListSurfacesIterationError(t *testing.T) {
	boom := errors.New("connection reset by peer")
	s := &script{
		columns: checkpointColumns,
		rows: [][]driver.Value{
			checkpointRow("ckpt-1", "t1", "node-1", int64(1)),
			checkpointRow("ckpt-2", "t1", "node-2", int64(2)),
			checkpointRow("ckpt-3", "t1", "node-3", int64(3)),
		},
		failAfter: 2, // two rows arrive, then the connection dies
		failErr:   boom,
	}

	cp := newScriptedCheckpointer(t, s, nil)

	got, err := cp.List(context.Background(), "t1")
	require.Error(t, err, "a result set that fails mid-iteration must not look like a complete listing")
	assert.ErrorIs(t, err, boom)
	assert.Nil(t, got, "no partial slice may be handed back alongside the error")
}

// TestPostgresCheckpointer_SearchDocumentsSurfacesIterationError is the same
// unchecked-rows.Err() defect in the RAG search path.
func TestPostgresCheckpointer_SearchDocumentsSurfacesIterationError(t *testing.T) {
	boom := errors.New("server closed the connection unexpectedly")
	s := &script{
		columns: []string{"id", "thread_id", "content", "metadata", "created_at", "updated_at"},
		rows: [][]driver.Value{
			{"doc-1", "t1", "content one", []byte(`{}`), time.Now().UTC(), time.Now().UTC()},
			{"doc-2", "t1", "content two", []byte(`{}`), time.Now().UTC(), time.Now().UTC()},
		},
		failAfter: 1,
		failErr:   boom,
	}

	cfg := NewPostgresConfig("fake", 5432, "fake", "fake", "fake")
	cfg.EnableRAG = true
	cp := newScriptedCheckpointer(t, s, cfg)

	got, err := cp.SearchDocuments(context.Background(), "t1", nil, 10)
	require.Error(t, err, "a truncated document search must be reported")
	assert.ErrorIs(t, err, boom)
	assert.Nil(t, got)
}

// TestPostgresCheckpointer_ListSucceedsOnCleanIteration is the control for the
// two tests above: with no injected failure the same code path must return
// every row and no error, so the checks are not just failing on principle.
func TestPostgresCheckpointer_ListSucceedsOnCleanIteration(t *testing.T) {
	s := &script{
		columns: checkpointColumns,
		rows: [][]driver.Value{
			checkpointRow("ckpt-1", "t1", "node-1", int64(1)),
			// Second row exercises NULL node_id/step_id at the driver level.
			checkpointRow("ckpt-2", "t1", nil, nil),
		},
		failAfter: -1,
	}

	cp := newScriptedCheckpointer(t, s, nil)

	got, err := cp.List(context.Background(), "t1")
	require.NoError(t, err)
	require.Len(t, got, 2)

	assert.Equal(t, "ckpt-1", got[0].ID)
	assert.Equal(t, "node-1", got[0].NodeID)
	assert.Equal(t, 1, got[0].StepID)
	assert.Equal(t, "v", got[0].Metadata["k"], "metadata must be decoded from the raw column bytes")

	assert.Equal(t, "", got[1].NodeID, "a NULL node_id must scan to the empty string")
	assert.Equal(t, 0, got[1].StepID, "a NULL step_id must scan to zero")
}

// --- generated SQL and bound parameters -----------------------------------

// TestPostgresCheckpointer_ListBindsThreadIDAsParameter pins the shape of the
// generated statement. The thread ID must travel as a bound parameter rather
// than being concatenated into the SQL text, which is what keeps a hostile
// thread ID inert.
func TestPostgresCheckpointer_ListBindsThreadIDAsParameter(t *testing.T) {
	s := &script{columns: checkpointColumns, failAfter: -1}
	cp := newScriptedCheckpointer(t, s, nil)

	hostile := `t1'; DROP TABLE checkpoints; --`
	_, err := cp.List(context.Background(), hostile)
	require.NoError(t, err)

	calls := s.recorded()
	require.Len(t, calls, 1)

	assert.Contains(t, calls[0].query, "FROM checkpoints")
	assert.Contains(t, calls[0].query, "WHERE thread_id = $1")
	assert.Contains(t, calls[0].query, "ORDER BY created_at DESC")
	assert.NotContains(t, calls[0].query, "DROP TABLE",
		"the thread ID must never be interpolated into the SQL text")

	require.Len(t, calls[0].args, 1)
	assert.Equal(t, hostile, calls[0].args[0].Value, "the thread ID must arrive as a bound parameter")
}

// TestPostgresCheckpointer_DeleteBindsBothIdentifiers checks the delete is
// scoped by thread as well as by checkpoint ID, so one thread cannot delete
// another's checkpoint by guessing its ID.
func TestPostgresCheckpointer_DeleteBindsBothIdentifiers(t *testing.T) {
	s := &script{execAffected: 1}
	cp := newScriptedCheckpointer(t, s, nil)

	require.NoError(t, cp.Delete(context.Background(), "thread-9", "ckpt-9"))

	calls := s.recorded()
	require.Len(t, calls, 1)
	assert.Contains(t, calls[0].query, "DELETE FROM checkpoints")
	assert.Contains(t, calls[0].query, "thread_id = $1")
	assert.Contains(t, calls[0].query, "id = $2")

	require.Len(t, calls[0].args, 2)
	assert.Equal(t, "thread-9", calls[0].args[0].Value)
	assert.Equal(t, "ckpt-9", calls[0].args[1].Value)
}

// TestPostgresCheckpointer_SaveRegistersThreadBeforeCheckpoint pins the fix for
// the foreign key violation: the parent thread row must be written first, and
// both statements must go out together so a failed checkpoint cannot leave an
// orphan thread behind.
func TestPostgresCheckpointer_SaveRegistersThreadBeforeCheckpoint(t *testing.T) {
	s := &script{execAffected: 1}
	cp := newScriptedCheckpointer(t, s, nil)

	ck := newCheckpoint("thread-x", "ckpt-x", 2, richState())
	require.NoError(t, cp.Save(context.Background(), ck))

	calls := s.recorded()
	require.Len(t, calls, 2, "Save must issue the thread upsert and the checkpoint upsert")

	assert.Contains(t, calls[0].query, "INSERT INTO threads")
	assert.Contains(t, calls[0].query, "ON CONFLICT (id) DO NOTHING",
		"registering an existing thread must not be an error")
	require.Len(t, calls[0].args, 1)
	assert.Equal(t, "thread-x", calls[0].args[0].Value)

	assert.Contains(t, calls[1].query, "INSERT INTO checkpoints")
	assert.Contains(t, calls[1].query, "ON CONFLICT (id) DO UPDATE")
	require.Len(t, calls[1].args, 7)
	assert.Equal(t, "ckpt-x", calls[1].args[0].Value)
	assert.Equal(t, "thread-x", calls[1].args[1].Value)

	// The state must reach the driver as real JSON. This is the same regression
	// the integration tests cover, asserted one layer lower: before BaseState
	// implemented json.Marshaler this argument was the two bytes "{}".
	stateArg, ok := calls[1].args[2].Value.([]byte)
	require.True(t, ok, "state should be bound as raw JSON bytes, got %T", calls[1].args[2].Value)
	assert.NotEqual(t, "{}", string(stateArg), "state was serialized as an empty object -- all state lost")
	assert.Contains(t, string(stateArg), "hello world")
}

// --- error handling in results --------------------------------------------

// TestPostgresCheckpointer_DeleteReportsRowsAffectedError covers the branch
// where the driver cannot say how many rows were removed. Reporting success
// there would claim a deletion that may not have happened.
func TestPostgresCheckpointer_DeleteReportsRowsAffectedError(t *testing.T) {
	boom := errors.New("no RowsAffected available")
	s := &script{execErr: boom}
	cp := newScriptedCheckpointer(t, s, nil)

	err := cp.Delete(context.Background(), "t", "c")
	require.Error(t, err)
	assert.ErrorIs(t, err, boom)
}

// TestPostgresCheckpointer_DeleteMissingRowIsNotFound is the driver-level twin
// of the integration test: zero rows affected must be an error.
func TestPostgresCheckpointer_DeleteMissingRowIsNotFound(t *testing.T) {
	s := &script{execAffected: 0}
	cp := newScriptedCheckpointer(t, s, nil)

	err := cp.Delete(context.Background(), "t", "c")
	require.Error(t, err)
	assert.Contains(t, err.Error(), "not found")
}

func TestPostgresCheckpointer_ListReportsQueryError(t *testing.T) {
	boom := errors.New("relation \"checkpoints\" does not exist")
	s := &script{columns: checkpointColumns, failAfter: -1, queryErr: boom}
	cp := newScriptedCheckpointer(t, s, nil)

	_, err := cp.List(context.Background(), "t1")
	require.Error(t, err)
	assert.ErrorIs(t, err, boom)
}

// --- guards against non-SQL connections -----------------------------------

// TestAsSQLRows_RejectsForeignTypes covers the helper that replaced inline
// `rows.(*sql.Rows)` assertions. Those are single-value assertions, so any
// DatabaseConnection implementation other than PostgresConnection -- the
// interface returns interface{}, so others are allowed -- panicked and took the
// process down instead of returning an error.
func TestAsSQLRows_RejectsForeignTypes(t *testing.T) {
	_, err := asSQLRows(nil)
	assert.Error(t, err, "nil must be an error, not a panic")

	_, err = asSQLRows("not rows")
	require.Error(t, err)
	assert.Contains(t, err.Error(), "want *sql.Rows")

	var typedNil *sql.Rows
	_, err = asSQLRows(typedNil)
	assert.Error(t, err, "a typed nil must be rejected too, or the caller nil-derefs")
}

func TestAsSQLRow_RejectsForeignTypes(t *testing.T) {
	_, err := asSQLRow(nil)
	assert.Error(t, err)

	_, err = asSQLRow(struct{}{})
	require.Error(t, err)
	assert.Contains(t, err.Error(), "want *sql.Row")
}

// --- pure helpers ---------------------------------------------------------

func TestEncodeDecodeVector(t *testing.T) {
	cases := [][]float64{
		{1, 2, 3},
		{-0.5, 0.25, 1e-7},
		{0},
	}
	for _, want := range cases {
		encoded := encodeVector(want)
		got, err := decodeVector(encoded)
		require.NoError(t, err)
		assert.Equal(t, want, got, "vector must survive encode/decode exactly")
	}

	assert.Equal(t, "[]", encodeVector(nil))

	// SQL NULL must stay absent rather than becoming an empty slice, so callers
	// can tell "no embedding stored" from "a zero-length embedding".
	got, err := decodeVector(nil)
	require.NoError(t, err)
	assert.Nil(t, got)

	got, err = decodeVector([]byte("[1,2]"))
	require.NoError(t, err)
	assert.Equal(t, []float64{1, 2}, got, "a []byte column value must decode as well as a string")

	_, err = decodeVector("[1,notanumber]")
	assert.Error(t, err, "a malformed vector must be reported, not silently truncated")

	_, err = decodeVector(42)
	assert.Error(t, err, "an unexpected column type must be reported")
}

func TestDecodeJSONMap(t *testing.T) {
	var m map[string]interface{}

	// SQL NULL arrives as a nil slice; the JSON literal null is also possible.
	require.NoError(t, decodeJSONMap(nil, &m))
	assert.NotNil(t, m, "NULL must decode to an empty map, never nil -- callers write into this map")
	assert.Empty(t, m)

	require.NoError(t, decodeJSONMap([]byte("null"), &m))
	assert.NotNil(t, m)

	require.NoError(t, decodeJSONMap([]byte(`{"a":1}`), &m))
	assert.EqualValues(t, 1, m["a"])

	assert.Error(t, decodeJSONMap([]byte("{broken"), &m))
}

// TestDecodeJSONMap_ResultIsWritable guards the reason NULL must not decode to
// nil: writing to a nil map panics, and callers treat checkpoint metadata as a
// normal map.
func TestDecodeJSONMap_ResultIsWritable(t *testing.T) {
	var m map[string]interface{}
	require.NoError(t, decodeJSONMap(nil, &m))

	assert.NotPanics(t, func() { m["added"] = true })
	assert.Equal(t, true, m["added"])
}
