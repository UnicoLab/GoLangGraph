// Copyright (c) 2024 GoLangGraph Team
//
// Licensed under the MIT License. See LICENSE file in the project root for full license information.
//
// Package: GoLangGraph - A powerful Go framework for building AI agent workflows

// Integration tests that run RedisCheckpointer against a REAL Redis server.
// The client, the key layout, the TTLs and the JSON payloads are all exercised
// for real; nothing is stubbed.
//
// The server is taken from REDIS_TEST_ADDR when set (an unreachable explicit
// address is a hard failure), otherwise 127.0.0.1:6379 is probed and every test
// here skips cleanly if nothing answers.
//
// Redis has no schemas, so isolation comes from a per-test thread-ID prefix and
// a cleanup that deletes exactly the keys a test created. No test ever flushes
// the database, which would destroy unrelated data on a shared server.

package persistence

import (
	"context"
	"fmt"
	"os"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/UnicoLab/GoLangGraph/pkg/core"
)

const defaultRedisTestAddr = "127.0.0.1:6379"

var (
	redisDiscoverOnce sync.Once
	redisBaseConfig   *DatabaseConfig
	redisDiscoverErr  error
	redisExplicitAddr bool
)

func parseRedisAddr(addr string) (*DatabaseConfig, error) {
	host, portStr, found := strings.Cut(addr, ":")
	if !found {
		return nil, fmt.Errorf("invalid redis address %q, want host:port", addr)
	}
	var port int
	if _, err := fmt.Sscanf(portStr, "%d", &port); err != nil {
		return nil, fmt.Errorf("invalid port in redis address %q: %w", addr, err)
	}
	// Password comes from REDIS_TEST_PASSWORD; sending a password to a server
	// with none configured is itself an error, so the default is empty.
	return NewRedisConfig(host, port, os.Getenv("REDIS_TEST_PASSWORD")), nil
}

func discoverRedis() (*DatabaseConfig, error) {
	redisDiscoverOnce.Do(func() {
		addr := defaultRedisTestAddr
		if explicit := os.Getenv("REDIS_TEST_ADDR"); explicit != "" {
			addr = explicit
			redisExplicitAddr = true
		}

		cfg, err := parseRedisAddr(addr)
		if err != nil {
			redisDiscoverErr = err
			return
		}
		cp, err := NewRedisCheckpointer(cfg)
		if err != nil {
			redisDiscoverErr = err
			return
		}
		_ = cp.Close()
		redisBaseConfig = cfg
	})
	return redisBaseConfig, redisDiscoverErr
}

func requireRedisConfig(t *testing.T) *DatabaseConfig {
	t.Helper()

	cfg, err := discoverRedis()
	if cfg == nil {
		if redisExplicitAddr {
			t.Fatalf("REDIS_TEST_ADDR is set but the server is unreachable: %v", err)
		}
		t.Skipf("no local Redis reachable at %s (%v). "+
			"Set REDIS_TEST_ADDR to run these integration tests.", defaultRedisTestAddr, err)
	}
	clone := *cfg
	return &clone
}

// newRedisCheckpointer returns a checkpointer plus a prefix unique to this test.
// Cleanup removes every key under that prefix, so a shared Redis is left exactly
// as it was found.
func newRedisCheckpointer(t *testing.T, mutate func(*DatabaseConfig)) (*RedisCheckpointer, string) {
	t.Helper()

	cfg := requireRedisConfig(t)
	if mutate != nil {
		mutate(cfg)
	}

	cp, err := NewRedisCheckpointer(cfg)
	require.NoError(t, err, "connect to Redis")

	prefix := fmt.Sprintf("gltest:%s:%d:", t.Name(), time.Now().UnixNano())

	t.Cleanup(func() {
		ctx := context.Background()
		// Match both the checkpoint payloads and the thread index sets. The
		// prefix contains ':' so it is escaped in keys exactly as the
		// checkpointer escapes it.
		for _, pattern := range []string{
			"checkpoint:" + redisKeySegment(prefix) + "*",
			"thread:" + redisKeySegment(prefix) + "*",
		} {
			keys, err := cp.client.Keys(ctx, pattern).Result()
			if err != nil {
				t.Logf("cleanup scan %q: %v", pattern, err)
				continue
			}
			if len(keys) > 0 {
				if err := cp.client.Del(ctx, keys...).Err(); err != nil {
					t.Logf("cleanup delete: %v", err)
				}
			}
		}
		if err := cp.Close(); err != nil {
			t.Logf("failed to close redis checkpointer: %v", err)
		}
	})

	return cp, prefix
}

// --- core round trip ------------------------------------------------------

// TestRedisCheckpointer_SaveLoadRoundTripPreservesState is the reason this file
// exists. BaseState stores its data in unexported fields, so before it gained
// MarshalJSON/UnmarshalJSON every checkpoint went to Redis as "{}" and came
// back empty. This proves a real server returns the real state.
func TestRedisCheckpointer_SaveLoadRoundTripPreservesState(t *testing.T) {
	cp, prefix := newRedisCheckpointer(t, nil)
	ctx := context.Background()

	want := newCheckpoint(prefix+"round-trip", "ckpt-1", 3, richState())
	require.NoError(t, cp.Save(ctx, want))

	got, err := cp.Load(ctx, want.ThreadID, want.ID)
	require.NoError(t, err)

	assert.Equal(t, want.ID, got.ID)
	assert.Equal(t, want.ThreadID, got.ThreadID)
	assert.Equal(t, want.NodeID, got.NodeID)
	assert.Equal(t, want.StepID, got.StepID)
	assert.WithinDuration(t, want.CreatedAt, got.CreatedAt, time.Millisecond)
	assert.Equal(t, "n3", got.Metadata["node"])
	assertRichState(t, got.State)
}

// TestRedisCheckpointer_StoresRealJSONNotEmptyObject inspects the stored bytes
// directly, so it holds even if Load were changed to fabricate a state.
func TestRedisCheckpointer_StoresRealJSONNotEmptyObject(t *testing.T) {
	cp, prefix := newRedisCheckpointer(t, nil)
	ctx := context.Background()

	ck := newCheckpoint(prefix+"json", "ckpt-json", 1, richState())
	require.NoError(t, cp.Save(ctx, ck))

	raw, err := cp.client.Get(ctx, redisCheckpointKey(ck.ThreadID, ck.ID)).Result()
	require.NoError(t, err)

	assert.NotContains(t, raw, `"state":{}`, "state was stored as an empty object -- all state lost")
	assert.Contains(t, raw, "hello world")
	assert.Contains(t, raw, "integration-test", "state metadata must be persisted too")
}

func TestRedisCheckpointer_SaveOverwritesSameID(t *testing.T) {
	cp, prefix := newRedisCheckpointer(t, nil)
	ctx := context.Background()

	threadID := prefix + "overwrite"
	first := core.NewBaseState()
	first.Set("version", "one")
	ck := newCheckpoint(threadID, "same", 1, first)
	require.NoError(t, cp.Save(ctx, ck))

	second := core.NewBaseState()
	second.Set("version", "two")
	ck.State = second
	require.NoError(t, cp.Save(ctx, ck))

	got, err := cp.Load(ctx, threadID, "same")
	require.NoError(t, err)
	assert.Equal(t, "two", got.State.GetAll()["version"])

	list, err := cp.List(ctx, threadID)
	require.NoError(t, err)
	assert.Len(t, list, 1, "the thread index must not gain a duplicate entry")
}

// --- isolation ------------------------------------------------------------

func TestRedisCheckpointer_ThreadIsolation(t *testing.T) {
	cp, prefix := newRedisCheckpointer(t, nil)
	ctx := context.Background()

	a := core.NewBaseState()
	a.Set("owner", "alice")
	b := core.NewBaseState()
	b.Set("owner", "bob")

	require.NoError(t, cp.Save(ctx, newCheckpoint(prefix+"thread-a", "shared-id", 1, a)))
	require.NoError(t, cp.Save(ctx, newCheckpoint(prefix+"thread-b", "shared-id", 1, b)))

	gotA, err := cp.Load(ctx, prefix+"thread-a", "shared-id")
	require.NoError(t, err)
	assert.Equal(t, "alice", gotA.State.GetAll()["owner"], "threads sharing a checkpoint ID must not share data")

	gotB, err := cp.Load(ctx, prefix+"thread-b", "shared-id")
	require.NoError(t, err)
	assert.Equal(t, "bob", gotB.State.GetAll()["owner"])

	listA, err := cp.List(ctx, prefix+"thread-a")
	require.NoError(t, err)
	assert.Len(t, listA, 1, "one thread's index must not include the other's checkpoints")
}

// TestRedisCheckpointer_ColonInIdentifiersDoesNotCollide covers a cross-thread
// data leak. Keys were built as fmt.Sprintf("checkpoint:%s:%s", threadID, id),
// so thread "x:a" + checkpoint "b:c1" and thread "x:a:b" + checkpoint "c1" both
// produced the key "checkpoint:x:a:b:c1": each thread read and overwrote the
// other's state. Thread IDs are routinely derived from user or session
// identifiers, which makes this a tenant-isolation failure, not a curiosity.
func TestRedisCheckpointer_ColonInIdentifiersDoesNotCollide(t *testing.T) {
	cp, prefix := newRedisCheckpointer(t, nil)
	ctx := context.Background()

	stateA := core.NewBaseState()
	stateA.Set("which", "A")
	stateB := core.NewBaseState()
	stateB.Set("which", "B")

	// These two (threadID, checkpointID) pairs concatenate to the same string.
	a := newCheckpoint(prefix+"x:a", "b:c1", 1, stateA)
	b := newCheckpoint(prefix+"x:a:b", "c1", 1, stateB)

	require.NoError(t, cp.Save(ctx, a))
	require.NoError(t, cp.Save(ctx, b))

	gotA, err := cp.Load(ctx, a.ThreadID, a.ID)
	require.NoError(t, err)
	assert.Equal(t, "A", gotA.State.GetAll()["which"],
		"checkpoint A was overwritten by B -- colliding Redis keys leak state across threads")

	gotB, err := cp.Load(ctx, b.ThreadID, b.ID)
	require.NoError(t, err)
	assert.Equal(t, "B", gotB.State.GetAll()["which"])

	// The two thread indexes must also stay separate.
	listA, err := cp.List(ctx, a.ThreadID)
	require.NoError(t, err)
	require.Len(t, listA, 1)
	assert.Equal(t, "b:c1", listA[0].ID)

	listB, err := cp.List(ctx, b.ThreadID)
	require.NoError(t, err)
	require.Len(t, listB, 1)
	assert.Equal(t, "c1", listB[0].ID)
}

// TestRedisKeySegment_EscapingIsUnambiguous checks the escaping directly, and
// pins the property that ordinary identifiers keep their original key bytes so
// data written before the fix is still readable.
func TestRedisKeySegment_EscapingIsUnambiguous(t *testing.T) {
	assert.Equal(t, "plain-id_123", redisKeySegment("plain-id_123"),
		"identifiers without ':' or '%' must be left byte-identical")
	assert.Equal(t, "checkpoint:thread:ckpt", redisCheckpointKey("thread", "ckpt"))

	assert.NotEqual(t,
		redisCheckpointKey("x:a", "b:c"),
		redisCheckpointKey("x:a:b", "c"),
		"identifier boundaries must survive escaping")

	// '%' is escaped too, so an attacker cannot hand-craft an identifier whose
	// escaped form equals another pair's.
	assert.NotEqual(t,
		redisCheckpointKey("a%3Ab", "c"),
		redisCheckpointKey("a:b", "c"))
}

// --- listing and deletion -------------------------------------------------

func TestRedisCheckpointer_ListReturnsEveryCheckpoint(t *testing.T) {
	cp, prefix := newRedisCheckpointer(t, nil)
	ctx := context.Background()

	threadID := prefix + "list"
	const n = 5
	for i := 0; i < n; i++ {
		st := core.NewBaseState()
		st.Set("i", i)
		require.NoError(t, cp.Save(ctx, newCheckpoint(threadID, fmt.Sprintf("ckpt-%d", i), i, st)))
	}

	list, err := cp.List(ctx, threadID)
	require.NoError(t, err)
	require.Len(t, list, n)

	seen := map[string]bool{}
	for _, m := range list {
		seen[m.ID] = true
		assert.Equal(t, threadID, m.ThreadID)
	}
	assert.Len(t, seen, n, "every checkpoint must appear exactly once")

	empty, err := cp.List(ctx, prefix+"never-used")
	require.NoError(t, err, "listing an unknown thread is not an error")
	assert.Empty(t, empty)
}

func TestRedisCheckpointer_Delete(t *testing.T) {
	cp, prefix := newRedisCheckpointer(t, nil)
	ctx := context.Background()

	threadID := prefix + "delete"
	require.NoError(t, cp.Save(ctx, newCheckpoint(threadID, "ckpt", 1, core.NewBaseState())))
	require.NoError(t, cp.Delete(ctx, threadID, "ckpt"))

	_, err := cp.Load(ctx, threadID, "ckpt")
	assert.Error(t, err)

	list, err := cp.List(ctx, threadID)
	require.NoError(t, err)
	assert.Empty(t, list, "Delete must also remove the thread index entry")
}

// TestRedisCheckpointer_DeleteMissingReportsNotFound covers a defect where
// deleting something that was not there reported success. The memory and file
// backends both return an error, so callers could not rely on the result.
func TestRedisCheckpointer_DeleteMissingReportsNotFound(t *testing.T) {
	cp, prefix := newRedisCheckpointer(t, nil)
	ctx := context.Background()

	err := cp.Delete(ctx, prefix+"nothing-here", "ckpt")
	require.Error(t, err, "deleting a missing checkpoint must not report success")
	assert.Contains(t, err.Error(), "not found")
}

// --- damaged data ---------------------------------------------------------

// TestRedisCheckpointer_CorruptPayload pins deliberate behaviour: a single
// unreadable entry must not make the whole thread unlistable (and therefore
// unresumable), but reading it directly must still report the problem.
func TestRedisCheckpointer_CorruptPayload(t *testing.T) {
	cp, prefix := newRedisCheckpointer(t, nil)
	ctx := context.Background()

	threadID := prefix + "corrupt"
	require.NoError(t, cp.Save(ctx, newCheckpoint(threadID, "good", 1, richState())))

	// Write a broken payload and index it the way Save would.
	require.NoError(t, cp.client.Set(ctx,
		redisCheckpointKey(threadID, "bad"), "{not json", time.Minute).Err())
	require.NoError(t, cp.client.SAdd(ctx, redisThreadIndexKey(threadID), "bad").Err())

	_, err := cp.Load(ctx, threadID, "bad")
	require.Error(t, err, "a corrupt payload must be reported when loaded directly")
	assert.Contains(t, err.Error(), "unmarshal")

	list, err := cp.List(ctx, threadID)
	require.NoError(t, err, "one bad entry must not make the thread unlistable")
	require.Len(t, list, 1, "the readable checkpoint must still be returned")
	assert.Equal(t, "good", list[0].ID)
}

// TestRedisCheckpointer_StaleIndexEntryIsSkipped simulates the normal
// consequence of TTL expiry: the index still names a checkpoint whose payload
// is gone.
func TestRedisCheckpointer_StaleIndexEntryIsSkipped(t *testing.T) {
	cp, prefix := newRedisCheckpointer(t, nil)
	ctx := context.Background()

	threadID := prefix + "stale"
	require.NoError(t, cp.Save(ctx, newCheckpoint(threadID, "alive", 1, core.NewBaseState())))
	require.NoError(t, cp.client.SAdd(ctx, redisThreadIndexKey(threadID), "expired").Err())

	list, err := cp.List(ctx, threadID)
	require.NoError(t, err)
	require.Len(t, list, 1)
	assert.Equal(t, "alive", list[0].ID)

	// Deleting a stale entry must clean the index even though the payload is
	// already gone, so the index cannot accumulate dead members forever.
	assert.Error(t, cp.Delete(ctx, threadID, "expired"), "the payload really is missing")
	members, err := cp.client.SMembers(ctx, redisThreadIndexKey(threadID)).Result()
	require.NoError(t, err)
	assert.NotContains(t, members, "expired", "Delete must drop the stale index entry")
}

// --- TTL handling ---------------------------------------------------------

// TestRedisCheckpointer_ThreadIndexExpiresWithCheckpoints covers an unbounded
// leak: checkpoint payloads were written with a TTL but the thread index set was
// created with none, so it survived forever, growing a dead member per expired
// checkpoint and costing List a wasted round trip for each.
func TestRedisCheckpointer_ThreadIndexExpiresWithCheckpoints(t *testing.T) {
	cp, prefix := newRedisCheckpointer(t, func(c *DatabaseConfig) { c.CheckpointTTL = "90s" })
	ctx := context.Background()

	threadID := prefix + "ttl"
	require.NoError(t, cp.Save(ctx, newCheckpoint(threadID, "ckpt", 1, core.NewBaseState())))

	payloadTTL, err := cp.client.TTL(ctx, redisCheckpointKey(threadID, "ckpt")).Result()
	require.NoError(t, err)
	assert.Greater(t, payloadTTL, time.Duration(0), "the checkpoint must carry the configured TTL")
	assert.LessOrEqual(t, payloadTTL, 90*time.Second)

	indexTTL, err := cp.client.TTL(ctx, redisThreadIndexKey(threadID)).Result()
	require.NoError(t, err)
	assert.Greater(t, indexTTL, time.Duration(0),
		"the thread index must expire too, or it leaks dead entries forever")
	assert.LessOrEqual(t, indexTTL, 90*time.Second)
}

// TestRedisCheckpointer_TTLIsConfigurable covers a value that was hard-coded at
// 24h with no way to change it, so every deployment silently lost its
// checkpoints after a day.
func TestRedisCheckpointer_TTLIsConfigurable(t *testing.T) {
	requireRedisConfig(t)

	t.Run("default", func(t *testing.T) {
		cp, _ := newRedisCheckpointer(t, nil)
		assert.Equal(t, 24*time.Hour, cp.ttl)
	})

	t.Run("configured", func(t *testing.T) {
		cp, prefix := newRedisCheckpointer(t, func(c *DatabaseConfig) { c.CheckpointTTL = "168h" })
		assert.Equal(t, 168*time.Hour, cp.ttl)

		ctx := context.Background()
		threadID := prefix + "week"
		require.NoError(t, cp.Save(ctx, newCheckpoint(threadID, "c", 1, core.NewBaseState())))
		ttl, err := cp.client.TTL(ctx, redisCheckpointKey(threadID, "c")).Result()
		require.NoError(t, err)
		assert.Greater(t, ttl, 167*time.Hour, "the configured TTL must reach Redis")
	})

	t.Run("zero disables expiry", func(t *testing.T) {
		cp, prefix := newRedisCheckpointer(t, func(c *DatabaseConfig) { c.CheckpointTTL = "0" })
		assert.Equal(t, time.Duration(0), cp.ttl)

		ctx := context.Background()
		threadID := prefix + "forever"
		require.NoError(t, cp.Save(ctx, newCheckpoint(threadID, "c", 1, core.NewBaseState())))
		ttl, err := cp.client.TTL(ctx, redisCheckpointKey(threadID, "c")).Result()
		require.NoError(t, err)
		// Redis reports -1 for a key with no expiry, which go-redis maps to -1ns.
		assert.Equal(t, time.Duration(-1), ttl, "TTL 0 must mean no expiry at all")
	})

	t.Run("invalid is rejected", func(t *testing.T) {
		cfg := requireRedisConfig(t)
		cfg.CheckpointTTL = "one week"
		_, err := NewRedisCheckpointer(cfg)
		require.Error(t, err, "an unparseable TTL must be reported, not silently ignored")
		assert.Contains(t, err.Error(), "checkpoint_ttl")
	})
}

// --- concurrency ----------------------------------------------------------

func TestRedisCheckpointer_ConcurrentAccess(t *testing.T) {
	cp, prefix := newRedisCheckpointer(t, nil)
	ctx := context.Background()

	const workers = 8
	const perWorker = 5

	var wg sync.WaitGroup
	errCh := make(chan error, workers*perWorker*2)

	for w := 0; w < workers; w++ {
		wg.Add(1)
		go func(w int) {
			defer wg.Done()
			threadID := fmt.Sprintf("%sconcurrent-%d", prefix, w)
			for i := 0; i < perWorker; i++ {
				st := core.NewBaseState()
				st.Set("worker", w)
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

	for w := 0; w < workers; w++ {
		list, err := cp.List(ctx, fmt.Sprintf("%sconcurrent-%d", prefix, w))
		require.NoError(t, err)
		assert.Len(t, list, perWorker, "thread %d lost or gained checkpoints", w)
	}
}

// --- context propagation --------------------------------------------------

func TestRedisCheckpointer_HonoursContextCancellation(t *testing.T) {
	cp, prefix := newRedisCheckpointer(t, nil)

	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	threadID := prefix + "ctx"
	err := cp.Save(ctx, newCheckpoint(threadID, "c", 1, core.NewBaseState()))
	require.Error(t, err)
	assert.ErrorIs(t, err, context.Canceled)

	_, err = cp.Load(ctx, threadID, "c")
	require.Error(t, err)
	assert.ErrorIs(t, err, context.Canceled)

	_, err = cp.List(ctx, threadID)
	require.Error(t, err)
	assert.ErrorIs(t, err, context.Canceled)

	err = cp.Delete(ctx, threadID, "c")
	require.Error(t, err)
	assert.ErrorIs(t, err, context.Canceled)
}

// --- integration with the graph-facing helpers ----------------------------

func TestRedisCheckpointer_SaverAndLatest(t *testing.T) {
	cp, prefix := newRedisCheckpointer(t, nil)
	ctx := context.Background()

	saver := NewCheckpointSaver(cp)
	threadID := prefix + "saver"

	for step := 1; step <= 4; step++ {
		st := core.NewBaseState()
		st.Set("step", step)
		st.Set("payload", fmt.Sprintf("after-node-%d", step))
		require.NoError(t, saver.SaveState(ctx, threadID, fmt.Sprintf("node-%d", step), step, st))
	}

	// List is backed by an unordered Redis set, so this also confirms Latest
	// does its own ordering rather than trusting the listing order.
	latest, err := Latest(ctx, cp, threadID)
	require.NoError(t, err)
	require.NotNil(t, latest)
	assert.EqualValues(t, 4, latest.State.GetAll()["step"], "Latest must return the highest step")
	assert.Equal(t, "after-node-4", latest.State.GetAll()["payload"])

	none, err := Latest(ctx, cp, prefix+"empty")
	require.NoError(t, err)
	assert.Nil(t, none)
}

// --- construction ---------------------------------------------------------

func TestNewRedisCheckpointer_UnreachableServerFails(t *testing.T) {
	// Port 1 is reserved and never listening, so this is deterministic and
	// needs no server at all -- it runs even where Redis is absent.
	cfg := NewRedisConfig("127.0.0.1", 1, "")
	cp, err := NewRedisCheckpointer(cfg)
	require.Error(t, err, "an unreachable server must fail construction")
	assert.Nil(t, cp)
	assert.Contains(t, err.Error(), "failed to connect to Redis")
}

func TestRedisCheckpointer_SaveNilCheckpointIsRejected(t *testing.T) {
	cp, _ := newRedisCheckpointer(t, nil)
	err := cp.Save(context.Background(), nil)
	require.Error(t, err, "a nil checkpoint must be rejected, not panic")
	assert.Contains(t, err.Error(), "nil checkpoint")
}
