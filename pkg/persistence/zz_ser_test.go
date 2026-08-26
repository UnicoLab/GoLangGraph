package persistence

import (
	"context"
	"encoding/json"
	"testing"

	"github.com/piotrlaczkowski/GoLangGraph/pkg/core"
)

func TestProve_CheckpointStateSerialization(t *testing.T) {
	st := core.NewBaseState()
	st.Set("messages", []interface{}{"hello"})
	st.Set("counter", 42)

	cp := &Checkpoint{ID: "c1", ThreadID: "t1", State: st, NodeID: "n1", StepID: 0}
	data, err := json.Marshal(cp)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	t.Logf("SERIALIZED: %s", string(data))

	var back Checkpoint
	if err := json.Unmarshal(data, &back); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if back.State == nil {
		t.Fatal("state nil after round-trip")
	}
	v, ok := back.State.Get("counter")
	if !ok {
		t.Fatalf("DATA LOSS: counter missing after JSON round-trip; payload=%s", string(data))
	}
	t.Logf("counter=%v", v)
}

func TestProve_FileCheckpointerRoundTrip(t *testing.T) {
	dir := t.TempDir()
	c := NewFileCheckpointer(dir)
	st := core.NewBaseState()
	st.Set("counter", 42)
	cp := &Checkpoint{ID: "c1", ThreadID: "t1", State: st}
	if err := c.Save(context.Background(), cp); err != nil {
		t.Fatalf("save: %v", err)
	}
	got, err := c.Load(context.Background(), "t1", "c1")
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	if _, ok := got.State.Get("counter"); !ok {
		t.Fatal("DATA LOSS: FileCheckpointer lost state across save/load")
	}
}
