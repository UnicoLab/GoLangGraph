// Copyright (c) 2024 GoLangGraph Team
//
// Licensed under the MIT License. See LICENSE file in the project root for full license information.
//
// Package: GoLangGraph - A powerful Go framework for building AI agent workflows

package persistence

import (
	"context"
	"fmt"
	"sync/atomic"
	"time"

	"github.com/google/uuid"
	"github.com/piotrlaczkowski/GoLangGraph/pkg/core"
)

// CheckpointSaver adapts a Checkpointer to core.StateSaver so a graph can
// persist its state after every node execution. This is what makes execution
// durable: a run that dies mid-graph can be resumed from its last checkpoint.
//
// core deliberately does not import this package; the adapter lives here so the
// engine stays free of storage dependencies.
type CheckpointSaver struct {
	checkpointer Checkpointer
	seq          atomic.Uint64
	// IDFunc generates checkpoint IDs. Overridable for deterministic tests.
	IDFunc func(threadID, nodeID string, step int) string
	// Now supplies timestamps. Overridable for deterministic tests.
	Now func() time.Time
}

// NewCheckpointSaver wraps a Checkpointer for use with core.Graph.
func NewCheckpointSaver(cp Checkpointer) *CheckpointSaver {
	return &CheckpointSaver{checkpointer: cp}
}

// SaveState implements core.StateSaver.
func (s *CheckpointSaver) SaveState(ctx context.Context, threadID, nodeID string, step int, state *core.BaseState) error {
	if s == nil || s.checkpointer == nil {
		return nil
	}
	if state == nil {
		return fmt.Errorf("cannot checkpoint a nil state")
	}

	now := time.Now
	if s.Now != nil {
		now = s.Now
	}

	id := s.nextID(threadID, nodeID, step)

	return s.checkpointer.Save(ctx, &Checkpoint{
		ID:        id,
		ThreadID:  threadID,
		State:     state.Clone(),
		NodeID:    nodeID,
		StepID:    step,
		CreatedAt: now(),
		Metadata: map[string]interface{}{
			"node_id": nodeID,
			"step":    step,
		},
	})
}

func (s *CheckpointSaver) nextID(threadID, nodeID string, step int) string {
	if s.IDFunc != nil {
		return s.IDFunc(threadID, nodeID, step)
	}
	// Monotonic prefix keeps checkpoint IDs sortable by creation order, and the
	// UUID suffix keeps them unique across processes sharing a thread.
	return fmt.Sprintf("%010d-%s", s.seq.Add(1), uuid.New().String())
}

// Latest returns the most recent checkpoint for a thread, or nil when the
// thread has none. Checkpoints are ordered by step then creation time.
func Latest(ctx context.Context, cp Checkpointer, threadID string) (*Checkpoint, error) {
	metas, err := cp.List(ctx, threadID)
	if err != nil {
		return nil, err
	}
	if len(metas) == 0 {
		return nil, nil
	}
	best := metas[0]
	for _, m := range metas[1:] {
		if m.StepID > best.StepID || (m.StepID == best.StepID && m.CreatedAt.After(best.CreatedAt)) {
			best = m
		}
	}
	return cp.Load(ctx, threadID, best.ID)
}
