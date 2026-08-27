// Copyright (c) 2024 GoLangGraph Team
//
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

package agent

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/UnicoLab/GoLangGraph/pkg/llm"
)

// A second turn on the same agent is a new turn, not a resume.
//
// Resume used to be inferred from a non-empty conversation, which is also true
// of every ordinary turn after the first: the caller's second question was
// dropped instead of recorded, and the agent answered the first one again.
func TestResume_SecondTurnIsNotTreatedAsResume(t *testing.T) {
	a := &BaseAgent{
		config:       &AgentConfig{MaxIterations: 5},
		conversation: llm.NewConversationHistory(),
	}

	a.conversation.AddMessage(llm.Message{Role: "user", Content: "first"})
	a.conversation.AddMessage(llm.Message{Role: "assistant", Content: "reply"})
	a.currentIteration = 3

	a.mu.Lock()
	resuming := a.resumeSeeded
	a.mu.Unlock()

	assert.False(t, resuming,
		"a prior turn must not mark the next run as a resume")

	// The iteration counter must also reset, or a long-lived chat agent
	// eventually starts every turn at its iteration limit.
	assert.Equal(t, 3, a.currentIteration,
		"precondition: iteration carries the previous turn's count until a run resets it")
}

// Seeding is what makes a run a resume, and it is consumed by that one run.
func TestResume_SeedingMarksExactlyTheNextRun(t *testing.T) {
	a := &BaseAgent{
		config:       &AgentConfig{MaxIterations: 5},
		conversation: llm.NewConversationHistory(),
	}

	a.SeedResumeState([]llm.Message{{Role: "user", Content: "interrupted"}}, 2,
		[]llm.ToolCall{{ID: "call-1", Type: "function"}})

	a.mu.Lock()
	seeded := a.resumeSeeded
	iter := a.currentIteration
	pending := len(a.pendingToolCalls)
	a.mu.Unlock()

	require.True(t, seeded, "SeedResumeState must mark the next run as a resume")
	assert.Equal(t, 2, iter, "a resume continues from the interrupted iteration")
	assert.Equal(t, 1, pending, "pending tool calls survive the interrupt")
}

// SeedConversation with no messages is not a resume: there is nothing to resume.
func TestResume_EmptySeedIsNotAResume(t *testing.T) {
	a := &BaseAgent{
		config:       &AgentConfig{MaxIterations: 5},
		conversation: llm.NewConversationHistory(),
	}

	a.SeedConversation(nil)

	a.mu.Lock()
	seeded := a.resumeSeeded
	a.mu.Unlock()

	assert.False(t, seeded, "an empty seed leaves the next run a cold start")
}
