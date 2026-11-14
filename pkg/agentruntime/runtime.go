// Copyright (c) 2024 GoLangGraph Team
//
// Licensed under the MIT License. See LICENSE file in the project root for full license information.
//
// Package: GoLangGraph - A powerful Go framework for building AI agent workflows

// Package agentruntime executes agentspec packages using GoLangGraph. It is an
// opt-in adapter; the existing agent API is unchanged. Each request has a durable
// journal entry; successful turns commit their complete thread state. Interrupted
// requests are never replayed automatically. Locks apply within one Runtime.
package agentruntime

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"reflect"
	"sort"
	"strings"
	"sync"
	"time"

	"github.com/google/uuid"

	"github.com/UnicoLab/GoLangGraph/pkg/agentspec"
	"github.com/UnicoLab/GoLangGraph/pkg/core"
	"github.com/UnicoLab/GoLangGraph/pkg/llm"
	"github.com/UnicoLab/GoLangGraph/pkg/persistence"
)

// Some backends (including PostgreSQL) key checkpoints globally by ID, so the
// identifier must be unique across threads as well as stable across turns.
func checkpointID(threadID string) string { return "agent-package-" + threadID }

var ErrModelCallLimit = errors.New("agent model call limit exceeded")

// Completer is satisfied by llm.ProviderManager. Keeping this port narrow also
// permits an offline model double without bypassing tools, graphs or storage.
type Completer interface {
	Complete(context.Context, string, llm.CompletionRequest) (*llm.CompletionResponse, error)
}

type Config struct {
	Providers    Completer
	Checkpointer persistence.Checkpointer
	// Models overrides package defaults by fully qualified agent ID. The chosen
	// profile is persisted when a thread is created and remains fixed for it.
	Models map[string]agentspec.Model
	// MaxRecordedRuns bounds each thread's journal. No idempotency keys are
	// silently evicted. Create a new thread when this limit is reached.
	MaxRecordedRuns int
}

type entry struct {
	definition agentspec.Definition
	version    string
}

type threadLock struct {
	token chan struct{}
	refs  int
}

type Runtime struct {
	providers       Completer
	checkpointer    persistence.Checkpointer
	models          map[string]agentspec.Model
	mu              sync.RWMutex
	packages        map[string]agentspec.Package
	agents          map[string]entry
	locksMu         sync.Mutex
	locks           map[string]*threadLock
	maxRecordedRuns int
}

func New(config Config) (*Runtime, error) {
	if config.MaxRecordedRuns == 0 {
		config.MaxRecordedRuns = 1000
	}
	if config.MaxRecordedRuns < 1 {
		return nil, fmt.Errorf("max recorded runs must be positive")
	}
	if config.Checkpointer == nil {
		config.Checkpointer = persistence.NewMemoryCheckpointer()
	}
	models := make(map[string]agentspec.Model, len(config.Models))
	for id, model := range config.Models {
		if err := model.Validate(); err != nil {
			return nil, fmt.Errorf("model override %s: %w", id, err)
		}
		models[id] = model
	}
	return &Runtime{
		providers: config.Providers, checkpointer: config.Checkpointer, models: models,
		packages: make(map[string]agentspec.Package), agents: make(map[string]entry),
		locks:           make(map[string]*threadLock),
		maxRecordedRuns: config.MaxRecordedRuns,
	}, nil
}

// Register installs already-linked Go packages. A batch is all-or-nothing;
// versions cannot be replaced underneath existing threads.
func (r *Runtime) Register(packages ...agentspec.Package) error {
	r.mu.Lock()
	defer r.mu.Unlock()
	pending := make(map[string]bool)
	entries := make(map[string]entry, len(r.agents))
	for id, registered := range r.agents {
		entries[id] = registered
	}
	for _, pkg := range packages {
		manifest := pkg.Manifest()
		if manifest.Module == "" || len(manifest.Agents) == 0 || manifest.ContractVersion != agentspec.ContractVersion {
			return fmt.Errorf("uninitialized or unsupported agent package")
		}
		if _, exists := r.packages[manifest.Module]; exists || pending[manifest.Module] {
			return fmt.Errorf("package %s is already registered", manifest.Module)
		}
		pending[manifest.Module] = true
		for _, definition := range pkg.Definitions() {
			if definition.Manifest().Kind == "prompt" && r.providers == nil {
				return fmt.Errorf("agent %s requires a model provider", definition.Manifest().Name)
			}
			entries[pkg.AgentID(definition.Manifest().Name)] = entry{definition, manifest.Version}
		}
	}
	if err := validateDependencies(entries); err != nil {
		return err
	}
	for _, pkg := range packages {
		manifest := pkg.Manifest()
		r.packages[manifest.Module] = pkg
		for _, definition := range pkg.Definitions() {
			r.agents[pkg.AgentID(definition.Manifest().Name)] = entry{definition, manifest.Version}
		}
	}
	return nil
}

func validateDependencies(entries map[string]entry) error {
	visiting, visited := make(map[string]bool), make(map[string]bool)
	var visit func(string) error
	visit = func(id string) error {
		if visiting[id] {
			return fmt.Errorf("cyclic child-agent dependency at %s", id)
		}
		if visited[id] {
			return nil
		}
		visiting[id] = true
		for _, child := range entries[id].definition.Manifest().Subagents {
			target, ok := entries[child.AgentID]
			if !ok || target.version != child.PackageVersion {
				return fmt.Errorf("%s requires child %s at %s", id, child.AgentID, child.PackageVersion)
			}
			manifest := target.definition.Manifest()
			if !reflect.DeepEqual(child.InputSchema, manifest.InputSchema) || !reflect.DeepEqual(child.OutputSchema, manifest.OutputSchema) {
				return fmt.Errorf("%s child %s has incompatible input/output schemas", id, child.Name)
			}
			if err := visit(child.AgentID); err != nil {
				return err
			}
		}
		delete(visiting, id)
		visited[id] = true
		return nil
	}
	for id := range entries {
		if err := visit(id); err != nil {
			return err
		}
	}
	return nil
}

// Catalog returns independent, JSON-serializable package manifests for tooling.
func (r *Runtime) Catalog() []agentspec.PackageManifest {
	r.mu.RLock()
	defer r.mu.RUnlock()
	result := make([]agentspec.PackageManifest, 0, len(r.packages))
	for _, pkg := range r.packages {
		result = append(result, pkg.Manifest())
	}
	sort.Slice(result, func(i, j int) bool { return result[i].Module < result[j].Module })
	return result
}

// Thread is a durable conversation instance, distinct from its reusable agent
// definition. Histories and tool values never live on shared definitions.
type Thread struct {
	ID                    string                     `json:"id"`
	AgentID               string                     `json:"agent_id"`
	PackageVersion        string                     `json:"package_version"`
	ContractVersion       string                     `json:"contract_version"`
	DefinitionFingerprint string                     `json:"definition_fingerprint"`
	Kind                  string                     `json:"kind"`
	ParentID              string                     `json:"parent_id,omitempty"`
	ChildName             string                     `json:"child_name,omitempty"`
	Children              map[string]string          `json:"children"`
	Runs                  map[string]RunRecord       `json:"runs"`
	Model                 agentspec.Model            `json:"model"`
	Messages              []llm.Message              `json:"messages"`
	Values                map[string]json.RawMessage `json:"values"`
	CompletedRuns         int                        `json:"completed_runs"`
	CreatedAt             time.Time                  `json:"created_at"`
	UpdatedAt             time.Time                  `json:"updated_at"`
}

func (r *Runtime) CreateThread(ctx context.Context, agentID string) (*Thread, error) {
	// Materialize the dependency tree before writing any records. A bounded,
	// acyclic tree gives each parent/alias its own stable child identity.
	var created []*Thread
	thread, err := r.buildThread(ctx, agentID, "", "", &created)
	if err != nil {
		return nil, err
	}
	for _, item := range created {
		if err := r.save(ctx, item); err != nil {
			cleanup, cancel := context.WithTimeout(context.WithoutCancel(ctx), 5*time.Second)
			defer cancel()
			for _, saved := range created {
				_ = r.checkpointer.Delete(cleanup, saved.ID, checkpointID(saved.ID))
			}
			return nil, err
		}
	}
	return thread, nil
}

func (r *Runtime) buildThread(ctx context.Context, agentID, parentID, childName string, created *[]*Thread) (*Thread, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	r.mu.RLock()
	registered, exists := r.agents[agentID]
	r.mu.RUnlock()
	if !exists {
		return nil, fmt.Errorf("agent %s is not registered", agentID)
	}
	if len(*created) >= 128 {
		return nil, fmt.Errorf("agent dependency tree exceeds 128 threads")
	}
	manifest := registered.definition.Manifest()
	var model agentspec.Model
	if manifest.Prompt != nil {
		model = manifest.Prompt.Model
		if override, ok := r.models[agentID]; ok {
			model = override
		}
	}
	now := time.Now().UTC()
	thread := &Thread{
		ID: uuid.NewString(), AgentID: agentID, PackageVersion: registered.version,
		ContractVersion: agentspec.ContractVersion, Model: model,
		DefinitionFingerprint: registered.definition.Fingerprint(), Kind: manifest.Kind,
		ParentID: parentID, ChildName: childName, Children: make(map[string]string), Runs: make(map[string]RunRecord),
		Messages: []llm.Message{}, Values: make(map[string]json.RawMessage),
		CreatedAt: now, UpdatedAt: now,
	}
	*created = append(*created, thread)
	for _, child := range manifest.Subagents {
		instance, err := r.buildThread(ctx, child.AgentID, thread.ID, child.Name, created)
		if err != nil {
			return nil, err
		}
		thread.Children[child.Name] = instance.ID
	}
	return thread, nil
}

// ReadThread reloads a snapshot from storage. File/database checkpoint backends
// allow a new Runtime with the same package versions to continue the thread.
func (r *Runtime) ReadThread(ctx context.Context, id string) (*Thread, error) {
	release, err := r.acquire(ctx, id)
	if err != nil {
		return nil, err
	}
	defer release()
	return r.load(ctx, id)
}

type Result struct {
	ThreadID   string          `json:"thread_id"`
	RequestID  string          `json:"request_id"`
	Output     json.RawMessage `json:"output"`
	ModelCalls int             `json:"model_calls"`
	ToolCalls  int             `json:"tool_calls"`
	ChildCalls int             `json:"child_calls"`
	Usage      llm.Usage       `json:"usage"`
}

// Run starts a new request. Use RunWithOptions with a stable RequestID when a
// caller may redeliver a request after a timeout or restart.
func (r *Runtime) Run(ctx context.Context, threadID string, input any) (*Result, error) {
	return r.RunWithOptions(ctx, threadID, input, RunOptions{})
}

// RunTyped adds compile-time input/output types at an application's call site.
// The registered definition remains authoritative for runtime validation.
func RunTyped[I, O any](ctx context.Context, runtime *Runtime, threadID string, input I) (O, error) {
	return RunTypedWithOptions[I, O](ctx, runtime, threadID, input, RunOptions{})
}

func RunTypedWithOptions[I, O any](ctx context.Context, runtime *Runtime, threadID string, input I, options RunOptions) (O, error) {
	var output O
	thread, err := runtime.ReadThread(ctx, threadID)
	if err != nil {
		return output, err
	}
	runtime.mu.RLock()
	registered, exists := runtime.agents[thread.AgentID]
	runtime.mu.RUnlock()
	if !exists {
		return output, fmt.Errorf("agent %s is not registered", thread.AgentID)
	}
	if typeErr := agentspec.CheckTypes[I, O](registered.definition); typeErr != nil {
		return output, typeErr
	}
	result, err := runtime.RunWithOptions(ctx, threadID, input, options)
	if err != nil {
		return output, err
	}
	if err := json.Unmarshal(result.Output, &output); err != nil {
		return output, fmt.Errorf("decode result: %w", err)
	}
	return output, nil
}

func (r *Runtime) load(ctx context.Context, id string) (*Thread, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	checkpoint, err := r.checkpointer.Load(ctx, id, checkpointID(id))
	if err != nil {
		return nil, fmt.Errorf("load thread %s: %w", id, err)
	}
	if checkpoint == nil || checkpoint.State == nil {
		return nil, fmt.Errorf("thread %s has no checkpoint state", id)
	}
	value, exists := checkpoint.State.Get("thread")
	if !exists {
		return nil, fmt.Errorf("thread %s has no thread record", id)
	}
	data, ok := value.(string)
	if !ok {
		return nil, fmt.Errorf("invalid thread record encoding")
	}
	var thread Thread
	if err := json.Unmarshal([]byte(data), &thread); err != nil {
		return nil, fmt.Errorf("decode stored thread: %w", err)
	}
	if thread.ID != id || thread.ContractVersion != agentspec.ContractVersion || thread.Values == nil || thread.Runs == nil || thread.Children == nil || thread.DefinitionFingerprint == "" {
		return nil, fmt.Errorf("invalid or unsupported stored thread %s", id)
	}
	if thread.Kind == "prompt" {
		if err := thread.Model.Validate(); err != nil {
			return nil, fmt.Errorf("stored thread model: %w", err)
		}
	} else if thread.Kind != "handler" {
		return nil, fmt.Errorf("unsupported thread kind %q", thread.Kind)
	}
	return &thread, nil
}

func (r *Runtime) save(ctx context.Context, thread *Thread) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	// Keep JSON in a string envelope so a checkpointer's generic map decoding
	// cannot turn large integer tool values into lossy float64 values.
	data, err := json.Marshal(thread)
	if err != nil {
		return err
	}
	state := core.NewBaseState()
	state.Set("thread", string(data))
	if err := r.checkpointer.Save(ctx, &persistence.Checkpoint{
		ID: checkpointID(thread.ID), ThreadID: thread.ID, State: state,
		CreatedAt: thread.UpdatedAt, NodeID: "agent_thread", StepID: thread.CompletedRuns,
	}); err != nil {
		return fmt.Errorf("persist thread %s: %w", thread.ID, err)
	}
	return nil
}

func (r *Runtime) acquire(ctx context.Context, id string) (func(), error) {
	if strings.TrimSpace(id) == "" {
		return nil, fmt.Errorf("thread ID is required")
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	r.locksMu.Lock()
	lock := r.locks[id]
	if lock == nil {
		lock = &threadLock{token: make(chan struct{}, 1)}
		r.locks[id] = lock
	}
	lock.refs++
	r.locksMu.Unlock()
	drop := func() {
		r.locksMu.Lock()
		defer r.locksMu.Unlock()
		lock.refs--
		if lock.refs == 0 {
			delete(r.locks, id)
		}
	}
	select {
	case lock.token <- struct{}{}:
		return func() { <-lock.token; drop() }, nil
	case <-ctx.Done():
		drop()
		return nil, ctx.Err()
	}
}
