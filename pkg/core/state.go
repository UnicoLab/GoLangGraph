// Copyright (c) 2024 GoLangGraph Team
//
// Licensed under the MIT License. See LICENSE file in the project root for full license information.
//
// Package: GoLangGraph - A powerful Go framework for building AI agent workflows

package core

import (
	"encoding/json"
	"fmt"
	"reflect"
	"sort"
	"sync"
	"time"

	"github.com/google/uuid"
)

// StateValue represents any value that can be stored in state
type StateValue interface{}

// StateSnapshot represents a snapshot of the state at a specific point in time
type StateSnapshot struct {
	ID        string                 `json:"id"`
	Timestamp time.Time              `json:"timestamp"`
	Data      map[string]StateValue  `json:"data"`
	Metadata  map[string]interface{} `json:"metadata"`
}

// StateHistory manages the history of state changes
type StateHistory struct {
	snapshots []StateSnapshot
	maxSize   int
	mu        sync.RWMutex
}

// NewStateHistory creates a new state history with a maximum size
func NewStateHistory(maxSize int) *StateHistory {
	return &StateHistory{
		snapshots: make([]StateSnapshot, 0),
		maxSize:   maxSize,
	}
}

// AddSnapshot adds a new snapshot to the history
func (sh *StateHistory) AddSnapshot(snapshot StateSnapshot) {
	sh.mu.Lock()
	defer sh.mu.Unlock()

	sh.snapshots = append(sh.snapshots, snapshot)
	if len(sh.snapshots) > sh.maxSize {
		sh.snapshots = sh.snapshots[1:]
	}
}

// GetSnapshots returns all snapshots in the history
func (sh *StateHistory) GetSnapshots() []StateSnapshot {
	sh.mu.RLock()
	defer sh.mu.RUnlock()

	result := make([]StateSnapshot, len(sh.snapshots))
	copy(result, sh.snapshots)
	return result
}

// GetSnapshot returns a specific snapshot by ID
func (sh *StateHistory) GetSnapshot(id string) (*StateSnapshot, error) {
	sh.mu.RLock()
	defer sh.mu.RUnlock()

	for _, snapshot := range sh.snapshots {
		if snapshot.ID == id {
			return &snapshot, nil
		}
	}
	return nil, fmt.Errorf("snapshot with ID %s not found", id)
}

// BaseState represents the base state structure
type BaseState struct {
	data     map[string]StateValue
	metadata map[string]interface{}
	history  *StateHistory
	mu       sync.RWMutex
}

// NewBaseState creates a new base state
func NewBaseState() *BaseState {
	return &BaseState{
		data:     make(map[string]StateValue),
		metadata: make(map[string]interface{}),
		history:  NewStateHistory(100), // Keep last 100 snapshots
	}
}

// Get retrieves a value from the state
func (bs *BaseState) Get(key string) (StateValue, bool) {
	if bs == nil {
		return nil, false
	}
	bs.mu.RLock()
	defer bs.mu.RUnlock()

	value, exists := bs.data[key]
	return value, exists
}

// Set sets a value in the state
func (bs *BaseState) Set(key string, value StateValue) {
	bs.mu.Lock()
	defer bs.mu.Unlock()

	if bs.data == nil {
		bs.data = make(map[string]StateValue)
	}
	bs.data[key] = value
}

// Delete removes a key from the state
func (bs *BaseState) Delete(key string) {
	bs.mu.Lock()
	defer bs.mu.Unlock()

	delete(bs.data, key)
}

// Keys returns all keys in the state
func (bs *BaseState) Keys() []string {
	bs.mu.RLock()
	defer bs.mu.RUnlock()

	keys := make([]string, 0, len(bs.data))
	for k := range bs.data {
		keys = append(keys, k)
	}
	return keys
}

// GetAll returns a copy of all data in the state
func (bs *BaseState) GetAll() map[string]StateValue {
	if bs == nil {
		return map[string]StateValue{}
	}
	bs.mu.RLock()
	defer bs.mu.RUnlock()

	result := make(map[string]StateValue)
	for k, v := range bs.data {
		result[k] = deepCopy(v)
	}
	return result
}

// SetMetadata sets metadata for the state
func (bs *BaseState) SetMetadata(key string, value interface{}) {
	bs.mu.Lock()
	defer bs.mu.Unlock()

	if bs.metadata == nil {
		bs.metadata = make(map[string]interface{})
	}
	bs.metadata[key] = value
}

// GetMetadata retrieves metadata from the state
func (bs *BaseState) GetMetadata(key string) (interface{}, bool) {
	bs.mu.RLock()
	defer bs.mu.RUnlock()

	value, exists := bs.metadata[key]
	return value, exists
}

// CreateSnapshot creates a snapshot of the current state
func (bs *BaseState) CreateSnapshot() StateSnapshot {
	bs.mu.RLock()
	defer bs.mu.RUnlock()

	snapshot := StateSnapshot{
		ID:        uuid.New().String(),
		Timestamp: time.Now(),
		Data:      make(map[string]StateValue),
		Metadata:  make(map[string]interface{}),
	}

	// Deep copy data
	for k, v := range bs.data {
		snapshot.Data[k] = deepCopy(v)
	}

	// Deep copy metadata
	for k, v := range bs.metadata {
		snapshot.Metadata[k] = deepCopy(v)
	}

	// Add to history
	bs.history.AddSnapshot(snapshot)

	return snapshot
}

// RestoreFromSnapshot restores the state from a snapshot
func (bs *BaseState) RestoreFromSnapshot(snapshot StateSnapshot) {
	bs.mu.Lock()
	defer bs.mu.Unlock()

	// Clear current data
	bs.data = make(map[string]StateValue)
	bs.metadata = make(map[string]interface{})

	// Restore data
	for k, v := range snapshot.Data {
		bs.data[k] = deepCopy(v)
	}

	// Restore metadata
	for k, v := range snapshot.Metadata {
		bs.metadata[k] = deepCopy(v)
	}
}

// GetHistory returns the state history
func (bs *BaseState) GetHistory() *StateHistory {
	return bs.history
}

// Merge merges another state into this state using last-write-wins semantics
// for every key. Use MergeWithSchema to apply reducers.
func (bs *BaseState) Merge(other *BaseState) {
	if bs == nil || other == nil {
		return
	}

	otherData := other.GetAll()

	bs.mu.Lock()
	defer bs.mu.Unlock()
	if bs.data == nil {
		bs.data = make(map[string]StateValue)
	}
	for k, v := range otherData {
		bs.data[k] = v
	}
}

// MergeWithSchema merges another state into this one, applying the schema's
// reducer for each key. Keys without a reducer use last-write-wins, matching
// LangGraph's default channel behaviour.
func (bs *BaseState) MergeWithSchema(other *BaseState, schema *StateSchema) {
	if bs == nil || other == nil {
		return
	}
	if schema == nil {
		bs.Merge(other)
		return
	}

	otherData := other.GetAll()

	bs.mu.Lock()
	defer bs.mu.Unlock()
	if bs.data == nil {
		bs.data = make(map[string]StateValue)
	}
	for _, k := range sortedKeys(otherData) {
		v := otherData[k]
		if reducer := schema.Reducer(k); reducer != nil {
			existing, hadExisting := bs.data[k]
			if !hadExisting {
				existing = schema.Default(k)
			}
			bs.data[k] = reducer(existing, v)
			continue
		}
		bs.data[k] = v
	}
}

// Update applies a single key update through the schema reducer, if any.
func (bs *BaseState) Update(schema *StateSchema, key string, value StateValue) {
	if bs == nil {
		return
	}
	bs.mu.Lock()
	defer bs.mu.Unlock()
	if bs.data == nil {
		bs.data = make(map[string]StateValue)
	}
	if schema != nil {
		if reducer := schema.Reducer(key); reducer != nil {
			existing, hadExisting := bs.data[key]
			if !hadExisting {
				existing = schema.Default(key)
			}
			bs.data[key] = reducer(existing, value)
			return
		}
	}
	bs.data[key] = value
}

func sortedKeys(m map[string]StateValue) []string {
	keys := make([]string, 0, len(m))
	for k := range m {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	return keys
}

// Clone creates a deep copy of the state. Cloning a nil state yields a new
// empty state so that a node returning nil can never crash the engine.
func (bs *BaseState) Clone() *BaseState {
	if bs == nil {
		return NewBaseState()
	}

	bs.mu.RLock()
	defer bs.mu.RUnlock()

	clone := NewBaseState()

	// Deep copy data
	for k, v := range bs.data {
		clone.data[k] = deepCopy(v)
	}

	// Deep copy metadata
	for k, v := range bs.metadata {
		clone.metadata[k] = deepCopy(v)
	}

	return clone
}

// MarshalJSON implements json.Marshaler.
//
// BaseState keeps its data in unexported fields, so without this method
// encoding/json serialises it as "{}" and every persisted checkpoint, API
// response and WebSocket frame silently loses the entire state.
func (bs *BaseState) MarshalJSON() ([]byte, error) {
	if bs == nil {
		return []byte("null"), nil
	}
	bs.mu.RLock()
	defer bs.mu.RUnlock()

	data := bs.data
	if data == nil {
		data = map[string]StateValue{}
	}
	metadata := bs.metadata
	if metadata == nil {
		metadata = map[string]interface{}{}
	}

	return json.Marshal(statePayload{Data: data, Metadata: metadata})
}

// UnmarshalJSON implements json.Unmarshaler and accepts both the canonical
// {"data":...,"metadata":...} envelope and a bare object of state values, so
// older payloads and hand-written requests both load.
func (bs *BaseState) UnmarshalJSON(raw []byte) error {
	if bs == nil {
		return fmt.Errorf("cannot unmarshal into a nil BaseState")
	}

	var payload statePayload
	if err := json.Unmarshal(raw, &payload); err == nil && (payload.Data != nil || payload.Metadata != nil) {
		bs.mu.Lock()
		defer bs.mu.Unlock()
		bs.data = payload.Data
		bs.metadata = payload.Metadata
		if bs.data == nil {
			bs.data = make(map[string]StateValue)
		}
		if bs.metadata == nil {
			bs.metadata = make(map[string]interface{})
		}
		if bs.history == nil {
			bs.history = NewStateHistory(100)
		}
		return nil
	}

	// Fall back to a flat object of state values.
	var flat map[string]StateValue
	if err := json.Unmarshal(raw, &flat); err != nil {
		return err
	}
	bs.mu.Lock()
	defer bs.mu.Unlock()
	if flat == nil {
		flat = make(map[string]StateValue)
	}
	bs.data = flat
	bs.metadata = make(map[string]interface{})
	if bs.history == nil {
		bs.history = NewStateHistory(100)
	}
	return nil
}

// statePayload is the canonical wire format for a BaseState.
type statePayload struct {
	Data     map[string]StateValue  `json:"data"`
	Metadata map[string]interface{} `json:"metadata"`
}

// ToJSON converts the state to JSON
func (bs *BaseState) ToJSON() ([]byte, error) {
	bs.mu.RLock()
	defer bs.mu.RUnlock()

	return json.Marshal(statePayload{Data: bs.data, Metadata: bs.metadata})
}

// FromJSON loads the state from JSON
func (bs *BaseState) FromJSON(data []byte) error {
	bs.mu.Lock()
	defer bs.mu.Unlock()

	var stateData statePayload

	if err := json.Unmarshal(data, &stateData); err != nil {
		return err
	}

	bs.data = stateData.Data
	bs.metadata = stateData.Metadata
	if bs.data == nil {
		bs.data = make(map[string]StateValue)
	}
	if bs.metadata == nil {
		bs.metadata = make(map[string]interface{})
	}
	if bs.history == nil {
		bs.history = NewStateHistory(100)
	}

	return nil
}

// deepCopy creates a deep copy of a value.
//
// Values that cannot be meaningfully deep-copied (structs with unexported
// fields such as time.Time, channels, funcs) are returned as-is rather than
// panicking. Callers are expected to treat such values as immutable. Copying
// is depth-limited and cycle-aware so that self-referential data cannot cause
// unbounded recursion.
func deepCopy(src interface{}) interface{} {
	return deepCopyValue(src, make(map[uintptr]interface{}), 0)
}

const maxCopyDepth = 64

func deepCopyValue(src interface{}, seen map[uintptr]interface{}, depth int) interface{} {
	if src == nil {
		return nil
	}

	// Fast path for immutable scalars.
	switch v := src.(type) {
	case bool, int, int8, int16, int32, int64,
		uint, uint8, uint16, uint32, uint64,
		float32, float64, complex64, complex128, string:
		return v
	case time.Time:
		return v
	case []byte:
		dst := make([]byte, len(v))
		copy(dst, v)
		return dst
	}

	if depth >= maxCopyDepth {
		return src
	}

	srcVal := reflect.ValueOf(src)
	switch srcVal.Kind() {
	case reflect.Map:
		if srcVal.IsNil() {
			return src
		}
		if ptr := srcVal.Pointer(); ptr != 0 {
			if existing, ok := seen[ptr]; ok {
				return existing
			}
		}
		dst := reflect.MakeMapWithSize(srcVal.Type(), srcVal.Len())
		if ptr := srcVal.Pointer(); ptr != 0 {
			seen[ptr] = dst.Interface()
		}
		iter := srcVal.MapRange()
		for iter.Next() {
			copied := deepCopyValue(iter.Value().Interface(), seen, depth+1)
			dst.SetMapIndex(iter.Key(), reflectValueFor(copied, iter.Value().Type()))
		}
		return dst.Interface()

	case reflect.Slice:
		if srcVal.IsNil() {
			return src
		}
		dst := reflect.MakeSlice(srcVal.Type(), srcVal.Len(), srcVal.Len())
		for i := 0; i < srcVal.Len(); i++ {
			copied := deepCopyValue(srcVal.Index(i).Interface(), seen, depth+1)
			dst.Index(i).Set(reflectValueFor(copied, srcVal.Type().Elem()))
		}
		return dst.Interface()

	case reflect.Array:
		dst := reflect.New(srcVal.Type()).Elem()
		for i := 0; i < srcVal.Len(); i++ {
			copied := deepCopyValue(srcVal.Index(i).Interface(), seen, depth+1)
			dst.Index(i).Set(reflectValueFor(copied, srcVal.Type().Elem()))
		}
		return dst.Interface()

	case reflect.Ptr:
		if srcVal.IsNil() {
			return src
		}
		elemType := srcVal.Type().Elem()
		if !isCopyableStruct(elemType) {
			// Cannot safely copy: share the pointer.
			return src
		}
		if ptr := srcVal.Pointer(); ptr != 0 {
			if existing, ok := seen[ptr]; ok {
				return existing
			}
		}
		dst := reflect.New(elemType)
		if ptr := srcVal.Pointer(); ptr != 0 {
			seen[ptr] = dst.Interface()
		}
		copied := deepCopyValue(srcVal.Elem().Interface(), seen, depth+1)
		dst.Elem().Set(reflectValueFor(copied, elemType))
		return dst.Interface()

	case reflect.Struct:
		if !isCopyableStruct(srcVal.Type()) {
			// Structs with unexported fields cannot be rebuilt via reflection.
			// Returning the original preserves the value instead of panicking.
			return src
		}
		dst := reflect.New(srcVal.Type()).Elem()
		for i := 0; i < srcVal.NumField(); i++ {
			field := srcVal.Field(i)
			if !dst.Field(i).CanSet() {
				continue
			}
			copied := deepCopyValue(field.Interface(), seen, depth+1)
			dst.Field(i).Set(reflectValueFor(copied, field.Type()))
		}
		return dst.Interface()

	default:
		// Chan, Func, UnsafePointer and scalars of named types: share as-is.
		return src
	}
}

// reflectValueFor converts a copied value back into a reflect.Value assignable
// to the destination type, falling back to the zero value when the copy did not
// preserve assignability.
func reflectValueFor(v interface{}, dstType reflect.Type) reflect.Value {
	if v == nil {
		return reflect.Zero(dstType)
	}
	rv := reflect.ValueOf(v)
	if rv.Type().AssignableTo(dstType) {
		return rv
	}
	if rv.Type().ConvertibleTo(dstType) {
		return rv.Convert(dstType)
	}
	return reflect.Zero(dstType)
}

// isCopyableStruct reports whether every field of a struct type is exported, so
// that reflection can rebuild it field by field.
func isCopyableStruct(t reflect.Type) bool {
	if t.Kind() != reflect.Struct {
		return true
	}
	for i := 0; i < t.NumField(); i++ {
		if t.Field(i).PkgPath != "" { // unexported
			return false
		}
	}
	return true
}

// StateManager manages multiple states and provides advanced operations
type StateManager struct {
	states map[string]*BaseState
	mu     sync.RWMutex
}

// NewStateManager creates a new state manager
func NewStateManager() *StateManager {
	return &StateManager{
		states: make(map[string]*BaseState),
	}
}

// CreateState creates a new state with the given ID
func (sm *StateManager) CreateState(id string) *BaseState {
	sm.mu.Lock()
	defer sm.mu.Unlock()

	state := NewBaseState()
	sm.states[id] = state
	return state
}

// GetState retrieves a state by ID
func (sm *StateManager) GetState(id string) (*BaseState, bool) {
	sm.mu.RLock()
	defer sm.mu.RUnlock()

	state, exists := sm.states[id]
	return state, exists
}

// DeleteState removes a state by ID
func (sm *StateManager) DeleteState(id string) {
	sm.mu.Lock()
	defer sm.mu.Unlock()

	delete(sm.states, id)
}

// ListStates returns all state IDs
func (sm *StateManager) ListStates() []string {
	sm.mu.RLock()
	defer sm.mu.RUnlock()

	ids := make([]string, 0, len(sm.states))
	for id := range sm.states {
		ids = append(ids, id)
	}
	return ids
}
