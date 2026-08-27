// Copyright (c) 2024 GoLangGraph Team
//
// Licensed under the MIT License. See LICENSE file in the project root for full license information.
//
// Package: GoLangGraph - A powerful Go framework for building AI agent workflows

// Copyright (c) 2024 GoLangGraph Team
//
// Licensed under the MIT License. See LICENSE file in the project root for full license information.
//
// Package backends provides filesystem backend implementations

package backends

import (
	"context"
	"encoding/json"
	"fmt"
	"path/filepath"
	"sort"
	"strings"
	"time"

	"github.com/UnicoLab/GoLangGraph/pkg/persistence"
)

// StoreBackend persists files across sessions using persistence.Store
// Files are stored in the persistent store and survive beyond thread lifetime
// This matches LangGraph's StoreBackend for long-term memory
type StoreBackend struct {
	store     persistence.Store
	namespace string // Namespace for isolating file storage (e.g., user ID)
}

// NewStoreBackend creates a new store-based backend
func NewStoreBackend(store persistence.Store, namespace string) *StoreBackend {
	if namespace == "" {
		namespace = "default"
	}
	return &StoreBackend{
		store:     store,
		namespace: namespace,
	}
}

// Read reads file content from persistent store
func (b *StoreBackend) Read(ctx context.Context, path string, offset, limit int) (string, error) {
	if offset < 0 {
		return "", fmt.Errorf("offset must not be negative")
	}
	fileData, err := b.getFileData(ctx, path)
	if err != nil {
		return "", err
	}

	lines := fileData.Content
	if len(lines) == 0 {
		return "System reminder: File exists but has empty contents\n", nil
	}

	// Apply pagination
	if offset >= len(lines) {
		return "", fmt.Errorf("offset %d is out of bounds (file has %d lines)", offset, len(lines))
	}

	end := len(lines)
	if limit > 0 {
		end = offset + limit
		if end > len(lines) {
			end = len(lines)
		}
	}

	// Format with line numbers
	var sb strings.Builder
	for i := offset; i < end; i++ {
		line := lines[i]
		if len(line) > 2000 {
			line = line[:2000] + "... (truncated)"
		}
		sb.WriteString(fmt.Sprintf("%6d\t%s\n", i+1, line))
	}

	return sb.String(), nil
}

// Write creates or overwrites a file in persistent store
func (b *StoreBackend) Write(ctx context.Context, path string, content string) (*WriteResult, error) {
	now := time.Now().Format(time.RFC3339)

	fileData := &FileData{
		Content:    strings.Split(content, "\n"),
		CreatedAt:  now,
		ModifiedAt: now,
	}

	if err := b.saveFileData(ctx, path, fileData); err != nil {
		return &WriteResult{Error: err.Error()}, nil
	}

	return &WriteResult{
		Path:        path,
		FilesUpdate: nil, // Store backend doesn't update state
		Error:       "",
	}, nil
}

// Edit performs string replacement in a file
func (b *StoreBackend) Edit(ctx context.Context, path string, oldStr, newStr string, replaceAll bool) (*EditResult, error) {
	fileData, err := b.getFileData(ctx, path)
	if err != nil {
		return &EditResult{Error: err.Error()}, nil
	}

	content := strings.Join(fileData.Content, "\n")

	if !strings.Contains(content, oldStr) {
		return &EditResult{Error: "old_string not found in file"}, nil
	}

	occurrences := strings.Count(content, oldStr)

	if !replaceAll && occurrences > 1 {
		return &EditResult{
			Error: fmt.Sprintf("old_string found %d times, use replace_all=true or provide more context", occurrences),
		}, nil
	}

	var newContent string
	if replaceAll {
		newContent = strings.ReplaceAll(content, oldStr, newStr)
	} else {
		newContent = strings.Replace(content, oldStr, newStr, 1)
		occurrences = 1
	}

	fileData.Content = strings.Split(newContent, "\n")
	fileData.ModifiedAt = time.Now().Format(time.RFC3339)

	if err := b.saveFileData(ctx, path, fileData); err != nil {
		return &EditResult{Error: err.Error()}, nil
	}

	return &EditResult{
		Path:        path,
		Occurrences: occurrences,
		FilesUpdate: nil,
		Error:       "",
	}, nil
}

// List lists files in a directory from persistent store
func (b *StoreBackend) List(ctx context.Context, path string) ([]FileInfo, error) {
	paths, err := b.filePaths(ctx)
	if err != nil {
		return nil, err
	}

	if path != "/" && !strings.HasSuffix(path, "/") {
		path += "/"
	}
	infos := make([]FileInfo, 0, len(paths))
	for _, filePath := range paths {
		if path != "/" && !strings.HasPrefix(filePath, path) {
			continue
		}
		relPath := strings.TrimPrefix(filePath, path)
		if path != "/" && strings.Contains(relPath, "/") {
			continue
		}
		fileData, err := b.getFileData(ctx, filePath)
		if err != nil {
			return nil, err
		}
		modifiedAt, err := time.Parse(time.RFC3339, fileData.ModifiedAt)
		if err != nil {
			return nil, fmt.Errorf("parse modified time for file %q: %w", filePath, err)
		}
		infos = append(infos, FileInfo{
			Path:       filePath,
			IsDir:      false,
			Size:       int64(len(strings.Join(fileData.Content, "\n"))),
			ModifiedAt: modifiedAt,
		})
	}
	sort.Slice(infos, func(i, j int) bool { return infos[i].Path < infos[j].Path })
	return infos, nil
}

// Glob finds files matching a pattern
func (b *StoreBackend) Glob(ctx context.Context, pattern string, basePath string) ([]string, error) {
	if _, err := filepath.Match(pattern, ""); err != nil {
		return nil, fmt.Errorf("invalid glob pattern %q: %w", pattern, err)
	}
	paths, err := b.filePaths(ctx)
	if err != nil {
		return nil, err
	}
	matches := make([]string, 0)
	for _, filePath := range paths {
		if basePath != "" && !pathWithinBase(filePath, basePath) {
			continue
		}
		matched, err := filepath.Match(pattern, filepath.Base(filePath))
		if err != nil {
			return nil, err
		}
		if matched {
			matches = append(matches, filePath)
		}
	}
	sort.Strings(matches)
	return matches, nil
}

// Grep searches for text in files
func (b *StoreBackend) Grep(ctx context.Context, pattern string, path string, globFilter string) ([]GrepMatch, error) {
	if globFilter != "" {
		if _, err := filepath.Match(globFilter, ""); err != nil {
			return nil, fmt.Errorf("invalid glob filter %q: %w", globFilter, err)
		}
	}
	paths, err := b.filePaths(ctx)
	if err != nil {
		return nil, err
	}
	matches := make([]GrepMatch, 0)
	for _, filePath := range paths {
		if path != "" && !pathWithinBase(filePath, path) {
			continue
		}
		if globFilter != "" {
			matched, err := filepath.Match(globFilter, filepath.Base(filePath))
			if err != nil {
				return nil, err
			}
			if !matched {
				continue
			}
		}
		fileData, err := b.getFileData(ctx, filePath)
		if err != nil {
			return nil, err
		}
		for i, line := range fileData.Content {
			if strings.Contains(line, pattern) {
				matches = append(matches, GrepMatch{Path: filePath, LineNumber: i + 1, Line: line})
			}
		}
	}
	sort.Slice(matches, func(i, j int) bool {
		if matches[i].Path == matches[j].Path {
			return matches[i].LineNumber < matches[j].LineNumber
		}
		return matches[i].Path < matches[j].Path
	})
	return matches, nil
}

// getKey creates a namespaced key for the file
func (b *StoreBackend) getKey(path string) string {
	return fmt.Sprintf("files:%s:%s", b.namespace, path)
}

// getFileData retrieves file data from store
func (b *StoreBackend) getFileData(ctx context.Context, path string) (*FileData, error) {
	if b.store == nil {
		return nil, fmt.Errorf("persistent store is not configured")
	}
	key := b.getKey(path)

	data, err := b.store.Get(ctx, key)
	if err != nil {
		return nil, fmt.Errorf("file '%s' not found", path)
	}

	var fileData FileData
	if err := json.Unmarshal([]byte(data), &fileData); err != nil {
		return nil, fmt.Errorf("failed to parse file data: %w", err)
	}

	return &fileData, nil
}

// saveFileData saves file data to store
func (b *StoreBackend) saveFileData(ctx context.Context, path string, fileData *FileData) error {
	if b.store == nil {
		return fmt.Errorf("persistent store is not configured")
	}
	key := b.getKey(path)

	data, err := json.Marshal(fileData)
	if err != nil {
		return fmt.Errorf("failed to serialize file data: %w", err)
	}

	return b.store.Set(ctx, key, string(data))
}

func (b *StoreBackend) filePaths(ctx context.Context) ([]string, error) {
	if b.store == nil {
		return nil, fmt.Errorf("persistent store is not configured")
	}
	prefix := fmt.Sprintf("files:%s:", b.namespace)
	keys, err := b.store.List(ctx, prefix)
	if err != nil {
		return nil, fmt.Errorf("list persistent files: %w", err)
	}
	paths := make([]string, 0, len(keys))
	for _, key := range keys {
		if strings.HasPrefix(key, prefix) {
			paths = append(paths, strings.TrimPrefix(key, prefix))
		}
	}
	sort.Strings(paths)
	return paths, nil
}

func pathWithinBase(path, base string) bool {
	if base == "" || base == "/" {
		return true
	}
	base = strings.TrimSuffix(base, "/")
	return path == base || strings.HasPrefix(path, base+"/")
}
