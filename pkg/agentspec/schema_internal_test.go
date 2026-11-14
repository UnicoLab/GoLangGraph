// Copyright (c) 2024 GoLangGraph Team
//
// Licensed under the MIT License. See LICENSE file in the project root for full license information.
//
// Package: GoLangGraph - A powerful Go framework for building AI agent workflows

package agentspec

import (
	"reflect"
	"testing"
)

func TestSchemaRejectsDuplicateWireNames(t *testing.T) {
	// Construct the invalid type dynamically: a literal duplicate tag is itself
	// rejected by go vet before this negative contract test can run.
	typeWithDuplicates := reflect.StructOf([]reflect.StructField{
		{Name: "First", Type: reflect.TypeOf(""), Tag: `json:"text"`},
		{Name: "Second", Type: reflect.TypeOf(""), Tag: `json:"text"`},
	})
	if _, err := schemaFor(typeWithDuplicates, make(map[reflect.Type]bool)); err == nil {
		t.Fatal("duplicate JSON field accepted")
	}
}
