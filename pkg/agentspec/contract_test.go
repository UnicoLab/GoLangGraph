// Copyright (c) 2024 GoLangGraph Team
//
// Licensed under the MIT License. See LICENSE file in the project root for full license information.
//
// Package: GoLangGraph - A powerful Go framework for building AI agent workflows

package agentspec_test

import (
	"context"
	"encoding/json"
	"strings"
	"testing"

	"github.com/UnicoLab/GoLangGraph/pkg/agentspec"
)

type constrained struct {
	Text  string `json:"text" description:"A short greeting" minLength:"1" maxLength:"3"`
	Style string `json:"style" enum:"formal,casual"`
}

func constrainedPackage(t *testing.T) agentspec.Package {
	t.Helper()
	d, err := agentspec.NewHandler[constrained, constrained](agentspec.HandlerConfig{Name: "echo", Description: "Echo"}, func(_ context.Context, _ agentspec.Execution, i constrained) (constrained, error) { return i, nil })
	if err != nil {
		t.Fatal(err)
	}
	pkg, err := agentspec.NewPackage("example.com/echo", "v0.1.0", "Echo", d)
	if err != nil {
		t.Fatal(err)
	}
	return pkg
}

func TestSchemaConstraintsAndAmbiguousJSON(t *testing.T) {
	pkg := constrainedPackage(t)
	d := pkg.Definitions()[0]
	if err := d.ValidateInput(json.RawMessage(`{"text":"猫猫猫","style":"formal"}`)); err != nil {
		t.Fatalf("length must count Unicode characters: %v", err)
	}
	for _, raw := range []string{
		`{"text":"","style":"formal"}`, `{"text":"long","style":"formal"}`, `{"text":"hi","style":"other"}`,
		`{"text":"hi","text":"bad","style":"formal"}`, `{"text":"hi","\u0074ext":"bad","style":"formal"}`,
		`{"text":"hi","style":"formal"} {}`, `{"text":"hi","style":"formal","unknown":1}`,
		strings.Repeat("[", 65) + strings.Repeat("]", 65),
	} {
		if err := d.ValidateInput(json.RawMessage(raw)); err == nil {
			t.Fatalf("accepted %s", raw)
		}
	}
	manifest := d.Manifest()
	property := manifest.InputSchema.Properties["text"]
	if property.Description != "A short greeting" || *property.MaxLength != 3 {
		t.Fatalf("constraints not advertised: %+v", property)
	}
	*property.MaxLength = 500
	style := manifest.InputSchema.Properties["style"]
	style.Enum[0] = "other"
	if err := d.ValidateInput(json.RawMessage(`{"text":"long","style":"formal"}`)); err == nil {
		t.Fatal("manifest mutation changed validation")
	}
}

func TestManifestRoundTripAndRejectedCapabilities(t *testing.T) {
	pkg := constrainedPackage(t)
	data, err := json.Marshal(pkg.Manifest())
	if err != nil {
		t.Fatal(err)
	}
	manifest, err := agentspec.ParseManifest(data)
	if err != nil {
		t.Fatal(err)
	}
	if manifest.Agents[0].Kind != "handler" || manifest.Agents[0].Prompt != nil {
		t.Fatal("handler became prompt")
	}
	for _, mutate := range []func(*agentspec.PackageManifest){
		func(m *agentspec.PackageManifest) { m.ContractVersion = "unknown" },
		func(m *agentspec.PackageManifest) { m.Agents[0].Kind = "unknown" },
		func(m *agentspec.PackageManifest) { m.Agents[0].MaxModelCalls = 3 },
		func(m *agentspec.PackageManifest) { m.Agents[0].InputSchema.AdditionalProperties = nil },
		func(m *agentspec.PackageManifest) { m.Agents[0].InputSchema.Required = []string{"missing"} },
		func(m *agentspec.PackageManifest) { m.Agents = append(m.Agents, m.Agents[0]) },
	} {
		copy, err := agentspec.ParseManifest(data)
		if err != nil {
			t.Fatal(err)
		}
		mutate(&copy)
		if err := copy.Validate(); err == nil {
			t.Fatalf("invalid manifest accepted: %+v", copy)
		}
	}
	unknown := strings.Replace(string(data), `"kind":"handler"`, `"kind":"handler","executable":"unsafe"`, 1)
	if _, err := agentspec.ParseManifest([]byte(unknown)); err == nil {
		t.Fatal("unknown field accepted")
	}
}

func TestInvalidConstraintTagsFailDefinitionCreation(t *testing.T) {
	check := func(err error) {
		t.Helper()
		if err == nil {
			t.Fatal("invalid constraints accepted")
		}
	}
	_, err := agentspec.NewHandler[struct {
		N int `enum:"a,b"`
	}, struct{}](agentspec.HandlerConfig{Name: "bad", Description: "Bad"}, func(context.Context, agentspec.Execution, struct {
		N int `enum:"a,b"`
	}) (struct{}, error) {
		return struct{}{}, nil
	})
	check(err)
	_, err = agentspec.NewHandler[struct {
		N string `minLength:"4" maxLength:"2"`
	}, struct{}](agentspec.HandlerConfig{Name: "bad", Description: "Bad"}, func(context.Context, agentspec.Execution, struct {
		N string `minLength:"4" maxLength:"2"`
	}) (struct{}, error) {
		return struct{}{}, nil
	})
	check(err)
	_, err = agentspec.NewSubagent[struct{}, struct{}]("child", "Child", "example.com/child#entry", "v1.0.0-01")
	check(err)
}

type hiddenFields struct {
	Text string `json:"text"`
}
type encodedNumber int

func (encodedNumber) MarshalText() ([]byte, error) { return []byte("number"), nil }

func TestSchemaRejectsHiddenEmbeddingAndTextEncoding(t *testing.T) {
	_, err := agentspec.NewHandler[struct{ hiddenFields }, struct{}](agentspec.HandlerConfig{Name: "bad", Description: "Bad"}, func(context.Context, agentspec.Execution, struct{ hiddenFields }) (struct{}, error) {
		return struct{}{}, nil
	})
	if err == nil {
		t.Fatal("unexported embedding silently omitted exported JSON fields")
	}
	_, err = agentspec.NewHandler[struct{ N encodedNumber }, struct{}](agentspec.HandlerConfig{Name: "bad", Description: "Bad"}, func(context.Context, agentspec.Execution, struct{ N encodedNumber }) (struct{}, error) {
		return struct{}{}, nil
	})
	if err == nil {
		t.Fatal("text marshaler advertised a numeric JSON type")
	}
	_, err = agentspec.NewHandler[struct {
		N string `json:"bad\\name"`
	}, struct{}](agentspec.HandlerConfig{Name: "bad", Description: "Bad"}, func(context.Context, agentspec.Execution, struct {
		N string `json:"bad\\name"`
	}) (struct{}, error) {
		return struct{}{}, nil
	})
	if err == nil {
		t.Fatal("invalid JSON tag advertised a different wire name")
	}
}
