// Copyright (c) 2024 GoLangGraph Team
//
// Licensed under the MIT License. See LICENSE file in the project root for full license information.
//
// Package: GoLangGraph - A powerful Go framework for building AI agent workflows

package server

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// The schema the auto server advertises for an agent's input.
func inputSchema() map[string]interface{} {
	return map[string]interface{}{
		"type": "object",
		"properties": map[string]interface{}{
			"message": map[string]interface{}{
				"type":      "string",
				"minLength": 1,
				"maxLength": 10,
			},
			"count": map[string]interface{}{
				"type":    "integer",
				"minimum": 0,
				"maximum": 5,
			},
			"mode": map[string]interface{}{
				"type": "string",
				"enum": []interface{}{"fast", "slow"},
			},
			"history": map[string]interface{}{
				"type": "array",
				"items": map[string]interface{}{
					"type": "object",
					"properties": map[string]interface{}{
						"role": map[string]interface{}{"type": "string"},
					},
					"required": []string{"role"},
				},
			},
		},
		"required": []string{"message"},
	}
}

// A conforming payload validates. The endpoint previously said this for every
// payload, so the interesting cases are the ones below it.
func TestSchemaValidation_AcceptsValidInput(t *testing.T) {
	errs := validateAgainstSchema(inputSchema(), map[string]interface{}{
		"message": "hello",
		"count":   float64(3),
		"mode":    "fast",
		"history": []interface{}{map[string]interface{}{"role": "user"}},
	})
	assert.Empty(t, errs, "a conforming payload must validate: %v", errs)
}

func TestSchemaValidation_RejectsMissingRequiredField(t *testing.T) {
	errs := validateAgainstSchema(inputSchema(), map[string]interface{}{"count": float64(1)})
	require.NotEmpty(t, errs)
	assert.Contains(t, errs[0], "message")
	assert.Contains(t, errs[0], "required")
}

func TestSchemaValidation_RejectsWrongTypes(t *testing.T) {
	cases := []struct {
		name    string
		payload map[string]interface{}
		want    string
	}{
		{"string field given a number", map[string]interface{}{"message": float64(5)}, "must be a string"},
		{"integer field given a string", map[string]interface{}{"message": "ok", "count": "many"}, "must be a number"},
		{"integer field given a fraction", map[string]interface{}{"message": "ok", "count": 1.5}, "must be an integer"},
		{"array field given an object", map[string]interface{}{"message": "ok", "history": map[string]interface{}{}}, "must be an array"},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			errs := validateAgainstSchema(inputSchema(), tc.payload)
			require.NotEmpty(t, errs)
			assert.Contains(t, summariseErrors(errs), tc.want)
		})
	}
}

func TestSchemaValidation_EnforcesBounds(t *testing.T) {
	tooShort := validateAgainstSchema(inputSchema(), map[string]interface{}{"message": ""})
	assert.Contains(t, summariseErrors(tooShort), "at least 1 characters")

	tooLong := validateAgainstSchema(inputSchema(), map[string]interface{}{"message": "far too long a message"})
	assert.Contains(t, summariseErrors(tooLong), "at most 10 characters")

	tooBig := validateAgainstSchema(inputSchema(), map[string]interface{}{"message": "ok", "count": float64(99)})
	assert.Contains(t, summariseErrors(tooBig), "at most")

	tooSmall := validateAgainstSchema(inputSchema(), map[string]interface{}{"message": "ok", "count": float64(-1)})
	assert.Contains(t, summariseErrors(tooSmall), "at least")
}

func TestSchemaValidation_EnforcesEnum(t *testing.T) {
	errs := validateAgainstSchema(inputSchema(), map[string]interface{}{"message": "ok", "mode": "sideways"})
	require.NotEmpty(t, errs)
	assert.Contains(t, summariseErrors(errs), "must be one of")
}

func TestSchemaValidation_ValidatesNestedItems(t *testing.T) {
	errs := validateAgainstSchema(inputSchema(), map[string]interface{}{
		"message": "ok",
		"history": []interface{}{
			map[string]interface{}{"role": "user"},
			map[string]interface{}{"content": "missing role"},
		},
	})
	require.NotEmpty(t, errs)
	assert.Contains(t, summariseErrors(errs), "role")
}

// A rule this validator does not implement must not make a valid payload fail.
func TestSchemaValidation_IgnoresUnsupportedKeywords(t *testing.T) {
	schema := map[string]interface{}{
		"type": "object",
		"properties": map[string]interface{}{
			"email": map[string]interface{}{"type": "string", "format": "email", "pattern": "^.+@.+$"},
		},
	}
	errs := validateAgainstSchema(schema, map[string]interface{}{"email": "not-an-email"})
	assert.Empty(t, errs, "unsupported keywords must be ignored, not reported as failures")
}

func TestSchemaValidation_EmptySchemaAcceptsAnything(t *testing.T) {
	assert.Empty(t, validateAgainstSchema(nil, map[string]interface{}{"anything": 1}))
	assert.Empty(t, validateAgainstSchema(map[string]interface{}{}, "a string"))
}

func TestSchemaValidation_RejectsNonObjectAtRoot(t *testing.T) {
	errs := validateAgainstSchema(inputSchema(), "a bare string")
	require.NotEmpty(t, errs)
	assert.Contains(t, summariseErrors(errs), "must be an object")
}

func TestSchemaValidation_AdditionalProperties(t *testing.T) {
	schema := map[string]interface{}{
		"type":                 "object",
		"properties":           map[string]interface{}{"known": map[string]interface{}{"type": "string"}},
		"additionalProperties": false,
	}
	errs := validateAgainstSchema(schema, map[string]interface{}{"known": "x", "surprise": 1})
	require.NotEmpty(t, errs)
	assert.Contains(t, summariseErrors(errs), "surprise")
}

// Results must not depend on Go's map iteration order.
func TestSchemaValidation_IsDeterministic(t *testing.T) {
	payload := map[string]interface{}{"count": "wrong", "mode": "sideways"}

	first := summariseErrors(validateAgainstSchema(inputSchema(), payload))
	for i := 0; i < 20; i++ {
		assert.Equal(t, first, summariseErrors(validateAgainstSchema(inputSchema(), payload)))
	}
}

// Arbitrary values must never panic the validator.
func TestSchemaValidation_DoesNotPanic(t *testing.T) {
	values := []interface{}{
		nil, "", 0, false, []interface{}{}, map[string]interface{}{},
		map[string]interface{}{"message": nil},
		map[string]interface{}{"history": []interface{}{nil, 1, "x"}},
		[]interface{}{map[string]interface{}{"deep": []interface{}{map[string]interface{}{}}}},
	}

	for _, v := range values {
		func() {
			defer func() {
				if r := recover(); r != nil {
					t.Fatalf("validator panicked on %#v: %v", v, r)
				}
			}()
			_ = validateAgainstSchema(inputSchema(), v)
		}()
	}
}
