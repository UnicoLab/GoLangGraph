// Copyright (c) 2024 GoLangGraph Team
//
// Licensed under the MIT License. See LICENSE file in the project root for full license information.
//
// Package: GoLangGraph - A powerful Go framework for building AI agent workflows

package server

import (
	"fmt"
	"sort"
	"strings"
)

// validateAgainstSchema checks a decoded JSON value against the JSON Schema
// subset that generateAgentSchema produces: type, properties, required,
// additionalProperties, minLength, maxLength, minimum, maximum, enum and items.
//
// The validation endpoint previously answered "valid": true for every input
// without inspecting it, so a client using it as a gate accepted anything.
// Unsupported keywords are ignored rather than treated as failures, so a richer
// schema is never reported invalid for a rule this validator cannot check.
func validateAgainstSchema(schema map[string]interface{}, value interface{}) []string {
	if len(schema) == 0 {
		return nil
	}
	var errs []string
	validateValue(schema, value, "", &errs)
	sort.Strings(errs)
	return errs
}

func fieldLabel(path string) string {
	if path == "" {
		return "value"
	}
	return path
}

func join(path, key string) string {
	if path == "" {
		return key
	}
	return path + "." + key
}

func validateValue(schema map[string]interface{}, value interface{}, path string, errs *[]string) {
	expected, _ := schema["type"].(string)

	if enum, ok := schema["enum"].([]interface{}); ok && len(enum) > 0 {
		matched := false
		for _, candidate := range enum {
			if fmt.Sprint(candidate) == fmt.Sprint(value) {
				matched = true
				break
			}
		}
		if !matched {
			*errs = append(*errs, fmt.Sprintf("%s must be one of %v", fieldLabel(path), enum))
		}
	}

	switch expected {
	case "object":
		obj, ok := value.(map[string]interface{})
		if !ok {
			*errs = append(*errs, fmt.Sprintf("%s must be an object", fieldLabel(path)))
			return
		}
		validateObject(schema, obj, path, errs)

	case "array":
		items, ok := toArray(value)
		if !ok {
			*errs = append(*errs, fmt.Sprintf("%s must be an array", fieldLabel(path)))
			return
		}
		if itemSchema, ok := schema["items"].(map[string]interface{}); ok {
			for i, item := range items {
				validateValue(itemSchema, item, fmt.Sprintf("%s[%d]", fieldLabel(path), i), errs)
			}
		}

	case "string":
		str, ok := value.(string)
		if !ok {
			*errs = append(*errs, fmt.Sprintf("%s must be a string", fieldLabel(path)))
			return
		}
		if min, ok := toNumber(schema["minLength"]); ok && float64(len(str)) < min {
			*errs = append(*errs, fmt.Sprintf("%s must be at least %d characters", fieldLabel(path), int(min)))
		}
		if max, ok := toNumber(schema["maxLength"]); ok && float64(len(str)) > max {
			*errs = append(*errs, fmt.Sprintf("%s must be at most %d characters", fieldLabel(path), int(max)))
		}

	case "number", "integer":
		num, ok := toNumber(value)
		if !ok {
			*errs = append(*errs, fmt.Sprintf("%s must be a number", fieldLabel(path)))
			return
		}
		if expected == "integer" && num != float64(int64(num)) {
			*errs = append(*errs, fmt.Sprintf("%s must be an integer", fieldLabel(path)))
		}
		if min, ok := toNumber(schema["minimum"]); ok && num < min {
			*errs = append(*errs, fmt.Sprintf("%s must be at least %v", fieldLabel(path), min))
		}
		if max, ok := toNumber(schema["maximum"]); ok && num > max {
			*errs = append(*errs, fmt.Sprintf("%s must be at most %v", fieldLabel(path), max))
		}

	case "boolean":
		if _, ok := value.(bool); !ok {
			*errs = append(*errs, fmt.Sprintf("%s must be a boolean", fieldLabel(path)))
		}

	case "":
		// No declared type: only structural keywords apply.
		if obj, ok := value.(map[string]interface{}); ok {
			validateObject(schema, obj, path, errs)
		}
	}
}

func validateObject(schema map[string]interface{}, obj map[string]interface{}, path string, errs *[]string) {
	properties, _ := schema["properties"].(map[string]interface{})

	if required, ok := toStringSlice(schema["required"]); ok {
		for _, key := range required {
			if _, present := obj[key]; !present {
				*errs = append(*errs, fmt.Sprintf("%s is required", join(path, key)))
			}
		}
	}

	if len(properties) > 0 {
		keys := make([]string, 0, len(obj))
		for k := range obj {
			keys = append(keys, k)
		}
		sort.Strings(keys)

		for _, key := range keys {
			propSchema, declared := properties[key].(map[string]interface{})
			if !declared {
				if allow, ok := schema["additionalProperties"].(bool); ok && !allow {
					*errs = append(*errs, fmt.Sprintf("%s is not an allowed property", join(path, key)))
				}
				continue
			}
			validateValue(propSchema, obj[key], join(path, key), errs)
		}
	}
}

func toArray(value interface{}) ([]interface{}, bool) {
	switch v := value.(type) {
	case []interface{}:
		return v, true
	case nil:
		return nil, false
	}
	return nil, false
}

func toNumber(value interface{}) (float64, bool) {
	switch v := value.(type) {
	case float64:
		return v, true
	case float32:
		return float64(v), true
	case int:
		return float64(v), true
	case int32:
		return float64(v), true
	case int64:
		return float64(v), true
	}
	return 0, false
}

func toStringSlice(value interface{}) ([]string, bool) {
	switch v := value.(type) {
	case []string:
		return v, true
	case []interface{}:
		out := make([]string, 0, len(v))
		for _, item := range v {
			out = append(out, fmt.Sprint(item))
		}
		return out, true
	}
	return nil, false
}

// schemaSection extracts the "input" or "output" half of a generated schema.
func schemaSection(schema map[string]interface{}, section string) map[string]interface{} {
	if schema == nil {
		return nil
	}
	if sub, ok := schema[section].(map[string]interface{}); ok {
		return sub
	}
	return nil
}

// summariseErrors renders validation errors for a log line.
func summariseErrors(errs []string) string {
	if len(errs) == 0 {
		return ""
	}
	return strings.Join(errs, "; ")
}
