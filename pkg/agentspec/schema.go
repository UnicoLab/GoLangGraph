// Copyright (c) 2024 GoLangGraph Team
//
// Licensed under the MIT License. See LICENSE file in the project root for full license information.
//
// Package: GoLangGraph - A powerful Go framework for building AI agent workflows

package agentspec

import (
	"bytes"
	"encoding"
	"encoding/json"
	"fmt"
	"io"
	"reflect"
	"strconv"
	"strings"
	"unicode"
	"unicode/utf8"
)

// Schema is the JSON Schema subset generated for plain Go value types. The
// initial contract supports structs, slices, strings, booleans and numbers.
// Pointers, maps, embedded fields and custom JSON encoders are rejected rather
// than advertising a schema that differs from the actual wire representation.
type Schema struct {
	Type                 string            `json:"type"`
	Properties           map[string]Schema `json:"properties,omitempty"`
	Required             []string          `json:"required,omitempty"`
	Items                *Schema           `json:"items,omitempty"`
	AdditionalProperties *bool             `json:"additionalProperties,omitempty"`
	Description          string            `json:"description,omitempty"`
	Enum                 []string          `json:"enum,omitempty"`
	MinLength            *int              `json:"minLength,omitempty"`
	MaxLength            *int              `json:"maxLength,omitempty"`
}

func schemaFor(t reflect.Type, visiting map[reflect.Type]bool) (Schema, error) {
	if len(visiting) > 32 {
		return Schema{}, fmt.Errorf("schema nesting exceeds 32 levels")
	}
	if t == nil || visiting[t] {
		return Schema{}, fmt.Errorf("unsupported recursive or interface type %v", t)
	}
	marshaler := reflect.TypeOf((*json.Marshaler)(nil)).Elem()
	unmarshaler := reflect.TypeOf((*json.Unmarshaler)(nil)).Elem()
	textMarshaler := reflect.TypeOf((*encoding.TextMarshaler)(nil)).Elem()
	textUnmarshaler := reflect.TypeOf((*encoding.TextUnmarshaler)(nil)).Elem()
	if t.Implements(marshaler) || reflect.PointerTo(t).Implements(marshaler) || reflect.PointerTo(t).Implements(unmarshaler) ||
		t.Implements(textMarshaler) || reflect.PointerTo(t).Implements(textMarshaler) || reflect.PointerTo(t).Implements(textUnmarshaler) {
		return Schema{}, fmt.Errorf("custom JSON encoding is unsupported for %v", t)
	}
	visiting[t] = true
	defer delete(visiting, t)
	s := Schema{}
	switch t.Kind() {
	case reflect.String:
		s.Type = "string"
	case reflect.Bool:
		s.Type = "boolean"
	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64,
		reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64:
		s.Type = "integer"
	case reflect.Float32, reflect.Float64:
		s.Type = "number"
	case reflect.Slice:
		if t.Elem().Kind() == reflect.Uint8 {
			return Schema{}, fmt.Errorf("byte slices have a special JSON representation; use a string")
		}
		item, err := schemaFor(t.Elem(), visiting)
		if err != nil {
			return Schema{}, err
		}
		s.Type, s.Items = "array", &item
	case reflect.Struct:
		s.Type = "object"
		allow := false
		s.AdditionalProperties = &allow
		s.Properties = make(map[string]Schema)
		for i := 0; i < t.NumField(); i++ {
			field := t.Field(i)
			if field.Tag.Get("json") == "-" {
				continue
			}
			if field.Anonymous {
				return Schema{}, fmt.Errorf("embedded field %s is unsupported", field.Name)
			}
			if field.PkgPath != "" {
				continue
			}
			parts := strings.Split(field.Tag.Get("json"), ",")
			name := parts[0]
			if name == "" {
				name = field.Name
			}
			// Match encoding/json's accepted tag characters. It silently ignores
			// invalid names; a contract must reject them instead of diverging.
			for _, char := range name {
				if !unicode.IsLetter(char) && !unicode.IsDigit(char) && !strings.ContainsRune("!#$%&()*+-./:;<=>?@[]^_{|}~ ", char) {
					return Schema{}, fmt.Errorf("invalid JSON field name %q", name)
				}
			}
			optional := false
			for _, option := range parts[1:] {
				if option != "omitempty" {
					return Schema{}, fmt.Errorf("unsupported JSON option %q on %s", option, field.Name)
				}
				optional = true
			}
			if _, exists := s.Properties[name]; exists {
				return Schema{}, fmt.Errorf("duplicate JSON field %q", name)
			}
			property, err := schemaFor(field.Type, visiting)
			if err != nil {
				return Schema{}, fmt.Errorf("field %s: %w", name, err)
			}
			property.Description = field.Tag.Get("description")
			if enum, exists := field.Tag.Lookup("enum"); exists {
				if property.Type != "string" || enum == "" {
					return Schema{}, fmt.Errorf("enum on %s requires a string and comma-separated values", name)
				}
				property.Enum = strings.Split(enum, ",")
				seen := make(map[string]bool)
				for _, value := range property.Enum {
					if seen[value] {
						return Schema{}, fmt.Errorf("duplicate enum value on %s", name)
					}
					seen[value] = true
				}
			}
			for tag, target := range map[string]**int{"minLength": &property.MinLength, "maxLength": &property.MaxLength} {
				if raw, exists := field.Tag.Lookup(tag); exists {
					value, err := strconv.Atoi(raw)
					if property.Type != "string" || err != nil || value < 0 {
						return Schema{}, fmt.Errorf("%s on %s requires a non-negative string length", tag, name)
					}
					*target = &value
				}
			}
			if property.MinLength != nil && property.MaxLength != nil && *property.MinLength > *property.MaxLength {
				return Schema{}, fmt.Errorf("minLength exceeds maxLength on %s", name)
			}
			s.Properties[name] = property
			if !optional {
				s.Required = append(s.Required, name)
			}
		}
	default:
		return Schema{}, fmt.Errorf("unsupported contract type %v", t)
	}
	return s, nil
}

func (s Schema) clone() Schema {
	out := s
	out.Required = append([]string(nil), s.Required...)
	out.Enum = append([]string(nil), s.Enum...)
	if s.MinLength != nil {
		value := *s.MinLength
		out.MinLength = &value
	}
	if s.MaxLength != nil {
		value := *s.MaxLength
		out.MaxLength = &value
	}
	if s.AdditionalProperties != nil {
		value := *s.AdditionalProperties
		out.AdditionalProperties = &value
	}
	if s.Properties != nil {
		out.Properties = make(map[string]Schema, len(s.Properties))
		for key, value := range s.Properties {
			out.Properties[key] = value.clone()
		}
	}
	if s.Items != nil {
		item := s.Items.clone()
		out.Items = &item
	}
	return out
}

func (s Schema) validate(value any, path string) error {
	valid := false
	switch s.Type {
	case "object":
		object, ok := value.(map[string]any)
		if !ok {
			break
		}
		for _, name := range s.Required {
			if _, exists := object[name]; !exists {
				return fmt.Errorf("%s.%s is required", path, name)
			}
		}
		for name, field := range object {
			property, exists := s.Properties[name]
			if !exists {
				return fmt.Errorf("%s.%s is not declared", path, name)
			}
			if err := property.validate(field, path+"."+name); err != nil {
				return err
			}
		}
		valid = true
	case "array":
		values, ok := value.([]any)
		if !ok {
			break
		}
		for i, item := range values {
			if err := s.Items.validate(item, fmt.Sprintf("%s[%d]", path, i)); err != nil {
				return err
			}
		}
		valid = true
	case "string":
		var text string
		text, valid = value.(string)
		if valid {
			length := utf8.RuneCountInString(text)
			if s.MinLength != nil && length < *s.MinLength {
				return fmt.Errorf("%s is shorter than minLength %d", path, *s.MinLength)
			}
			if s.MaxLength != nil && length > *s.MaxLength {
				return fmt.Errorf("%s exceeds maxLength %d", path, *s.MaxLength)
			}
			if len(s.Enum) > 0 {
				found := false
				for _, allowed := range s.Enum {
					if text == allowed {
						found = true
					}
				}
				if !found {
					return fmt.Errorf("%s must be one of %v", path, s.Enum)
				}
			}
		}
	case "boolean":
		_, valid = value.(bool)
	case "number":
		_, valid = value.(json.Number)
	case "integer":
		if number, ok := value.(json.Number); ok {
			_, signedErr := strconv.ParseInt(string(number), 10, 64)
			_, unsignedErr := strconv.ParseUint(string(number), 10, 64)
			valid = signedErr == nil || unsignedErr == nil
		}
	}
	if !valid {
		return fmt.Errorf("%s must be %s", path, s.Type)
	}
	return nil
}

// Validator adds semantic validation to a typed input or output. Implement it
// on a value or pointer receiver; schemas describe structure, Validate describes
// domain rules such as non-empty requests or allowed styles.
type Validator interface {
	Validate() error
}

func decode[T any](data json.RawMessage, schema Schema) (T, error) {
	var output T
	if err := validateJSON(data, 64); err != nil {
		return output, err
	}
	decoder := json.NewDecoder(bytes.NewReader(data))
	decoder.UseNumber()
	var value any
	if err := decoder.Decode(&value); err != nil {
		return output, err
	}
	if err := decoder.Decode(new(any)); err != io.EOF {
		return output, fmt.Errorf("expected exactly one JSON value")
	}
	if err := schema.validate(value, "$"); err != nil {
		return output, err
	}
	if err := json.Unmarshal(data, &output); err != nil {
		return output, err
	}
	if validator, ok := any(&output).(Validator); ok {
		if err := validator.Validate(); err != nil {
			return output, err
		}
	}
	return output, nil
}

// validateJSON rejects ambiguous duplicate keys (including escaped spellings),
// excessive nesting and trailing values before decoding into a Go struct.
func validateJSON(data []byte, maximumDepth int) error {
	if !utf8.Valid(data) {
		return fmt.Errorf("JSON must be valid UTF-8")
	}
	decoder := json.NewDecoder(bytes.NewReader(data))
	decoder.UseNumber()
	var value func(int) error
	value = func(depth int) error {
		if depth > maximumDepth {
			return fmt.Errorf("JSON nesting exceeds %d levels", maximumDepth)
		}
		token, err := decoder.Token()
		if err != nil {
			return err
		}
		delimiter, ok := token.(json.Delim)
		if !ok {
			return nil
		}
		switch delimiter {
		case '{':
			seen := make(map[string]bool)
			for decoder.More() {
				key, keyErr := decoder.Token()
				if keyErr != nil {
					return keyErr
				}
				name, ok := key.(string)
				if !ok || seen[name] {
					return fmt.Errorf("invalid or duplicate JSON key %q", key)
				}
				seen[name] = true
				if nestedErr := value(depth + 1); nestedErr != nil {
					return nestedErr
				}
			}
		case '[':
			for decoder.More() {
				if nestedErr := value(depth + 1); nestedErr != nil {
					return nestedErr
				}
			}
		default:
			return fmt.Errorf("unexpected JSON delimiter %s", delimiter)
		}
		_, err = decoder.Token()
		return err
	}
	if err := value(0); err != nil {
		return err
	}
	if _, err := decoder.Token(); err != io.EOF {
		return fmt.Errorf("expected exactly one JSON value")
	}
	return nil
}
