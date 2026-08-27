// Copyright (c) 2024 GoLangGraph Team
//
// Licensed under the MIT License. See LICENSE file in the project root for full license information.
//
// Package: GoLangGraph - A powerful Go framework for building AI agent workflows

package server

import (
	"encoding/json"
	"net/http"

	"github.com/gorilla/mux"
)

// handleCreatePipeline creates an executable, sequential multi-agent graph
// from Studio's declarative pipeline definition.  It is intentionally a
// separate resource from general graphs because externally registered graphs
// can contain application-owned functions and must not be overwritten by UI.
func (s *Server) handleCreatePipeline(w http.ResponseWriter, r *http.Request) {
	if s.graphManager == nil {
		s.writeError(w, http.StatusServiceUnavailable, "graph manager not available")
		return
	}
	if s.agentManager == nil {
		s.writeError(w, http.StatusServiceUnavailable, "agent manager not available")
		return
	}
	var definition AgentPipelineDefinition
	if err := json.NewDecoder(r.Body).Decode(&definition); err != nil {
		s.writeError(w, http.StatusBadRequest, "invalid pipeline definition")
		return
	}
	graph, err := BuildAgentPipeline(definition, s.agentManager)
	if err != nil {
		s.writeError(w, http.StatusBadRequest, err.Error())
		return
	}
	s.graphManager.Register(definition.ID, graph)
	s.writeJSON(w, http.StatusCreated, map[string]interface{}{
		"pipeline": summariseGraph(definition.ID, graph),
		"topology": describeTopology(graph),
	})
}

// handleDeletePipeline only removes pipelines that were created through the
// Studio API.  Registered application graphs are deliberately protected.
func (s *Server) handleDeletePipeline(w http.ResponseWriter, r *http.Request) {
	if s.graphManager == nil {
		s.writeError(w, http.StatusServiceUnavailable, "graph manager not available")
		return
	}
	id := mux.Vars(r)["id"]
	graph, ok := s.graphManager.Get(id)
	if !ok {
		s.writeError(w, http.StatusNotFound, "pipeline not found")
		return
	}
	createdByStudio, _ := graph.Metadata["studio_pipeline"].(bool)
	if !createdByStudio {
		s.writeError(w, http.StatusConflict, "registered application graphs cannot be deleted through the pipeline API")
		return
	}
	s.graphManager.Unregister(id)
	s.writeJSON(w, http.StatusOK, map[string]string{"message": "pipeline deleted successfully"})
}
