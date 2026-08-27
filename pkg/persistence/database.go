// Copyright (c) 2024 GoLangGraph Team
//
// Licensed under the MIT License. See LICENSE file in the project root for full license information.
//
// Package: GoLangGraph - A powerful Go framework for building AI agent workflows

package persistence

import (
	"context"
	"database/sql"
	"encoding/json"
	"errors"
	"fmt"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/go-redis/redis/v8"
	_ "github.com/lib/pq"
	"github.com/sirupsen/logrus"

	"github.com/UnicoLab/GoLangGraph/pkg/core"
)

// ErrSessionStoreUnavailable means a SessionManager was created without a
// durable database connection. Servers use it to return a truthful 503 rather
// than panicking or pretending that session data was stored.
var ErrSessionStoreUnavailable = errors.New("session manager has no database connection")

// DatabaseType represents supported database types
type DatabaseType string

const (
	DatabaseTypePostgres   DatabaseType = "postgres"
	DatabaseTypePostgresQL DatabaseType = "postgresql"
	DatabaseTypePgVector   DatabaseType = "pgvector"
	DatabaseTypeRedis      DatabaseType = "redis"
	DatabaseTypeOpenSearch DatabaseType = "opensearch"
	DatabaseTypeElastic    DatabaseType = "elasticsearch"
	DatabaseTypeMongoDB    DatabaseType = "mongodb"
	DatabaseTypeMySQL      DatabaseType = "mysql"
	DatabaseTypeSQLite     DatabaseType = "sqlite"
)

// DatabaseConfig represents database configuration
type DatabaseConfig struct {
	Type         DatabaseType `json:"type"` // "postgres", "pgvector", "redis", "opensearch", etc.
	Host         string       `json:"host"`
	Port         int          `json:"port"`
	Database     string       `json:"database"`
	Username     string       `json:"username"`
	Password     string       `json:"password"`
	SSLMode      string       `json:"ssl_mode"`
	MaxOpenConns int          `json:"max_open_conns"`
	MaxIdleConns int          `json:"max_idle_conns"`
	MaxLifetime  string       `json:"max_lifetime"`

	// CheckpointTTL bounds how long a checkpoint survives in stores that expire
	// keys (Redis). Empty means the 24h default. Set it to a duration string
	// such as "168h"; "0" disables expiry entirely.
	CheckpointTTL string `json:"checkpoint_ttl"`

	// Vector-specific configuration
	VectorDimension int    `json:"vector_dimension"`
	VectorMetric    string `json:"vector_metric"` // "cosine", "euclidean", "dot_product"

	// OpenSearch/Elasticsearch specific
	Index   string `json:"index"`
	APIKey  string `json:"api_key"`
	CloudID string `json:"cloud_id"`
	CACert  string `json:"ca_cert"`

	// Additional connection parameters
	ConnectionParams map[string]string `json:"connection_params"`

	// RAG-specific settings
	EnableRAG           bool    `json:"enable_rag"`
	EmbeddingModel      string  `json:"embedding_model"`
	EmbeddingDimension  int     `json:"embedding_dimension"`
	SimilarityThreshold float64 `json:"similarity_threshold"`
}

// DatabaseConnection represents a database connection interface
type DatabaseConnection interface {
	Connect() error
	Close() error
	Ping() error
	GetType() DatabaseType
	GetConfig() *DatabaseConfig
	ExecuteQuery(ctx context.Context, query string, args ...interface{}) error
	QueryRow(ctx context.Context, query string, args ...interface{}) interface{}
	QueryRows(ctx context.Context, query string, args ...interface{}) (interface{}, error)
}

// PostgresConnection implements PostgreSQL connection
type PostgresConnection struct {
	db     *sql.DB
	config *DatabaseConfig
	logger *logrus.Logger
}

// NewPostgresConnection creates a new PostgreSQL connection
func NewPostgresConnection(config *DatabaseConfig) (*PostgresConnection, error) {
	conn := &PostgresConnection{
		config: config,
		logger: logrus.New(),
	}

	if err := conn.Connect(); err != nil {
		return nil, err
	}

	return conn, nil
}

// Connect establishes the PostgreSQL connection
func (p *PostgresConnection) Connect() error {
	var dsn string

	// Support different PostgreSQL connection formats
	switch p.config.Type {
	case DatabaseTypePostgres, DatabaseTypePostgresQL:
		dsn = fmt.Sprintf("host=%s port=%d user=%s password=%s dbname=%s sslmode=%s",
			p.config.Host, p.config.Port, p.config.Username, p.config.Password, p.config.Database, p.config.SSLMode)
	case DatabaseTypePgVector:
		// PostgreSQL with pgvector extension
		dsn = fmt.Sprintf("host=%s port=%d user=%s password=%s dbname=%s sslmode=%s",
			p.config.Host, p.config.Port, p.config.Username, p.config.Password, p.config.Database, p.config.SSLMode)
	default:
		return fmt.Errorf("unsupported database type: %s", p.config.Type)
	}

	// Add additional connection parameters
	if p.config.ConnectionParams != nil {
		var params []string
		for k, v := range p.config.ConnectionParams {
			params = append(params, fmt.Sprintf("%s=%s", k, v))
		}
		if len(params) > 0 {
			dsn += " " + strings.Join(params, " ")
		}
	}

	db, err := sql.Open("postgres", dsn)
	if err != nil {
		return fmt.Errorf("failed to connect to database: %w", err)
	}

	// Configure connection pool
	if p.config.MaxOpenConns > 0 {
		db.SetMaxOpenConns(p.config.MaxOpenConns)
	} else {
		db.SetMaxOpenConns(25) // Default
	}

	if p.config.MaxIdleConns > 0 {
		db.SetMaxIdleConns(p.config.MaxIdleConns)
	} else {
		db.SetMaxIdleConns(5) // Default
	}

	if p.config.MaxLifetime != "" {
		// A typo'd duration used to be swallowed silently, leaving connections
		// with no lifetime cap at all -- the opposite of what the operator
		// configured. Fail loudly instead.
		duration, perr := time.ParseDuration(p.config.MaxLifetime)
		if perr != nil {
			_ = db.Close()
			return fmt.Errorf("invalid max_lifetime %q: %w", p.config.MaxLifetime, perr)
		}
		db.SetConnMaxLifetime(duration)
	} else {
		db.SetConnMaxLifetime(5 * time.Minute) // Default
	}

	p.db = db
	if err := p.Ping(); err != nil {
		// Ping failing leaves an open *sql.DB (and its pool goroutines) behind
		// unless we close it; callers only see the error and drop the object.
		_ = db.Close()
		p.db = nil
		return err
	}
	return nil
}

// Ping tests the database connection
func (p *PostgresConnection) Ping() error {
	if p.db == nil {
		return fmt.Errorf("database connection is nil")
	}

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	if p.db == nil {
		return fmt.Errorf("database connection is not open")
	}
	return p.db.PingContext(ctx)
}

// Close closes the database connection
func (p *PostgresConnection) Close() error {
	if p.db != nil {
		return p.db.Close()
	}
	return nil
}

// GetType returns the database type
func (p *PostgresConnection) GetType() DatabaseType {
	return p.config.Type
}

// GetConfig returns the database configuration
func (p *PostgresConnection) GetConfig() *DatabaseConfig {
	return p.config
}

// ExecuteQuery executes a query without returning results
func (p *PostgresConnection) ExecuteQuery(ctx context.Context, query string, args ...interface{}) error {
	if p.db == nil {
		return fmt.Errorf("database connection is not open")
	}
	_, err := p.db.ExecContext(ctx, query, args...)
	return err
}

// QueryRow executes a query that returns a single row
func (p *PostgresConnection) QueryRow(ctx context.Context, query string, args ...interface{}) interface{} {
	if p.db == nil {
		return nil
	}
	return p.db.QueryRowContext(ctx, query, args...)
}

// QueryRows executes a query that returns multiple rows
func (p *PostgresConnection) QueryRows(ctx context.Context, query string, args ...interface{}) (interface{}, error) {
	if p.db == nil {
		return nil, fmt.Errorf("database connection is not open")
	}
	return p.db.QueryContext(ctx, query, args...)
}

// Exec runs a statement and returns its sql.Result.
//
// ExecuteQuery throws the result away, which made it impossible for callers to
// tell "deleted one row" from "matched nothing" -- see Delete below.
func (p *PostgresConnection) Exec(ctx context.Context, query string, args ...interface{}) (sql.Result, error) {
	if p.db == nil {
		return nil, fmt.Errorf("database connection is not open")
	}
	return p.db.ExecContext(ctx, query, args...)
}

// WithTx runs fn inside a transaction, committing on success and rolling back
// on any error or panic.
func (p *PostgresConnection) WithTx(ctx context.Context, fn func(*sql.Tx) error) (err error) {
	if p.db == nil {
		return fmt.Errorf("database connection is not open")
	}

	tx, err := p.db.BeginTx(ctx, nil)
	if err != nil {
		return fmt.Errorf("failed to begin transaction: %w", err)
	}

	defer func() {
		if r := recover(); r != nil {
			_ = tx.Rollback()
			panic(r)
		}
		if err != nil {
			// Rollback error is deliberately not surfaced: the caller needs the
			// original failure, and a rollback after a failed statement is
			// frequently a no-op the driver reports as ErrTxDone.
			_ = tx.Rollback()
		}
	}()

	if err = fn(tx); err != nil {
		return err
	}

	if err = tx.Commit(); err != nil {
		return fmt.Errorf("failed to commit transaction: %w", err)
	}
	return nil
}

// asSQLRow converts a DatabaseConnection result to *sql.Row.
//
// DatabaseConnection is a public interface returning interface{}, so an
// implementation other than PostgresConnection previously caused a panic at
// the unchecked type assertion. Returning an error keeps a custom or absent
// backend from crashing the process.
func asSQLRow(v interface{}) (*sql.Row, error) {
	row, ok := v.(*sql.Row)
	if !ok || row == nil {
		return nil, fmt.Errorf("database connection returned %T, want *sql.Row", v)
	}
	return row, nil
}

// asSQLRows is the multi-row counterpart of asSQLRow.
//
// The row-iterating code used to write `rows.(*sql.Rows)` inline -- a
// single-value type assertion that panics rather than erroring. QueryRows is
// declared on the public DatabaseConnection interface as returning interface{},
// so any implementation other than PostgresConnection crashed the process.
func asSQLRows(v interface{}) (*sql.Rows, error) {
	rows, ok := v.(*sql.Rows)
	if !ok || rows == nil {
		return nil, fmt.Errorf("database connection returned %T, want *sql.Rows", v)
	}
	return rows, nil
}

// decodeJSONMap unmarshals a JSONB column into a map, tolerating SQL NULL and
// the JSON literal null.
//
// Every caller previously ran json.Unmarshal on the raw bytes, so a row with a
// NULL metadata column -- which the schema permits, and which any row written
// by another tool or an older release may well have -- failed with "unexpected
// end of JSON input". That broke Load *and* List, and a broken List breaks
// Latest(), i.e. resuming a thread at all.
func decodeJSONMap(data []byte, target *map[string]interface{}) error {
	if len(data) == 0 || string(data) == "null" {
		*target = map[string]interface{}{}
		return nil
	}
	if err := json.Unmarshal(data, target); err != nil {
		return err
	}
	if *target == nil {
		*target = map[string]interface{}{}
	}
	return nil
}

// encodeVector renders a float slice as a pgvector literal ("[1,2,3]").
//
// The RAG methods used to hand []float64 straight to database/sql, which
// rejects it with "unsupported type []float64, a slice of float64" -- so
// SaveDocument and the vector branch of SearchDocuments could never succeed.
func encodeVector(v []float64) string {
	var b strings.Builder
	b.WriteByte('[')
	for i, f := range v {
		if i > 0 {
			b.WriteByte(',')
		}
		b.WriteString(strconv.FormatFloat(f, 'f', -1, 64))
	}
	b.WriteByte(']')
	return b.String()
}

// decodeVector parses a pgvector literal back into a float slice. Returns nil
// for SQL NULL so an absent embedding stays absent rather than becoming [].
func decodeVector(raw interface{}) ([]float64, error) {
	var s string
	switch v := raw.(type) {
	case nil:
		return nil, nil
	case []byte:
		s = string(v)
	case string:
		s = v
	default:
		return nil, fmt.Errorf("unexpected embedding column type %T", raw)
	}

	s = strings.TrimSpace(s)
	s = strings.TrimPrefix(s, "[")
	s = strings.TrimSuffix(s, "]")
	if s == "" {
		return nil, nil
	}

	parts := strings.Split(s, ",")
	out := make([]float64, 0, len(parts))
	for _, p := range parts {
		f, err := strconv.ParseFloat(strings.TrimSpace(p), 64)
		if err != nil {
			return nil, fmt.Errorf("failed to parse embedding component %q: %w", p, err)
		}
		out = append(out, f)
	}
	return out, nil
}

// PostgresCheckpointer implements database-based checkpointing with PostgreSQL
type PostgresCheckpointer struct {
	conn   *PostgresConnection
	config *DatabaseConfig
	logger *logrus.Logger
}

// NewPostgresCheckpointer creates a new PostgreSQL checkpointer
func NewPostgresCheckpointer(config *DatabaseConfig) (*PostgresCheckpointer, error) {
	conn, err := NewPostgresConnection(config)
	if err != nil {
		return nil, fmt.Errorf("failed to create postgres connection: %w", err)
	}

	checkpointer := &PostgresCheckpointer{
		conn:   conn,
		config: config,
		logger: logrus.New(),
	}

	// Initialize schema
	if err := checkpointer.initSchema(); err != nil {
		// The connection pool was already open at this point and used to be
		// abandoned here, leaking sockets and goroutines on every failed start.
		_ = conn.Close()
		return nil, fmt.Errorf("failed to initialize schema: %w", err)
	}

	return checkpointer, nil
}

// initSchema initializes the database schema with enhanced support for RAG and vector operations
func (p *PostgresCheckpointer) initSchema() error {
	// Enable pgvector extension if using pgvector
	if p.config.Type == DatabaseTypePgVector {
		if err := p.conn.ExecuteQuery(context.Background(), "CREATE EXTENSION IF NOT EXISTS vector;"); err != nil {
			p.logger.Warnf("Failed to create vector extension (may not be available): %v", err)
		}
	}

	// Create main tables
	schema := `
	-- Threads table for conversation management
	CREATE TABLE IF NOT EXISTS threads (
		id VARCHAR(255) PRIMARY KEY,
		name VARCHAR(255),
		metadata JSONB,
		created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
		updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
	);

	-- Checkpoints table for state persistence
	CREATE TABLE IF NOT EXISTS checkpoints (
		id VARCHAR(255) PRIMARY KEY,
		thread_id VARCHAR(255) NOT NULL,
		state_data JSONB NOT NULL,
		metadata JSONB,
		created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
		node_id VARCHAR(255),
		step_id INTEGER,
		FOREIGN KEY (thread_id) REFERENCES threads(id) ON DELETE CASCADE
	);

	-- Sessions table for user session management
	CREATE TABLE IF NOT EXISTS sessions (
		id VARCHAR(255) PRIMARY KEY,
		thread_id VARCHAR(255) NOT NULL,
		user_id VARCHAR(255),
		metadata JSONB,
		created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
		expires_at TIMESTAMP WITH TIME ZONE,
		FOREIGN KEY (thread_id) REFERENCES threads(id) ON DELETE CASCADE
	);

	-- Create indexes for better performance
	CREATE INDEX IF NOT EXISTS idx_checkpoints_thread_id ON checkpoints(thread_id);
	CREATE INDEX IF NOT EXISTS idx_checkpoints_created_at ON checkpoints(created_at);
	CREATE INDEX IF NOT EXISTS idx_checkpoints_node_id ON checkpoints(node_id);
	CREATE INDEX IF NOT EXISTS idx_checkpoints_step_id ON checkpoints(step_id);
	CREATE INDEX IF NOT EXISTS idx_sessions_thread_id ON sessions(thread_id);
	CREATE INDEX IF NOT EXISTS idx_sessions_user_id ON sessions(user_id);
	CREATE INDEX IF NOT EXISTS idx_sessions_expires_at ON sessions(expires_at);
	`

	if err := p.conn.ExecuteQuery(context.Background(), schema); err != nil {
		return fmt.Errorf("failed to create basic schema: %w", err)
	}

	// Create RAG-specific tables if enabled
	if p.config.EnableRAG {
		if err := p.initRAGSchema(); err != nil {
			return fmt.Errorf("failed to initialize RAG schema: %w", err)
		}
	}

	return nil
}

// initRAGSchema initializes RAG-specific database schema
func (p *PostgresCheckpointer) initRAGSchema() error {
	var vectorSchema string

	if p.config.Type == DatabaseTypePgVector {
		// Use pgvector for vector storage
		vectorDim := p.config.VectorDimension
		if vectorDim == 0 {
			vectorDim = 1536 // Default OpenAI embedding dimension
		}

		vectorSchema = fmt.Sprintf(`
		-- Documents table for RAG document storage
		CREATE TABLE IF NOT EXISTS documents (
			id VARCHAR(255) PRIMARY KEY,
			thread_id VARCHAR(255),
			content TEXT NOT NULL,
			metadata JSONB,
			embedding vector(%d),
			created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
			updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
			FOREIGN KEY (thread_id) REFERENCES threads(id) ON DELETE CASCADE
		);

		-- Memory table for conversational memory with embeddings
		CREATE TABLE IF NOT EXISTS memory (
			id VARCHAR(255) PRIMARY KEY,
			thread_id VARCHAR(255) NOT NULL,
			user_id VARCHAR(255),
			content TEXT NOT NULL,
			memory_type VARCHAR(50) DEFAULT 'conversation',
			embedding vector(%d),
			metadata JSONB,
			created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
			FOREIGN KEY (thread_id) REFERENCES threads(id) ON DELETE CASCADE
		);

		-- Vector indexes for similarity search
		CREATE INDEX IF NOT EXISTS idx_documents_embedding ON documents USING ivfflat (embedding vector_cosine_ops);
		CREATE INDEX IF NOT EXISTS idx_memory_embedding ON memory USING ivfflat (embedding vector_cosine_ops);
		CREATE INDEX IF NOT EXISTS idx_documents_thread_id ON documents(thread_id);
		CREATE INDEX IF NOT EXISTS idx_memory_thread_id ON memory(thread_id);
		CREATE INDEX IF NOT EXISTS idx_memory_user_id ON memory(user_id);
		CREATE INDEX IF NOT EXISTS idx_memory_type ON memory(memory_type);
		`, vectorDim, vectorDim)
	} else {
		// Fallback without vector support
		vectorSchema = `
		-- Documents table for RAG document storage (without vectors)
		CREATE TABLE IF NOT EXISTS documents (
			id VARCHAR(255) PRIMARY KEY,
			thread_id VARCHAR(255),
			content TEXT NOT NULL,
			metadata JSONB,
			created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
			updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
			FOREIGN KEY (thread_id) REFERENCES threads(id) ON DELETE CASCADE
		);

		-- Memory table for conversational memory (without vectors)
		CREATE TABLE IF NOT EXISTS memory (
			id VARCHAR(255) PRIMARY KEY,
			thread_id VARCHAR(255) NOT NULL,
			user_id VARCHAR(255),
			content TEXT NOT NULL,
			memory_type VARCHAR(50) DEFAULT 'conversation',
			metadata JSONB,
			created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
			FOREIGN KEY (thread_id) REFERENCES threads(id) ON DELETE CASCADE
		);

		-- Indexes for text search
		CREATE INDEX IF NOT EXISTS idx_documents_thread_id ON documents(thread_id);
		CREATE INDEX IF NOT EXISTS idx_documents_content ON documents USING gin(to_tsvector('english', content));
		CREATE INDEX IF NOT EXISTS idx_memory_thread_id ON memory(thread_id);
		CREATE INDEX IF NOT EXISTS idx_memory_user_id ON memory(user_id);
		CREATE INDEX IF NOT EXISTS idx_memory_type ON memory(memory_type);
		CREATE INDEX IF NOT EXISTS idx_memory_content ON memory USING gin(to_tsvector('english', content));
		`
	}

	return p.conn.ExecuteQuery(context.Background(), vectorSchema)
}

// Save saves a checkpoint to PostgreSQL
func (p *PostgresCheckpointer) Save(ctx context.Context, checkpoint *Checkpoint) error {
	stateData, err := json.Marshal(checkpoint.State)
	if err != nil {
		return fmt.Errorf("failed to marshal state: %w", err)
	}

	metadataData, err := json.Marshal(checkpoint.Metadata)
	if err != nil {
		return fmt.Errorf("failed to marshal metadata: %w", err)
	}

	// checkpoints.thread_id carries a FOREIGN KEY to threads(id), but nothing in
	// the Checkpointer interface creates threads -- so every Save against a
	// thread that had not been registered out-of-band failed with
	// "violates foreign key constraint checkpoints_thread_id_fkey". The
	// in-memory and file checkpointers have no such requirement, so the
	// PostgreSQL backend was not usable as a drop-in Checkpointer at all.
	//
	// Registering the parent thread here makes Save self-sufficient. Both
	// statements run in one transaction so a failed checkpoint write cannot
	// leave an orphan thread row behind.
	const ensureThread = `
		INSERT INTO threads (id, created_at, updated_at)
		VALUES ($1, NOW(), NOW())
		ON CONFLICT (id) DO NOTHING
	`

	const upsertCheckpoint = `
		INSERT INTO checkpoints (id, thread_id, state_data, metadata, created_at, node_id, step_id)
		VALUES ($1, $2, $3, $4, $5, $6, $7)
		ON CONFLICT (id) DO UPDATE SET
			state_data = EXCLUDED.state_data,
			metadata = EXCLUDED.metadata,
			created_at = EXCLUDED.created_at,
			node_id = EXCLUDED.node_id,
			step_id = EXCLUDED.step_id
	`

	err = p.conn.WithTx(ctx, func(tx *sql.Tx) error {
		if _, txErr := tx.ExecContext(ctx, ensureThread, checkpoint.ThreadID); txErr != nil {
			return fmt.Errorf("failed to register thread: %w", txErr)
		}
		if _, txErr := tx.ExecContext(ctx, upsertCheckpoint,
			checkpoint.ID,
			checkpoint.ThreadID,
			stateData,
			metadataData,
			checkpoint.CreatedAt,
			checkpoint.NodeID,
			checkpoint.StepID,
		); txErr != nil {
			return txErr
		}
		return nil
	})

	if err != nil {
		return fmt.Errorf("failed to save checkpoint: %w", err)
	}

	p.logger.WithFields(logrus.Fields{
		"checkpoint_id": checkpoint.ID,
		"thread_id":     checkpoint.ThreadID,
	}).Info("Checkpoint saved to database")

	return nil
}

// Load loads a checkpoint from PostgreSQL
func (p *PostgresCheckpointer) Load(ctx context.Context, threadID, checkpointID string) (*Checkpoint, error) {
	query := `
		SELECT id, thread_id, state_data, metadata, created_at, node_id, step_id
		FROM checkpoints
		WHERE thread_id = $1 AND id = $2
	`

	row, err := asSQLRow(p.conn.QueryRow(ctx, query, threadID, checkpointID))
	if err != nil {
		return nil, err
	}

	var checkpoint Checkpoint
	var stateData, metadataData []byte
	// node_id and step_id are nullable in the schema; scanning a NULL straight
	// into string/int fails with "converting NULL to string is unsupported".
	var nodeID sql.NullString
	var stepID sql.NullInt64

	err = row.Scan(
		&checkpoint.ID,
		&checkpoint.ThreadID,
		&stateData,
		&metadataData,
		&checkpoint.CreatedAt,
		&nodeID,
		&stepID,
	)

	if err != nil {
		if err == sql.ErrNoRows {
			return nil, fmt.Errorf("checkpoint %s not found in thread %s", checkpointID, threadID)
		}
		return nil, fmt.Errorf("failed to load checkpoint: %w", err)
	}

	checkpoint.NodeID = nodeID.String
	checkpoint.StepID = int(stepID.Int64)

	// Unmarshal state
	var state core.BaseState
	if err := json.Unmarshal(stateData, &state); err != nil {
		return nil, fmt.Errorf("failed to unmarshal state: %w", err)
	}
	checkpoint.State = &state

	// Unmarshal metadata
	if err := decodeJSONMap(metadataData, &checkpoint.Metadata); err != nil {
		return nil, fmt.Errorf("failed to unmarshal metadata: %w", err)
	}

	return &checkpoint, nil
}

// List lists checkpoints for a thread
func (p *PostgresCheckpointer) List(ctx context.Context, threadID string) ([]*CheckpointMetadata, error) {
	query := `
		SELECT id, thread_id, metadata, created_at, node_id, step_id
		FROM checkpoints
		WHERE thread_id = $1
		ORDER BY created_at DESC
	`

	raw, err := p.conn.QueryRows(ctx, query, threadID)
	if err != nil {
		return nil, fmt.Errorf("failed to list checkpoints: %w", err)
	}
	rows, err := asSQLRows(raw)
	if err != nil {
		return nil, err
	}
	defer func() { _ = rows.Close() }()

	checkpoints := []*CheckpointMetadata{}
	for rows.Next() {
		var checkpoint CheckpointMetadata
		var metadataData []byte
		var nodeID sql.NullString
		var stepID sql.NullInt64

		err := rows.Scan(
			&checkpoint.ID,
			&checkpoint.ThreadID,
			&metadataData,
			&checkpoint.CreatedAt,
			&nodeID,
			&stepID,
		)

		if err != nil {
			return nil, fmt.Errorf("failed to scan checkpoint: %w", err)
		}

		checkpoint.NodeID = nodeID.String
		checkpoint.StepID = int(stepID.Int64)

		// Unmarshal metadata
		if err := decodeJSONMap(metadataData, &checkpoint.Metadata); err != nil {
			return nil, fmt.Errorf("failed to unmarshal metadata: %w", err)
		}

		checkpoints = append(checkpoints, &checkpoint)
	}

	// Without this check a connection that drops mid-iteration returns a
	// silently truncated list and a nil error -- the caller cannot tell a
	// partial result from a complete one.
	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("failed to iterate checkpoints: %w", err)
	}

	return checkpoints, nil
}

// Delete deletes a checkpoint
func (p *PostgresCheckpointer) Delete(ctx context.Context, threadID, checkpointID string) error {
	query := `DELETE FROM checkpoints WHERE thread_id = $1 AND id = $2`

	res, err := p.conn.Exec(ctx, query, threadID, checkpointID)
	if err != nil {
		return fmt.Errorf("failed to delete checkpoint: %w", err)
	}

	// Deleting a checkpoint that is not there used to report success, so a
	// typo'd or already-collected ID looked like a completed deletion. The
	// memory and file checkpointers both return an error here; matching them
	// keeps the Checkpointer contract the same across backends.
	affected, err := res.RowsAffected()
	if err != nil {
		return fmt.Errorf("failed to confirm checkpoint deletion: %w", err)
	}
	if affected == 0 {
		return fmt.Errorf("checkpoint %s not found in thread %s", checkpointID, threadID)
	}

	return nil
}

// Close closes the PostgreSQL checkpointer
func (p *PostgresCheckpointer) Close() error {
	return p.conn.Close()
}

// RAG-specific methods

// SaveDocument saves a document for RAG
func (p *PostgresCheckpointer) SaveDocument(ctx context.Context, doc *Document) error {
	if !p.config.EnableRAG {
		return fmt.Errorf("RAG is not enabled")
	}

	if doc == nil {
		return fmt.Errorf("cannot save a nil document")
	}

	// doc.Metadata used to be handed to database/sql as a bare map, which the
	// driver rejects with "unsupported type map[string]interface {}, a map".
	// SaveDocument therefore failed 100% of the time; it had never been run.
	metadataData, err := json.Marshal(doc.Metadata)
	if err != nil {
		return fmt.Errorf("failed to marshal document metadata: %w", err)
	}

	var query string
	var args []interface{}

	if p.config.Type == DatabaseTypePgVector && doc.Embedding != nil {
		query = `
			INSERT INTO documents (id, thread_id, content, metadata, embedding, created_at, updated_at)
			VALUES ($1, $2, $3, $4, $5, $6, $7)
			ON CONFLICT (id) DO UPDATE SET
				content = EXCLUDED.content,
				metadata = EXCLUDED.metadata,
				embedding = EXCLUDED.embedding,
				updated_at = EXCLUDED.updated_at
		`
		// Likewise []float64 is not a driver value; pgvector accepts its text
		// literal form and casts it to the column type.
		args = []interface{}{doc.ID, doc.ThreadID, doc.Content, metadataData, encodeVector(doc.Embedding), doc.CreatedAt, doc.UpdatedAt}
	} else {
		query = `
			INSERT INTO documents (id, thread_id, content, metadata, created_at, updated_at)
			VALUES ($1, $2, $3, $4, $5, $6)
			ON CONFLICT (id) DO UPDATE SET
				content = EXCLUDED.content,
				metadata = EXCLUDED.metadata,
				updated_at = EXCLUDED.updated_at
		`
		args = []interface{}{doc.ID, doc.ThreadID, doc.Content, metadataData, doc.CreatedAt, doc.UpdatedAt}
	}

	// documents.thread_id references threads(id); register the parent for the
	// same reason Save does, so a document can be stored for a thread that has
	// not been created out-of-band.
	return p.conn.WithTx(ctx, func(tx *sql.Tx) error {
		if doc.ThreadID != "" {
			if _, txErr := tx.ExecContext(ctx,
				`INSERT INTO threads (id, created_at, updated_at) VALUES ($1, NOW(), NOW()) ON CONFLICT (id) DO NOTHING`,
				doc.ThreadID); txErr != nil {
				return fmt.Errorf("failed to register thread: %w", txErr)
			}
		}
		if _, txErr := tx.ExecContext(ctx, query, args...); txErr != nil {
			return fmt.Errorf("failed to save document: %w", txErr)
		}
		return nil
	})
}

// SearchDocuments performs similarity search on documents
func (p *PostgresCheckpointer) SearchDocuments(ctx context.Context, threadID string, queryEmbedding []float64, limit int) ([]*Document, error) {
	if !p.config.EnableRAG {
		return nil, fmt.Errorf("RAG is not enabled")
	}

	var query string
	var args []interface{}

	if p.config.Type == DatabaseTypePgVector && queryEmbedding != nil {
		query = `
			SELECT id, thread_id, content, metadata, embedding, created_at, updated_at
			FROM documents
			WHERE thread_id = $1
			ORDER BY embedding <-> $2::vector
			LIMIT $3
		`
		// The raw []float64 the caller passes is not a valid driver value, so
		// every vector similarity search failed with "unsupported type
		// []float64, a slice of float64". Send the pgvector text literal and
		// cast it server-side.
		args = []interface{}{threadID, encodeVector(queryEmbedding), limit}
	} else {
		// Fallback to text search
		query = `
			SELECT id, thread_id, content, metadata, created_at, updated_at
			FROM documents
			WHERE thread_id = $1
			ORDER BY created_at DESC
			LIMIT $2
		`
		args = []interface{}{threadID, limit}
	}

	raw, err := p.conn.QueryRows(ctx, query, args...)
	if err != nil {
		return nil, fmt.Errorf("failed to search documents: %w", err)
	}
	rows, err := asSQLRows(raw)
	if err != nil {
		return nil, err
	}
	defer func() { _ = rows.Close() }()

	documents := []*Document{}
	for rows.Next() {
		var doc Document
		var metadataData []byte
		// thread_id is nullable on documents.
		var docThreadID sql.NullString

		if p.config.Type == DatabaseTypePgVector {
			var embedding interface{}
			err := rows.Scan(&doc.ID, &docThreadID, &doc.Content, &metadataData, &embedding, &doc.CreatedAt, &doc.UpdatedAt)
			if err != nil {
				return nil, fmt.Errorf("failed to scan document: %w", err)
			}
			// The stored embedding used to be scanned and then dropped on the
			// floor behind a "handle conversion if needed" comment, so every
			// document read back had a nil Embedding regardless of what was in
			// the column. Decode it properly.
			doc.Embedding, err = decodeVector(embedding)
			if err != nil {
				return nil, fmt.Errorf("failed to decode embedding for document %s: %w", doc.ID, err)
			}
		} else {
			err := rows.Scan(&doc.ID, &docThreadID, &doc.Content, &metadataData, &doc.CreatedAt, &doc.UpdatedAt)
			if err != nil {
				return nil, fmt.Errorf("failed to scan document: %w", err)
			}
		}

		doc.ThreadID = docThreadID.String

		// Unmarshal metadata
		if err := decodeJSONMap(metadataData, &doc.Metadata); err != nil {
			return nil, fmt.Errorf("failed to unmarshal metadata: %w", err)
		}

		documents = append(documents, &doc)
	}

	// See List: an unchecked rows.Err() turns a mid-iteration failure into a
	// silently short result set.
	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("failed to iterate documents: %w", err)
	}

	return documents, nil
}

// RedisCheckpointer implements Redis-based checkpointing
type RedisCheckpointer struct {
	client *redis.Client
	config *DatabaseConfig
	logger *logrus.Logger
	ttl    time.Duration
}

// redisKeySegment escapes an identifier for safe use inside a colon-delimited
// Redis key.
//
// Keys were built with a plain fmt.Sprintf("checkpoint:%s:%s", threadID, id),
// so any identifier containing a colon made distinct checkpoints collide:
// thread "a:b" + checkpoint "c" and thread "a" + checkpoint "b:c" both produced
// "checkpoint:a:b:c". One thread then read and overwrote another thread's
// state. Thread IDs are routinely derived from user or session identifiers, so
// this was a cross-tenant data leak, not just an oddity.
//
// Escaping only ':' and the escape character itself leaves keys byte-identical
// for the ordinary identifiers that contain neither, so existing data stays
// readable.
func redisKeySegment(s string) string {
	if !strings.ContainsAny(s, ":%") {
		return s
	}
	var b strings.Builder
	for _, r := range s {
		switch r {
		case ':':
			b.WriteString("%3A")
		case '%':
			b.WriteString("%25")
		default:
			b.WriteRune(r)
		}
	}
	return b.String()
}

func redisCheckpointKey(threadID, checkpointID string) string {
	return fmt.Sprintf("checkpoint:%s:%s", redisKeySegment(threadID), redisKeySegment(checkpointID))
}

func redisThreadIndexKey(threadID string) string {
	return fmt.Sprintf("thread:%s:checkpoints", redisKeySegment(threadID))
}

// NewRedisCheckpointer creates a new Redis checkpointer
func NewRedisCheckpointer(config *DatabaseConfig) (*RedisCheckpointer, error) {
	client := redis.NewClient(&redis.Options{
		Addr:     fmt.Sprintf("%s:%d", config.Host, config.Port),
		Password: config.Password,
		DB:       0,
	})

	// Test connection
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	if err := client.Ping(ctx).Err(); err != nil {
		// The client owns a connection pool and background goroutines; the
		// failure path used to drop it without closing, leaking both on every
		// unsuccessful connection attempt.
		_ = client.Close()
		return nil, fmt.Errorf("failed to connect to Redis: %w", err)
	}

	// Checkpoints expire after this long. The value was previously hard-coded
	// with no way to change it, so every deployment silently lost its
	// checkpoints after 24 hours.
	ttl := 24 * time.Hour
	if config.CheckpointTTL != "" {
		parsed, err := time.ParseDuration(config.CheckpointTTL)
		if err != nil {
			_ = client.Close()
			return nil, fmt.Errorf("invalid checkpoint_ttl %q: %w", config.CheckpointTTL, err)
		}
		ttl = parsed
	}

	return &RedisCheckpointer{
		client: client,
		config: config,
		logger: logrus.New(),
		ttl:    ttl,
	}, nil
}

// Save saves a checkpoint to Redis
func (r *RedisCheckpointer) Save(ctx context.Context, checkpoint *Checkpoint) error {
	if checkpoint == nil {
		return fmt.Errorf("cannot save a nil checkpoint")
	}

	data, err := json.Marshal(checkpoint)
	if err != nil {
		return fmt.Errorf("failed to marshal checkpoint: %w", err)
	}

	key := redisCheckpointKey(checkpoint.ThreadID, checkpoint.ID)

	if err := r.client.Set(ctx, key, data, r.ttl).Err(); err != nil {
		return fmt.Errorf("failed to save checkpoint to Redis: %w", err)
	}

	// Add to thread index
	threadKey := redisThreadIndexKey(checkpoint.ThreadID)
	if err := r.client.SAdd(ctx, threadKey, checkpoint.ID).Err(); err != nil {
		return fmt.Errorf("failed to add checkpoint to thread index: %w", err)
	}

	// The index set was created without an expiry while the checkpoints it
	// points at expire, so it accumulated dead member IDs forever -- an
	// unbounded leak that also made List do a wasted round trip per dead entry.
	// Refreshing it alongside the newest checkpoint keeps the two in step.
	if r.ttl > 0 {
		if err := r.client.Expire(ctx, threadKey, r.ttl).Err(); err != nil {
			return fmt.Errorf("failed to set thread index expiry: %w", err)
		}
	}

	return nil
}

// Load loads a checkpoint from Redis
func (r *RedisCheckpointer) Load(ctx context.Context, threadID, checkpointID string) (*Checkpoint, error) {
	key := redisCheckpointKey(threadID, checkpointID)

	data, err := r.client.Get(ctx, key).Result()
	if err != nil {
		if err == redis.Nil {
			return nil, fmt.Errorf("checkpoint %s not found in thread %s", checkpointID, threadID)
		}
		return nil, fmt.Errorf("failed to load checkpoint from Redis: %w", err)
	}

	var checkpoint Checkpoint
	if err := json.Unmarshal([]byte(data), &checkpoint); err != nil {
		return nil, fmt.Errorf("failed to unmarshal checkpoint: %w", err)
	}

	// Defense in depth against key aliasing: the stored payload records which
	// thread it belongs to, so refuse to hand a caller another thread's state
	// even if some future key scheme lets two identifiers map to one key.
	if checkpoint.ThreadID != "" && checkpoint.ThreadID != threadID {
		return nil, fmt.Errorf("checkpoint %s belongs to thread %s, not %s", checkpointID, checkpoint.ThreadID, threadID)
	}

	return &checkpoint, nil
}

// List lists checkpoints for a thread
func (r *RedisCheckpointer) List(ctx context.Context, threadID string) ([]*CheckpointMetadata, error) {
	threadKey := redisThreadIndexKey(threadID)

	checkpointIDs, err := r.client.SMembers(ctx, threadKey).Result()
	if err != nil {
		return nil, fmt.Errorf("failed to get checkpoint IDs: %w", err)
	}

	metadata := []*CheckpointMetadata{}
	for _, checkpointID := range checkpointIDs {
		checkpoint, err := r.Load(ctx, threadID, checkpointID)
		if err != nil {
			// An index entry whose checkpoint has expired or been corrupted is
			// skipped rather than failing the whole listing, so one bad entry
			// cannot make a thread unresumable. It is logged because a silent
			// skip would hide real data loss.
			r.logger.Warnf("Failed to load checkpoint %s: %v", checkpointID, err)
			continue
		}

		meta := &CheckpointMetadata{
			ID:        checkpoint.ID,
			ThreadID:  checkpoint.ThreadID,
			Metadata:  checkpoint.Metadata,
			CreatedAt: checkpoint.CreatedAt,
			NodeID:    checkpoint.NodeID,
			StepID:    checkpoint.StepID,
		}
		metadata = append(metadata, meta)
	}

	return metadata, nil
}

// Delete deletes a checkpoint
func (r *RedisCheckpointer) Delete(ctx context.Context, threadID, checkpointID string) error {
	key := redisCheckpointKey(threadID, checkpointID)

	removed, err := r.client.Del(ctx, key).Result()
	if err != nil {
		return fmt.Errorf("failed to delete checkpoint from Redis: %w", err)
	}

	// Remove from thread index. This runs even when the payload was already
	// gone so an expired checkpoint's index entry still gets cleaned up.
	threadKey := redisThreadIndexKey(threadID)
	if err := r.client.SRem(ctx, threadKey, checkpointID).Err(); err != nil {
		return fmt.Errorf("failed to remove checkpoint from thread index: %w", err)
	}

	// Matches the memory and file checkpointers, which both report a missing
	// checkpoint rather than pretending the delete succeeded.
	if removed == 0 {
		return fmt.Errorf("checkpoint %s not found in thread %s", checkpointID, threadID)
	}

	return nil
}

// Close closes the Redis checkpointer
func (r *RedisCheckpointer) Close() error {
	return r.client.Close()
}

// Document represents a document for RAG
type Document struct {
	ID        string                 `json:"id"`
	ThreadID  string                 `json:"thread_id"`
	Content   string                 `json:"content"`
	Metadata  map[string]interface{} `json:"metadata"`
	Embedding []float64              `json:"embedding,omitempty"`
	CreatedAt time.Time              `json:"created_at"`
	UpdatedAt time.Time              `json:"updated_at"`
}

// SessionManager manages user sessions and threads
type SessionManager struct {
	conn   DatabaseConnection
	logger *logrus.Logger
}

// Session represents a user session
type Session struct {
	ID        string                 `json:"id"`
	ThreadID  string                 `json:"thread_id"`
	UserID    string                 `json:"user_id"`
	Metadata  map[string]interface{} `json:"metadata"`
	CreatedAt time.Time              `json:"created_at"`
	ExpiresAt *time.Time             `json:"expires_at,omitempty"`
}

// Thread represents a conversation thread
type Thread struct {
	ID        string                 `json:"id"`
	Name      string                 `json:"name"`
	Metadata  map[string]interface{} `json:"metadata"`
	CreatedAt time.Time              `json:"created_at"`
	UpdatedAt time.Time              `json:"updated_at"`
}

// NewSessionManager creates a new session manager
func NewSessionManager(conn DatabaseConnection) *SessionManager {
	return &SessionManager{
		conn:   conn,
		logger: logrus.New(),
	}
}

// CreateSession creates a new session
func (sm *SessionManager) CreateSession(ctx context.Context, session *Session) error {
	if sm == nil || sm.conn == nil {
		return ErrSessionStoreUnavailable
	}
	metadataData, err := json.Marshal(session.Metadata)
	if err != nil {
		return fmt.Errorf("failed to marshal metadata: %w", err)
	}

	query := `
		INSERT INTO sessions (id, thread_id, user_id, metadata, created_at, expires_at)
		VALUES ($1, $2, $3, $4, $5, $6)
	`

	return sm.conn.ExecuteQuery(ctx, query,
		session.ID,
		session.ThreadID,
		session.UserID,
		metadataData,
		session.CreatedAt,
		session.ExpiresAt,
	)
}

// GetSession retrieves a session
func (sm *SessionManager) GetSession(ctx context.Context, sessionID string) (*Session, error) {
	query := `
		SELECT id, thread_id, user_id, metadata, created_at, expires_at
		FROM sessions
		WHERE id = $1
	`

	if sm == nil || sm.conn == nil {
		return nil, ErrSessionStoreUnavailable
	}
	row, err := asSQLRow(sm.conn.QueryRow(ctx, query, sessionID))
	if err != nil {
		return nil, err
	}

	var session Session
	var metadataData []byte
	// user_id is nullable; scanning NULL straight into a string fails.
	var userID sql.NullString

	err = row.Scan(
		&session.ID,
		&session.ThreadID,
		&userID,
		&metadataData,
		&session.CreatedAt,
		&session.ExpiresAt,
	)

	if err != nil {
		if err == sql.ErrNoRows {
			return nil, fmt.Errorf("session %s not found", sessionID)
		}
		return nil, fmt.Errorf("failed to get session: %w", err)
	}

	session.UserID = userID.String

	// Unmarshal metadata
	if err := decodeJSONMap(metadataData, &session.Metadata); err != nil {
		return nil, fmt.Errorf("failed to unmarshal metadata: %w", err)
	}

	return &session, nil
}

// CreateThread creates a new thread
func (sm *SessionManager) CreateThread(ctx context.Context, thread *Thread) error {
	if sm == nil || sm.conn == nil {
		return ErrSessionStoreUnavailable
	}
	metadataData, err := json.Marshal(thread.Metadata)
	if err != nil {
		return fmt.Errorf("failed to marshal metadata: %w", err)
	}

	query := `
		INSERT INTO threads (id, name, metadata, created_at, updated_at)
		VALUES ($1, $2, $3, $4, $5)
	`

	return sm.conn.ExecuteQuery(ctx, query,
		thread.ID,
		thread.Name,
		metadataData,
		thread.CreatedAt,
		thread.UpdatedAt,
	)
}

// GetThread retrieves a thread
func (sm *SessionManager) GetThread(ctx context.Context, threadID string) (*Thread, error) {
	query := `
		SELECT id, name, metadata, created_at, updated_at
		FROM threads
		WHERE id = $1
	`

	if sm == nil || sm.conn == nil {
		return nil, ErrSessionStoreUnavailable
	}
	row, err := asSQLRow(sm.conn.QueryRow(ctx, query, threadID))
	if err != nil {
		return nil, err
	}

	var thread Thread
	var metadataData []byte
	// name is nullable, and threads created implicitly by Save have no name at
	// all -- scanning that NULL into a string failed with "converting NULL to
	// string is unsupported", making every auto-registered thread unreadable.
	var name sql.NullString

	err = row.Scan(
		&thread.ID,
		&name,
		&metadataData,
		&thread.CreatedAt,
		&thread.UpdatedAt,
	)

	if err != nil {
		if err == sql.ErrNoRows {
			return nil, fmt.Errorf("thread %s not found", threadID)
		}
		return nil, fmt.Errorf("failed to get thread: %w", err)
	}

	thread.Name = name.String

	// Unmarshal metadata
	if err := decodeJSONMap(metadataData, &thread.Metadata); err != nil {
		return nil, fmt.Errorf("failed to unmarshal metadata: %w", err)
	}

	return &thread, nil
}

// DatabaseConnectionManager manages multiple database connections
type DatabaseConnectionManager struct {
	// mu guards connections. Without it, concurrent AddConnection/GetConnection
	// calls are a concurrent map read and write, which the Go runtime turns
	// into an unrecoverable process crash rather than a recoverable error.
	mu          sync.RWMutex
	connections map[string]DatabaseConnection
	logger      *logrus.Logger
}

// NewDatabaseConnectionManager creates a new connection manager
func NewDatabaseConnectionManager() *DatabaseConnectionManager {
	return &DatabaseConnectionManager{
		connections: make(map[string]DatabaseConnection),
		logger:      logrus.New(),
	}
}

// AddConnection adds a database connection
func (dcm *DatabaseConnectionManager) AddConnection(name string, config *DatabaseConfig) error {
	var conn DatabaseConnection
	var err error

	switch config.Type {
	case DatabaseTypePostgres, DatabaseTypePostgresQL, DatabaseTypePgVector:
		conn, err = NewPostgresConnection(config)
	case DatabaseTypeRedis:
		// Redis connection would be implemented here
		return fmt.Errorf("redis connection is not implemented in this version")
	case DatabaseTypeOpenSearch, DatabaseTypeElastic:
		// OpenSearch/Elasticsearch connections would be implemented here
		return fmt.Errorf("openSearch/Elasticsearch connection is not implemented in this version")
	case DatabaseTypeMongoDB:
		// MongoDB connection would be implemented here
		return fmt.Errorf("MongoDB connection not implemented in this version")
	case DatabaseTypeMySQL:
		// MySQL connection would be implemented here
		return fmt.Errorf("MySQL connection not implemented in this version")
	case DatabaseTypeSQLite:
		// SQLite connection would be implemented here
		return fmt.Errorf("SQLite connection not implemented in this version")
	default:
		return fmt.Errorf("unsupported database type: %s", config.Type)
	}

	if err != nil {
		return fmt.Errorf("failed to create connection for %s: %w", name, err)
	}

	dcm.mu.Lock()
	if dcm.connections == nil {
		dcm.connections = make(map[string]DatabaseConnection)
	}
	// Reusing a name used to overwrite the entry and leak the previous pool,
	// which stayed open with no remaining reference for CloseAll to find.
	previous, replaced := dcm.connections[name]
	dcm.connections[name] = conn
	dcm.mu.Unlock()

	if replaced && previous != nil {
		if cerr := previous.Close(); cerr != nil {
			dcm.logger.Warnf("Failed to close replaced connection %s: %v", name, cerr)
		}
	}

	dcm.logger.Infof("Added database connection: %s (%s)", name, config.Type)
	return nil
}

// GetConnection retrieves a database connection
func (dcm *DatabaseConnectionManager) GetConnection(name string) (DatabaseConnection, error) {
	dcm.mu.RLock()
	conn, exists := dcm.connections[name]
	dcm.mu.RUnlock()

	if !exists {
		return nil, fmt.Errorf("connection %s not found", name)
	}
	return conn, nil
}

// CloseAll closes all database connections and forgets them, so a second call
// cannot double-close a pool.
func (dcm *DatabaseConnectionManager) CloseAll() error {
	dcm.mu.Lock()
	conns := dcm.connections
	dcm.connections = make(map[string]DatabaseConnection)
	dcm.mu.Unlock()

	var errors []string
	for name, conn := range conns {
		if err := conn.Close(); err != nil {
			errors = append(errors, fmt.Sprintf("failed to close %s: %v", name, err))
		}
	}

	if len(errors) > 0 {
		return fmt.Errorf("errors closing connections: %s", strings.Join(errors, "; "))
	}

	return nil
}

// CreateCheckpointer creates a checkpointer for the specified database
func CreateCheckpointer(config *DatabaseConfig) (Checkpointer, error) {
	switch config.Type {
	case DatabaseTypePostgres, DatabaseTypePostgresQL, DatabaseTypePgVector:
		return NewPostgresCheckpointer(config)
	case DatabaseTypeRedis:
		return NewRedisCheckpointer(config)
	case DatabaseTypeOpenSearch, DatabaseTypeElastic:
		// OpenSearch/Elasticsearch checkpointer would be implemented here
		return nil, fmt.Errorf("OpenSearch/Elasticsearch checkpointer not implemented in this version")
	case DatabaseTypeMongoDB:
		// MongoDB checkpointer would be implemented here
		return nil, fmt.Errorf("MongoDB checkpointer not implemented in this version")
	case DatabaseTypeMySQL:
		// MySQL checkpointer would be implemented here
		return nil, fmt.Errorf("MySQL checkpointer not implemented in this version")
	case DatabaseTypeSQLite:
		// SQLite checkpointer would be implemented here
		return nil, fmt.Errorf("SQLite checkpointer not implemented in this version")
	default:
		return nil, fmt.Errorf("unsupported database type for checkpointer: %s", config.Type)
	}
}

// Helper function to create a default PostgreSQL configuration
func NewPostgresConfig(host string, port int, database, username, password string) *DatabaseConfig {
	return &DatabaseConfig{
		Type:         DatabaseTypePostgres,
		Host:         host,
		Port:         port,
		Database:     database,
		Username:     username,
		Password:     password,
		SSLMode:      "disable",
		MaxOpenConns: 25,
		MaxIdleConns: 5,
		MaxLifetime:  "5m",
	}
}

// Helper function to create a PostgreSQL with pgvector configuration
func NewPgVectorConfig(host string, port int, database, username, password string, vectorDim int) *DatabaseConfig {
	return &DatabaseConfig{
		Type:            DatabaseTypePgVector,
		Host:            host,
		Port:            port,
		Database:        database,
		Username:        username,
		Password:        password,
		SSLMode:         "disable",
		MaxOpenConns:    25,
		MaxIdleConns:    5,
		MaxLifetime:     "5m",
		VectorDimension: vectorDim,
		VectorMetric:    "cosine",
		EnableRAG:       true,
	}
}

// Helper function to create a Redis configuration
func NewRedisConfig(host string, port int, password string) *DatabaseConfig {
	return &DatabaseConfig{
		Type:     DatabaseTypeRedis,
		Host:     host,
		Port:     port,
		Password: password,
	}
}
