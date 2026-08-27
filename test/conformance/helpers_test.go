package conformance

import (
	"encoding/json"
	"os"
	"testing"

	"github.com/stretchr/testify/require"
)

// corrupt overwrites a file with arbitrary content to simulate storage damage.
func corrupt(t *testing.T, path, content string) {
	t.Helper()
	require.NoError(t, os.WriteFile(path, []byte(content), 0o600))
}

// jsonMarshal is a tiny indirection so serialisation assertions read clearly.
func jsonMarshal(v interface{}) ([]byte, error) { return json.Marshal(v) }
