// Copyright (c) 2024 GoLangGraph Team
//
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

package e2e

import (
	"bufio"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// repoRoot locates the module root from the test's working directory.
func repoRoot(t *testing.T) string {
	t.Helper()
	dir, err := os.Getwd()
	require.NoError(t, err)
	for i := 0; i < 6; i++ {
		if _, err := os.Stat(filepath.Join(dir, "go.mod")); err == nil {
			return dir
		}
		dir = filepath.Dir(dir)
	}
	t.Fatal("could not locate the module root")
	return ""
}

// dockerfileInstruction is one parsed line of a Dockerfile.
type dockerfileInstruction struct {
	Verb string
	Args []string
	Line int
}

// parseDockerfile reads a Dockerfile, joining continuation lines.
func parseDockerfile(t *testing.T, path string) []dockerfileInstruction {
	t.Helper()

	file, err := os.Open(path) // #nosec G304 -- path is built from the repository root
	require.NoError(t, err)
	defer func() { _ = file.Close() }()

	var instructions []dockerfileInstruction
	scanner := bufio.NewScanner(file)
	scanner.Buffer(make([]byte, 0, 64*1024), 1024*1024)

	var pending string
	lineNo, startLine := 0, 0
	for scanner.Scan() {
		lineNo++
		line := strings.TrimSpace(scanner.Text())
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		if pending == "" {
			startLine = lineNo
		}
		if strings.HasSuffix(line, `\`) {
			pending += strings.TrimSuffix(line, `\`) + " "
			continue
		}
		full := pending + line
		pending = ""

		fields := strings.Fields(full)
		if len(fields) == 0 {
			continue
		}
		instructions = append(instructions, dockerfileInstruction{
			Verb: strings.ToUpper(fields[0]),
			Args: fields[1:],
			Line: startLine,
		})
	}
	require.NoError(t, scanner.Err())
	return instructions
}

// dockerfiles returns the images built from the repository root context.
func dockerfiles(t *testing.T) []string {
	t.Helper()
	root := repoRoot(t)
	var found []string
	for _, name := range []string{"Dockerfile", "Dockerfile.agent"} {
		path := filepath.Join(root, name)
		if _, err := os.Stat(path); err == nil {
			found = append(found, path)
		}
	}
	require.NotEmpty(t, found, "no Dockerfiles found")
	return found
}

// A COPY from the build context must reference a path that exists, or the
// image cannot be built at all. This is checked statically because no
// container runtime is available in every environment the tests run in.
func TestDocker_CopySourcesExist(t *testing.T) {
	root := repoRoot(t)

	for _, path := range dockerfiles(t) {
		t.Run(filepath.Base(path), func(t *testing.T) {
			for _, ins := range parseDockerfile(t, path) {
				if ins.Verb != "COPY" && ins.Verb != "ADD" {
					continue
				}

				args := ins.Args
				fromStage := false
				filtered := args[:0]
				for _, a := range args {
					if strings.HasPrefix(a, "--from=") {
						fromStage = true
						continue
					}
					if strings.HasPrefix(a, "--") {
						continue
					}
					filtered = append(filtered, a)
				}
				if fromStage || len(filtered) < 2 {
					// Copies from an earlier build stage are not context paths.
					continue
				}

				for _, src := range filtered[:len(filtered)-1] {
					src = strings.Trim(src, `"`)
					if strings.ContainsAny(src, "*?[") {
						continue // globs are resolved by the builder
					}
					full := filepath.Join(root, filepath.Clean("/"+src))
					_, err := os.Stat(full)
					assert.NoError(t, err,
						"%s:%d copies %q from the build context, but it does not exist",
						filepath.Base(path), ins.Line, src)
				}
			}
		})
	}
}

// Images must not run as root.
func TestDocker_RunsAsNonRoot(t *testing.T) {
	for _, path := range dockerfiles(t) {
		t.Run(filepath.Base(path), func(t *testing.T) {
			var lastUser string
			for _, ins := range parseDockerfile(t, path) {
				if ins.Verb == "USER" && len(ins.Args) > 0 {
					lastUser = ins.Args[0]
				}
			}
			require.NotEmpty(t, lastUser, "no USER instruction: the image would run as root")
			assert.NotEqual(t, "root", lastUser)
			assert.NotEqual(t, "0", lastUser)
		})
	}
}

// The health check must probe the server rather than scanning local
// dependencies: a serving container is healthy even when an optional
// dependency is absent, and a dependency scan would restart it forever.
func TestDocker_HealthcheckProbesTheServer(t *testing.T) {
	for _, path := range dockerfiles(t) {
		t.Run(filepath.Base(path), func(t *testing.T) {
			var healthcheck string
			for _, ins := range parseDockerfile(t, path) {
				if ins.Verb == "HEALTHCHECK" {
					healthcheck = strings.Join(ins.Args, " ")
				}
			}
			require.NotEmpty(t, healthcheck, "image has no HEALTHCHECK")
			assert.Contains(t, healthcheck, "--server",
				"the health check must probe the serving endpoint")
		})
	}
}

// Every image must define an entrypoint and expose the serving port.
func TestDocker_EntrypointAndPort(t *testing.T) {
	for _, path := range dockerfiles(t) {
		t.Run(filepath.Base(path), func(t *testing.T) {
			verbs := map[string]bool{}
			for _, ins := range parseDockerfile(t, path) {
				verbs[ins.Verb] = true
			}
			assert.True(t, verbs["ENTRYPOINT"] || verbs["CMD"], "image has no entrypoint")
			assert.True(t, verbs["EXPOSE"], "image does not expose a port")
		})
	}
}

// The binary the image builds and the one it runs must be the same, or the
// container starts and immediately fails.
func TestDocker_BuiltBinaryMatchesEntrypoint(t *testing.T) {
	for _, path := range dockerfiles(t) {
		t.Run(filepath.Base(path), func(t *testing.T) {
			var built, entry string
			for _, ins := range parseDockerfile(t, path) {
				switch ins.Verb {
				case "RUN":
					joined := strings.Join(ins.Args, " ")
					if idx := strings.Index(joined, " -o "); idx >= 0 {
						rest := strings.Fields(joined[idx+4:])
						if len(rest) > 0 {
							built = rest[0]
						}
					}
				case "ENTRYPOINT":
					entry = strings.Trim(strings.Join(ins.Args, " "), `[]"`)
					entry = strings.TrimPrefix(strings.Trim(entry, `"`), "./")
				}
			}
			require.NotEmpty(t, built, "no build output found")
			require.NotEmpty(t, entry, "no ENTRYPOINT found")
			assert.Equal(t, built, entry,
				"the image builds %q but runs %q", built, entry)
		})
	}
}

// When a container runtime is available, actually build the image. Skipped
// otherwise so the suite stays runnable everywhere.
func TestDocker_ImageBuilds(t *testing.T) {
	if testing.Short() {
		t.Skip("short mode")
	}
	if _, err := exec.LookPath("docker"); err != nil {
		t.Skip("docker binary not available")
	}
	if err := exec.Command("docker", "info").Run(); err != nil {
		t.Skip("docker daemon not available in this environment")
	}

	root := repoRoot(t)
	for _, path := range dockerfiles(t) {
		name := filepath.Base(path)
		t.Run(name, func(t *testing.T) {
			cmd := exec.Command("docker", "build", "-f", path, "-t",
				"golanggraph-test:"+strings.ToLower(strings.ReplaceAll(name, ".", "-")), root)
			out, err := cmd.CombinedOutput()
			require.NoError(t, err, "docker build failed:\n%s", string(out))
		})
	}
}
