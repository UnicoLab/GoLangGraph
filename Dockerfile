# Build stage
FROM golang:1.25.13-alpine AS builder

# Set working directory
WORKDIR /app

# Install git and ca-certificates (needed for fetching dependencies)
RUN apk add --no-cache git ca-certificates tzdata

# Copy go mod files
COPY go.mod go.sum ./

# Download dependencies
RUN go mod download

# Copy source code
COPY . .

# Build the binary
ARG VERSION=dev
RUN CGO_ENABLED=0 GOOS=linux go build \
    -ldflags="-w -s -X main.version=${VERSION}" \
    -a -installsuffix cgo \
    -o golanggraph \
    ./cmd/golanggraph

# Final stage
FROM alpine:3.20.3

# Install ca-certificates for HTTPS requests and create non-root user
RUN apk --no-cache add ca-certificates tzdata && \
    addgroup -g 1001 -S golanggraph && \
    adduser -u 1001 -S golanggraph -G golanggraph

WORKDIR /app

# Copy the binary from builder stage
COPY --from=builder /app/golanggraph .

# Create directories for optional files and change ownership to non-root user
RUN mkdir -p ./configs ./docs && \
    chown -R golanggraph:golanggraph /app

# Switch to non-root user
USER golanggraph

# Expose port (adjust as needed)
EXPOSE 8080

# Health check.
#
# Probes the server's own endpoint rather than running a local dependency
# scan: the container is healthy when it is serving. A plain "health" run
# reports on dependencies, which is a different question and would mark a
# perfectly serving container unhealthy whenever an optional one is absent.
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD ["./golanggraph", "health", "--server", "http://127.0.0.1:8080"]

# Run the binary
ENTRYPOINT ["./golanggraph"]
CMD ["serve"]
