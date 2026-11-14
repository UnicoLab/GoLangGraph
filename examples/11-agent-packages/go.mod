module github.com/UnicoLab/GoLangGraph/examples/11-agent-packages

go 1.23.0

require (
	github.com/UnicoLab/GoLangGraph v0.0.0
	github.com/UnicoLab/GoLangGraph/agents/designer v0.1.0
	github.com/UnicoLab/GoLangGraph/agents/habitat v0.1.0
	github.com/UnicoLab/GoLangGraph/agents/storymaker v0.1.0
)

require (
	github.com/cespare/xxhash/v2 v2.3.0 // indirect
	github.com/dgryski/go-rendezvous v0.0.0-20200823014737-9f7001d12a5f // indirect
	github.com/dlclark/regexp2 v1.10.0 // indirect
	github.com/go-redis/redis/v8 v8.11.5 // indirect
	github.com/google/uuid v1.6.0 // indirect
	github.com/lib/pq v1.10.9 // indirect
	github.com/pkoukk/tiktoken-go v0.1.8 // indirect
	github.com/pkoukk/tiktoken-go-loader v0.0.2 // indirect
	github.com/sashabaranov/go-openai v1.40.5 // indirect
	github.com/sirupsen/logrus v1.9.3 // indirect
	golang.org/x/sys v0.33.0 // indirect
)

// Replaces are deliberately in the consumer: dependency replaces are not inherited.
replace github.com/UnicoLab/GoLangGraph => ../..

replace github.com/UnicoLab/GoLangGraph/agents/designer => ../../agents/designer

replace github.com/UnicoLab/GoLangGraph/agents/habitat => ../../agents/habitat

replace github.com/UnicoLab/GoLangGraph/agents/storymaker => ../../agents/storymaker
