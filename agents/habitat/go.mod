module github.com/UnicoLab/GoLangGraph/agents/habitat

go 1.23.0

require (
	github.com/UnicoLab/GoLangGraph v0.0.0
	github.com/UnicoLab/GoLangGraph/agents/designer v0.1.0
	github.com/UnicoLab/GoLangGraph/agents/storymaker v0.1.0
)

replace github.com/UnicoLab/GoLangGraph => ../..

replace github.com/UnicoLab/GoLangGraph/agents/designer => ../designer

replace github.com/UnicoLab/GoLangGraph/agents/storymaker => ../storymaker
