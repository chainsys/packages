# BI Agent App (Quarkus)

Quarkus service that wires **LangChain4j** (LLM calls) and **LangGraph4j** (agent flow orchestration) for BI querying.

## Flow
1. `SqlGeneratorNode` generates read-only SQL from natural language + schema context.
2. `SqlExecutorNode` runs SQL through JDBC.
3. `VizRecommenderNode` recommends chart type from result rows.

`BIAgentGraph` attempts to run through LangGraph4j (`StateGraph`) and falls back to deterministic sequential execution if the runtime API is unavailable.

## Endpoint
- `POST /api/bi-agent/query`
- body: `{ "question": "top 5 products by revenue" }`

## Required env vars
- `BI_AI_API_KEY`
- `BI_DB_URL`
- `BI_DB_USERNAME`
- `BI_DB_PASSWORD`

## Run locally
```bash
mvn quarkus:dev
```

## Build
```bash
mvn clean package
```
