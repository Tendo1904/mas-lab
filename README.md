# **Multi-Agentic-System Lab**

## Goal
To design simplified MAS utilizing local or remote LLM via OpenAI comparable API. The implementation is founded as LangGraph with agent-nodes that read/update application state.

---

## Agents list

1. **Router**
   - Analyzing user's query (classificator) and decides upon its type: RAG, code/util, design principles or general request.
   - Routes the stream to the corresponding graph node.

2. **Classifier**
   - Deliberate classification: intent, required tools, urgency, safety flags, etc.
   - Can tag the query as human-in-loop required (if safety policies), or computational resources required.

3. **Planner**
   - Sets up a processing plan. E.g.: fetch context -> execute RAG -> compile response -> verify code.
   - The plan is stored as state and transfered to the executors. 