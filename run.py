from state_types import GraphState
from agents import router_node, planner_node, rag_retriever_node, executor_node, formatter_node, supervisor_node

NODE_ORDER = [
    router_node,
    planner_node,
    rag_retriever_node,
    executor_node,
    formatter_node,
    supervisor_node
]

def run_graph(initial_query: str= "how to print hello world in python?", user_id: str = None) -> GraphState:
    state = GraphState(query=initial_query, user_id=user_id)
    for node in NODE_ORDER:
        state = node(state)
        if state.final_answer and "forbidden" in state.final_answer:
            break
    return state

if __name__ == "__main__":
    q = input("Input query: ").strip()
    s = run_graph(q)
    print("\n--- Final answer ---\n")
    print(s.final_answer)
    print("\n--- Agents activated ---\n", s.agents_activated)
    print("\n--- Session history entries ---", len(s.session_history))