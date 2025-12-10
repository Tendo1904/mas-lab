import json
import traceback
from typing import Callable

from src.state_types import GraphState
from src.utils.utils import append_note

import src.agents as agents

# вызов функции-агента с перехватом ошибок
def safe_call_agent(fn: Callable, state: GraphState, ctx_name: str = None) -> GraphState:
    """
    Вызвать агент fn(state) безопасно: если упадёт — запишем ошибку в state.partial_answers.extra["errors"].
    """
    try:
        res = fn(state)
        if isinstance(res, GraphState):
            return res
        # если агент мутирует state и ничего не возвращает — предполагаем, что state изменён
        return state
    except Exception as e:
        # логируем стек в state.partial_answers.extra.errors
        err = {"agent": getattr(fn, "__name__", str(fn)), "error": str(e), "trace": traceback.format_exc()}
        state.partial_answers.extra.setdefault("errors", []).append(err)
        append_note(f"Agent error: {err['agent']}: {err['error']}", tags=["error"])
        return state

def run_graph_once(state: GraphState) -> GraphState:
    """
    Прогоним state по основному потоку: router -> planner -> executor -> supervisor
    """
    state = safe_call_agent(agents.router_node, state)
    state = safe_call_agent(agents.planner_node, state)
    state = safe_call_agent(agents.executor_node ,state)
    state = safe_call_agent(agents.supervisor_node, state)

    return state


# обёртка: принимает строку запроса, возвращает state
def run_query(query: str, user_id: str = None) -> GraphState:
    # валидируем создание GraphState (pydantic займётся валидностью)
    state = GraphState(query=query, user_id=user_id)
    state = run_graph_once(state)
    return state


# CLI
if __name__ == "__main__":
    print("Running mas_lab. Make sure OPENAI_API_BASE/OPENAI_API_KEY/MODEL_NAME are provided.\n")
    while True:
        try:
            q = input("\nInput your query (or 'exit'): ").strip()
            if not q or q.lower() in ("exit", "quit"):
                break

            st = run_query(q)

            print("\n--- Final answer ---\n")
            print(st.final_answer or "(empty)")

            print("\n--- Agents activated (in order of appearance) ---")
            print(", ".join(st.agents_activated))

            print("\n--- Session history (last entry) ---")
            if st.session_history:
                last = st.session_history[-1]
                print(f"Q: {last.question}\nA: {last.answer}\nTime: {last.timestamp}")
            else:
                print("No session entries recorded.")

            # Для отладки: сохранить state в файл с timestamp
            import time
            fname = f"state_log_{int(time.time())}.json"
            with open(fname, "w", encoding="utf-8") as f:
                json.dump(st.to_persistable(), f, ensure_ascii=False, indent=2)
            print(f"(state saved -> {fname})")

        except KeyboardInterrupt:
            break
        except Exception as e:
            print("Ошибка в основном цикле:", e)
            traceback.print_exc()
            break
