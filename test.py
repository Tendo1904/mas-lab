from state_types import GraphState
s = GraphState(query="Напиши функцию на Python для вставки docstring-а")
print(s.model_dump_json(indent=2))

try:
    GraphState(query=" ")
except Exception as e:
    print("Validation error:", e)