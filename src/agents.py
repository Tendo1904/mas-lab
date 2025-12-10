from langchain_openai.chat_models import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate
from src.state_types import GraphState, Classification, Plan
from src.utils.utils import keyword_search_notes, append_note
import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_BASE = os.environ.get("OPENAI_API_BASE")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
MODEL_NAME = os.environ.get("MODEL_NAME", "qwen")

def make_llm():
    return ChatOpenAI(
        base_url=OPENAI_API_BASE, 
        api_key=OPENAI_API_KEY, 
        model=MODEL_NAME, 
        temperature=0.2, 
        max_tokens=1024
    )

def router_node(state: GraphState) -> GraphState:
    state.add_agent("router")
    q = state.query.lower()
    if any(w in q for w in ["code", "implement", "function", "python", "javascript"]):
        state.classification = Classification(type="technical")
    elif any(w in q for w in ["explain", "what is", "define", "concept"]):
        state.classification = Classification(type="geek")
    else:
        state.classification = Classification(type="general")
    return state

def planner_node(state: GraphState) -> GraphState:
    state.add_agent("planner")
    llm = make_llm()

    sys_msg = """
You are a planner agent. 
Given a user query and its classification, produce an executable plan.

Plan must be a JSON with fields: steps (list of strings), tools (list of strings)
Steps may include:
- gather_context
- ask_technical_agent
- ask_geek_agent
- ask_general_agent
- format_answer
"""
    prompt = ChatPromptTemplate.from_messages([
        ("system", sys_msg),
        ("human", "User query: {query}\nClassification: {ctype}")
    ])
    messages = prompt.format_messages(
        query=state.query,
        ctype=state.classification.type
    )

    result = llm.invoke(messages)
    content = result.content

    import json
    try:
        plan_json = json.loads(content)
    except:
        plan_json = {"steps": ["ask_general_agent", "format_answer"], "tools": []}
    
    state.plan = Plan(**plan_json)
    return state

def rag_retriever_node(state: GraphState) -> GraphState:
    state.add_agent("rag_retriever")
    notes = keyword_search_notes(state.query)
    if notes:
        state.partial_answers.rag_context = "\n\n".join([n.text for n in notes])
    return state

def technical_agent(state: GraphState) -> GraphState:
    state.add_agent("technical_agent")
    llm = make_llm()

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a senior software engineer."),
        ("human", "Query: {query}\n\nContext: {ctx}")
    ])

    msg = prompt.format_messages(
        query=state.query,
        ctx=state.partial_answers.rag_context or ""
    )

    result = llm.invoke(msg)
    state.partial_answers.extra["tech"] = result.content
    return state

def geek_agent(state: GraphState) -> GraphState:
    state.add_agent("geek_agent")
    llm = make_llm()

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a science & media specialist."),
        ("human", "Explain this: {query}\n\nRelated notes: {ctx}")
    ])

    msg = prompt.format_messages(
        query=state.query,
        ctx=state.partial_answers.rag_context or ""
    )

    result = llm.invoke(msg)
    state.partial_answers.extra["geek"] = result.content
    return state

def general_agent(state: GraphState) -> GraphState:
    state.add_agent("general_agent")
    llm = make_llm()

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful general-purpose assistant."),
        ("human", "{query}")
    ])

    msg = prompt.format_messages(query=state.query)
    result = llm.invoke(msg)
    state.partial_answers.extra["general"] = result.content
    return state

def executor_node(state: GraphState) -> GraphState:
    state.add_agent("executor")

    if not state.plan or not state.plan.steps:
        state.partial_answers.executor_result = "No plan generated."
        return state

    plan_steps = state.plan.steps
    state.partial_answers.extra["executor_steps"] = {}

    for idx, step in enumerate(plan_steps, start=0):
        before_agents = list(state.agents_activated)
        res_text = None

        if step == "ask_technical_agent":
            state = technical_agent(state)
            res_text = state.partial_answers.extra.get("tech")

        elif step == "ask_geek_agent":
            state = geek_agent(state)
            res_text = state.partial_answers.extra.get("geek")

        elif step == "ask_general_agent":
            state = general_agent(state)
            res_text = state.partial_answers.extra.get("general")

        elif step == "gather_context":
            state = rag_retriever_node(state)
            res_text = state.partial_answers.rag_context

        elif step == "format_answer":
            state = formatter_node(state)
            res_text = state.final_answer

        else:
            res_text = f"Unknown step: {step}"

        # Сохраняем результат в executor_result и extra
        state.partial_answers.executor_result = res_text
        after_agents = list(state.agents_activated)

        state.partial_answers.extra["executor_steps"][f"step_{idx}:{step}"] = {
            "status": "done",
            "activated_agents_delta": [a for a in after_agents if a not in before_agents],
            "result_snippet": (res_text[:400] if isinstance(res_text, str) else str(res_text)) if res_text else None
        }

    return state


def formatter_node(state: GraphState) -> GraphState:
    state.add_agent("formatter")
    llm = make_llm()

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a formatting and style expert. "
                   "Rewrite the response clearly, concisely and elegantly."),
        ("human", "{text}")
    ])

    text = state.partial_answers.executor_result or "No result."
    msg = prompt.format_messages(text=text)

    result = llm.invoke(msg)
    state.final_answer = result.content

    state.add_session_entry(question=state.query, answer=state.final_answer)
    append_note(f"QA: {state.query} -> {state.final_answer[:200]}", tags=["auto"])
    return state

def supervisor_node(state: GraphState) -> GraphState:
    state.add_agent("supervisor")

    if "forbidden" in (state.final_answer or "").lower():
        state.final_answer = "Query was rejected due to safety policies."
    return state