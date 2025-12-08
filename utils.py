import json, os
from typing import List, Dict
from state_types import MemoryNote
from datetime import datetime

MEMORY_PATH = "memory.json"

def ensure_memory_file():
    if not os.path.exists(MEMORY_PATH):
        with open(MEMORY_PATH, "w", encoding="utf-8") as f:
            json.dumps({"notes": []}, f, ensure_ascii=False, indent=2)

def load_long_memory() -> List[MemoryNote]:
    ensure_memory_file()
    with open(MEMORY_PATH, "r", encoding="utf-8") as f:
        data: Dict = json.load(f)
    notes = [MemoryNote.parse_obj(n) for n in data.get("notes", [])]
    return notes

def save_long_memory(notes: List[MemoryNote]):
    serializable = [n.dict() for n in notes]
    with open(MEMORY_PATH, "w", encoding="utf-8") as f:
        json.dump({"notes": serializable}, f, ensure_ascii=False, indent=2)

def append_note(text: str, tags=None) -> MemoryNote:
    notes = load_long_memory()
    note = MemoryNote(text=text, tags=tags or [])
    notes.append(note)
    save_long_memory(notes)
    return note

def keyword_search_notes(query: str, top_k=3):
    notes = load_long_memory()
    q = [tok.lower for tok in query.split()]
    scored = []
    for n in notes:
        text_lower = n.text.lower()
        score = sum(1 for tok in q if tok in text_lower)
        if score > 0:
            scored.append((score, n))
    scored.sort(reverse=True, key=lambda x: x[0])
    return [n for _, n in scored[:top_k]]