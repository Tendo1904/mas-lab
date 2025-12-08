from pydantic import BaseModel, Field, validator
from typing import List, Optional, Any, Dict
from datetime import datetime
import uuid

def utc_now_iso():
    return datetime.utcnow().isoformat()

class SessionEntry(BaseModel):
    question: str
    answer: str
    timestamp: str = Field(default_factory=utc_now_iso)

class MemoryNote(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    text: str
    tags: List[str] = Field(default_factory=list)
    created_at: str = Field(default_factory=utc_now_iso)

class Classification(BaseModel):
    type: str
    details: Optional[Dict[str, Any]] = None

class Plan(BaseModel):
    steps: List[str] = Field(default_factory=list)
    tools: List[str] = Field(default_factory=list)
    context_notes: List[MemoryNote] = Field(default_factory=str)

class PartialAnswers(BaseModel):
    rag_context: Optional[str] = None
    executor_result: Optional[str] = None
    extra: Dict[str, Any] = Field(default_factory=dict)

class GraphState(BaseModel):
    query: str
    user_id: Optional[str] = None

    classification: Optional[Classification] = None
    plan: Optional[Plan] = None
    partial_answers: PartialAnswers = Field(default_factory=PartialAnswers)
    final_answer: Optional[str] = None

    # memory
    session_history: List[SessionEntry] = Field(default_factory=list)
    long_memory: List[MemoryNote] = Field(default_factory=list)
    short_notes: List[str] = Field(default_factory=list)

    # logs
    agents_activated: List[str] = Field(default_factory=list)

    @validator("query")
    def query_must_not_be_empty(cls, v: str):
        if not v or not v.strip():
            raise ValueError("Query must be a non-empty string")
        return v
    
    def add_agent(self, name: str):
        self.agents_activated.append(name)
        return self
    
    def add_session_entry(self, question: str, answer: str):
        self.session_history.append(SessionEntry(question=question, answer=answer))
        return self
    
    def to_persistable(self) -> Dict:
        """Return JSON-serializable dict for saving to file."""
        return self.dict()
    
    @classmethod
    def from_persistable(cls, data: Dict):
        """Construct state from dict (e.g., loaded from JSON)."""
        return cls.parse_obj(data)