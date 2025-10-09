from pydantic import BaseModel

class Numbers(BaseModel):
    a: float
    b: float

class LogEntry(BaseModel):
    user_id: str
    endpoint: str
    method: str
    input: dict
    output: dict
    timestamp: str
