from typing import List, Optional
from openenv.core.env_server import Action, Observation, State
from pydantic import BaseModel

class EmailAction(Action):
    category_id: int  # 0: Support, 1: Sales, 2: Feedback, 3: Internal
    priority: int     # 1: High, 2: Medium, 3: Low
    extracted_info: str = "" # Required for Hard task
    reasoning: str    # Logic behind the decision

class EmailObservation(Observation):
    task_id: int      # 1: Easy, 2: Medium, 3: Hard
    subject: str
    body: str
    current_step: int
    total_steps: int
    reward: float = 0.0
    done: bool = False
    message: str = ""

class EmailItem(BaseModel):
    subject: str
    body: str
    true_category: int
    true_priority: int
    required_info: str = "" # Entity to extract for Hard task

class EmailState(State):
    emails: List[EmailItem] = []
    current_step: int = 0
    max_steps: int = 5
    task_id: int = 1
    score: float = 0.0
