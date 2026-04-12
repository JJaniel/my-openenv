import random
import json
import os
from typing import List, Optional
from openenv.core.env_server import Environment
from models import EmailAction, EmailObservation, EmailState, EmailItem

# Load real-world datasets from JSON files
def load_dataset(filename: str):
    path = os.path.join(os.path.dirname(__file__), "..", "datasets", filename)
    with open(path, "r") as f:
        return json.load(f)

EASY_EMAILS = load_dataset("easy_tasks.json")
MEDIUM_EMAILS = load_dataset("medium_tasks.json")
HARD_EMAILS = load_dataset("hard_tasks.json")

class EmailTriageEnv(Environment[EmailAction, EmailObservation, EmailState]):
    def __init__(self):
        super().__init__()
        self.env_state = EmailState()

    def reset(self, task_id: int = 1) -> EmailObservation:
        self.env_state.task_id = task_id
        source = {1: EASY_EMAILS, 2: MEDIUM_EMAILS, 3: HARD_EMAILS}.get(task_id, EASY_EMAILS)
        
        selected = random.sample(source, min(len(source), self.env_state.max_steps))
        self.env_state.emails = [
            EmailItem(
                subject=e["subject"], 
                body=e["body"], 
                true_category=e["category"], 
                true_priority=e["priority"],
                required_info=e.get("info", "")
            ) for e in selected
        ]
        self.env_state.current_step = 0
        self.env_state.score = 0.0
        
        return self._get_obs(f"Task {task_id} started.")

    def step(self, action: EmailAction) -> EmailObservation:
        current_email = self.env_state.emails[self.env_state.current_step]
        
        # Scoring
        cat_match = (action.category_id == current_email.true_category)
        prio_match = (action.priority == current_email.true_priority)
        
        # Robust info matching (handle whitespace and case)
        provided_info = action.extracted_info.strip().upper()
        required_info = current_email.required_info.strip().upper()
        info_match = (provided_info == required_info) if self.env_state.task_id == 3 else True
        
        reward = 0.0
        if self.env_state.task_id in [1, 2]:
            if cat_match: reward += 0.5
            if prio_match: reward += 0.5
        else: # Task 3
            if cat_match: reward += 0.3
            if prio_match: reward += 0.3
            if info_match: reward += 0.4
        
        self.env_state.score += reward
        self.env_state.current_step += 1
        done = (self.env_state.current_step >= len(self.env_state.emails))
        
        msg = f"Cat: {'OK' if cat_match else 'ERR'}, Prio: {'OK' if prio_match else 'ERR'}"
        if self.env_state.task_id == 3:
            msg += f", Info: {'OK' if info_match else 'ERR'}"
            if not info_match:
                msg += f" (Exp: {required_info})"
            
        if done:
            final_score = self.env_state.score / len(self.env_state.emails)
            msg += f". DONE. Score: {final_score:.2f}"
            
        return self._get_obs(msg, reward=reward, done=done)

    def state(self) -> EmailState:
        return self.env_state

    def _get_obs(self, msg: str, reward: float = 0.0, done: bool = False) -> EmailObservation:
        if self.env_state.current_step < len(self.env_state.emails):
            e = self.env_state.emails[self.env_state.current_step]
            return EmailObservation(
                task_id=self.env_state.task_id,
                subject=e.subject,
                body=e.body,
                current_step=self.env_state.current_step,
                total_steps=len(self.env_state.emails),
                reward=reward,
                done=done,
                message=msg
            )
        else:
            return EmailObservation(
                task_id=self.env_state.task_id,
                subject="DONE", body="DONE",
                current_step=self.env_state.current_step,
                total_steps=len(self.env_state.emails),
                reward=reward, done=done, message=msg
            )
