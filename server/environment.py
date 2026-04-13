import re
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

    def _priority_reward(self, predicted: int, actual: int) -> float:
        """Partial credit for off-by-one priority errors (e.g., High vs Medium)."""
        if predicted == actual:
            return 1.0
        elif abs(predicted - actual) == 1:
            return 0.4  # Off by one level — partial credit
        return 0.0

    def _info_reward(self, provided: str, required: str) -> float:
        """Fuzzy matching for entity extraction — rewards partial and near-matches."""
        provided = provided.strip().upper()
        required = required.strip().upper()

        if not required:
            return 1.0  # Nothing required — auto-pass

        if provided == required:
            return 1.0

        # Containment match: agent found a superset or substring of the required ID
        if required in provided or provided in required:
            return 0.7

        # Token overlap: check if the numeric/alpha tokens overlap
        tokens_provided = set(re.findall(r'[A-Z0-9]+', provided))
        tokens_required = set(re.findall(r'[A-Z0-9]+', required))
        overlap = tokens_provided & tokens_required
        if overlap:
            return 0.4 * (len(overlap) / len(tokens_required))

        return 0.0

    def step(self, action: EmailAction) -> EmailObservation:
        current_email = self.env_state.emails[self.env_state.current_step]

        # --- Category scoring ---
        cat_match = (action.category_id == current_email.true_category)

        # --- Priority scoring (fuzzy — partial credit for off-by-one) ---
        prio_reward = self._priority_reward(action.priority, current_email.true_priority)
        prio_match = (action.priority == current_email.true_priority)

        # --- Info scoring ---
        if self.env_state.task_id == 3:
            info_reward = self._info_reward(action.extracted_info, current_email.required_info)
            info_match = info_reward >= 1.0
        else:
            info_reward = 1.0
            info_match = True

        # --- Composite reward per task ---
        reward = 0.0
        if self.env_state.task_id == 1:
            # Category is primary task; priority is secondary
            reward += 0.6 if cat_match else 0.0
            reward += prio_reward * 0.4

        elif self.env_state.task_id == 2:
            # Both matter equally — but priority uses fuzzy scoring
            reward += 0.5 if cat_match else 0.0
            reward += prio_reward * 0.5

        else:  # Task 3 — entity extraction dominates
            reward += 0.25 if cat_match else 0.0
            reward += prio_reward * 0.25
            reward += info_reward * 0.5

        self.env_state.score += reward
        self.env_state.current_step += 1
        done = (self.env_state.current_step >= len(self.env_state.emails))

        msg = f"Cat: {'OK' if cat_match else 'ERR'}, Prio: {'OK' if prio_match else f'PARTIAL({prio_reward:.1f})'}"
        if self.env_state.task_id == 3:
            provided_up = action.extracted_info.strip().upper()
            required_up = current_email.required_info.strip().upper()
            msg += f", Info: {'OK' if info_match else f'PARTIAL({info_reward:.1f})'}"
            if not info_match:
                msg += f" (Exp: {required_up}, Got: {provided_up})"

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
