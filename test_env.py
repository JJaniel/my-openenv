import sys
import os
# Add the current directory to sys.path to allow relative imports
sys.path.append(os.getcwd())

from email_triage_env.server.environment import EmailTriageEnv
from email_triage_env.models import EmailAction

def test_env():
    env = EmailTriageEnv()
    print("--- Resetting Environment ---")
    obs = env.reset()
    print(f"Initial Obs: {obs.subject} | {obs.body}")
    
    done = False
    while not done:
        # Simple policy: choose category 0 if subject has 'Cannot' or 'Critical'
        cat = 0
        prio = 1
        reason = "This looks like a support issue or error."
        
        if "pricing" in obs.subject.lower() or "partnership" in obs.subject.lower():
            cat = 1
            prio = 2
            reason = "This is a business or pricing inquiry."
        elif "love" in obs.body.lower() or "suggestion" in obs.subject.lower():
            cat = 2
            prio = 3
            reason = "This is positive feedback or a feature suggestion."
        elif "meeting" in obs.subject.lower() or "internal" in obs.subject.lower():
            cat = 3
            prio = 3
            reason = "This is an internal team communication."
            
        action = EmailAction(category_id=cat, priority=prio, reasoning=reason) 
        print(f"Action: Category {cat}, Priority {prio}, Reason: {reason}")
        
        obs = env.step(action)
        print(f"Reward: {obs.reward:.2f}")
        print(f"Message: {obs.message}")
        if obs.done:
            print("--- Done ---")
            break
        print(f"Next Obs: {obs.subject}")

if __name__ == "__main__":
    test_env()
