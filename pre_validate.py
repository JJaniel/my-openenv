import os
import sys
import subprocess
import re
import json
from typing import List

def check_file(path: str):
    if os.path.exists(path):
        print(f"✅ Found: {path}")
        return True
    else:
        print(f"❌ Missing: {path}")
        return False

def validate_structure():
    print("\n--- Step 1: Structure Check ---")
    required = [
        "models.py",
        "openenv.yaml",
        "pyproject.toml",
        "uv.lock",
        "server/environment.py",
        "server/app.py",
        "inference.py",
        "Dockerfile"
    ]
    all_found = all([check_file(f) for f in required])
    return all_found

def validate_inference_output():
    print("\n--- Step 2: Inference Format Check ---")
    try:
        # Run inference.py and capture output
        result = subprocess.run(
            [sys.executable, "inference.py"], 
            capture_output=True, 
            text=True, 
            timeout=30
        )
        output = result.stdout
        
        # Check for [START], [STEP], [END] tags
        start_tags = re.findall(r"\[START\] task=.*? env=.*? model=.*?", output)
        step_tags = re.findall(r"\[STEP\] step=\d+ action=.*? reward=.*? done=.*? error=.*?", output)
        end_tags = re.findall(r"\[END\] success=.*? steps=\d+ score=.*? rewards=.*?", output)
        
        if len(start_tags) >= 3 and len(step_tags) >= 12 and len(end_tags) >= 3:
            print(f"✅ Found {len(start_tags)} [START] tags.")
            print(f"✅ Found {len(step_tags)} [STEP] tags.")
            print(f"✅ Found {len(end_tags)} [END] tags.")
        else:
            print("❌ Mandatory logging tags are missing or incorrectly formatted.")
            print(f"Debug: START={len(start_tags)}, STEP={len(step_tags)}, END={len(end_tags)}")
            return False

        # Check Score Ranges (Must be 0.0 to 1.0)
        scores = re.findall(r"score=([0-9.]+)", output)
        for s in scores:
            val = float(s)
            if not (0.0 <= val <= 1.0):
                print(f"❌ Score {val} is out of range [0, 1]!")
                return False
        print("✅ All scores are within [0.0, 1.0] range.")
        
        return True
    except Exception as e:
        print(f"❌ Inference check failed: {e}")
        return False

def validate_openenv():
    print("\n--- Step 3: openenv validate Check ---")
    try:
        result = subprocess.run(
            ["openenv", "validate", "."], 
            capture_output=True, 
            text=True
        )
        if "[OK]" in result.stdout:
            print("✅ openenv validate passed.")
            return True
        else:
            print("❌ openenv validate failed.")
            print(result.stdout)
            return False
    except Exception as e:
        print(f"❌ openenv command failed: {e}")
        return False

def main():
    print("========================================")
    print("   OpenEnv Pre-Submission Validator")
    print("========================================")
    
    s1 = validate_structure()
    s2 = validate_inference_output()
    s3 = validate_openenv()
    
    print("\n========================================")
    if s1 and s2 and s3:
        print("🏆 SUCCESS: Your submission is ready!")
    else:
        print("⚠️ FAILURE: Please fix the issues above.")
    print("========================================")

if __name__ == "__main__":
    main()
