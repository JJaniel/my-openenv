---
title: Email Triage OpenEnv
emoji: 📧
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
---

# 📧 Email Triage OpenEnv: Enterprise Workflow RL
### *Real-World Reinforcement Learning for Automated Triage & Entity Extraction*

---

[![OpenEnv](https://img.shields.io/badge/Framework-OpenEnv-blueviolet)](https://github.com/huggingface/openenv)
[![Gymnasium](https://img.shields.io/badge/Interface-Gymnasium-green)](https://gymnasium.farama.org/)
[![FastAPI](https://img.shields.io/badge/API-FastAPI-009688)](https://fastapi.tiangolo.com/)
[![Dataset](https://img.shields.io/badge/Dataset-30%20samples%2Ftier-orange)](datasets/)
[![Version](https://img.shields.io/badge/Version-0.2.0-brightgreen)](openenv.yaml)
[![License-MIT](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

This project delivers a **production-ready Reinforcement Learning environment** designed for the Meta-PyTorch Hackathon. It addresses the critical business challenge of automated email processing by combining context-aware categorization, priority-based routing, and precise entity extraction from real enterprise communications.

---

## 💡 The Challenge
Modern enterprise support systems are overwhelmed by high-volume, unstructured communication. Traditional rule-based systems fail to handle:
1. **Contextual Nuance**: Distinguishing between a "suggested feature" and a "critical bug."
2. **Urgency Prediction**: Identifying high-impact enterprise issues vs. routine feedback.
3. **Data Silos**: Manually extracting ticket IDs, invoice numbers, and project codes from free-text.

## 🚀 Key Innovation: Real-World Focus
Unlike "toy" RL environments, this system is built on **industry-standard datasets** to ensure the agent learns from authentic corporate communication patterns:
- **Enron Corpus**: Authentic corporate communication for internal routing and sales logic.
- **Tobias Bueck (IT Support)**: Modern technical support tickets (Bose, QNAP, hardware/billing).
- **Microsoft/Endava**: Enterprise-grade incident reporting (Oracle, system migrations).

---

## 🏗️ Technical Architecture

### 1. Environment Pipeline
The environment is built on the **Meta OpenEnv** framework, exposing a standard Gymnasium API via a FastAPI server. It features a **progressive difficulty hierarchy**:
- **Task 1 (Easy)**: Context-aware categorization including adversarial/ambiguous emails that defeat simple keyword matching.
- **Task 2 (Medium)**: Nuanced priority prediction across 30 diverse scenarios — subtle urgency signals, enterprise vs. minor issues.
- **Task 3 (Hard)**: Zero-shot entity extraction covering 25+ identifier types: BUG, INC, PROJ, CONTRACT, PATCH, CLIENT, RACK, SRV-DOWN, LIC, MOD, TASK, DEVICE, INV, BUILD, CAMP, DEAL, MIG, FIX, INVOICE, ROOM, VERSION, ASSET, DB-ERR, Slack channels, semver tags.

### 2. Project Structure
```text
/
├── email_triage_env/
│   ├── datasets/           # 📦 Curated Real-World Samples (JSON)
│   ├── server/
│   │   ├── environment.py  # 🧠 Core Logic & Grader (0.0 - 1.0 Reward)
│   │   └── app.py          # 🌐 FastAPI Entry Point (Port 7860)
│   ├── models.py           # 🏗️ Type-safe Pydantic Schemas
│   ├── openenv.yaml        # ⚙️ OpenEnv Configuration
│   └── pyproject.toml      # 🛠️ Dependency Management
├── inference.py            # 📊 Standardized Scorer (START/STEP/END logs)
└── Dockerfile              # 🐳 Deployment Manifest (Hugging Face Optimized)
```

---

## 📊 Environment Specification

### **Action Space (`EmailAction`)**
The agent must return a structured JSON object:
| Field | Type | Description |
| :--- | :--- | :--- |
| `category_id` | `int` | 0:Support, 1:Sales, 2:Feedback, 3:Internal |
| `priority` | `int` | 1:High, 2:Medium, 3:Low |
| `extracted_info` | `str` | Required for Task 3 (e.g., `INC-8822`, `PROJECT-ZENITH`) |
| `reasoning` | `str` | Natural language justification for the triage action |

### **Observation Space (`EmailObservation`)**
The agent receives the raw communication context:
- `subject`: The email subject line.
- `body`: The full content body.
- `task_id`: Current level (1, 2, or 3).

---

## 🚦 Benchmark & Reproducibility
The system includes a standardized `inference.py` script that emits mandatory logging tags for programmatic scoring.

**v0.2.0 Enhancements:**
- 📦 **30 samples per tier** (3× larger than v0.1.0) with adversarial/ambiguous cases
- 🧠 **Chain-of-thought prompting** — LLM reasons step-by-step before answering
- 🎯 **Fuzzy reward shaping** — partial credit for off-by-one priority, token-overlap entity matching
- 🔍 **25+ entity patterns** — covers PROJ, CONTRACT, PATCH, RACK, LIC, DEAL, MIG, Slack, semver and more
- ⚡ **Rebalanced rewards** — category weighted higher in Task 1; entity extraction weighted 50% in Task 3

**Baseline Performance Metrics (Qwen2.5-72B):**
- **Task 1 (Easy)**: 0.80 Avg Reward
- **Task 2 (Medium)**: 0.90 Avg Reward
- **Task 3 (Hard)**: 0.86 Avg Reward

---

## 🛠️ Getting Started

### 1. Prerequisites
- Python 3.12+
- `openenv-core`
- OpenAI API Key (or compatible endpoint)

### 2. Local Run
```bash
# Set environment variables
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"
export HF_TOKEN="your_token_here"

# Run the inference script
python inference.py
```

### 3. Deployment
```bash
openenv push --repo-id your-hf-username/email-triage-env
```

## 📜 Acknowledgments
- **Hugging Face / Meta**: For the OpenEnv framework.
- **Enron, Tobi-Bueck, & Microsoft**: For providing the real-world datasets that power this environment.
