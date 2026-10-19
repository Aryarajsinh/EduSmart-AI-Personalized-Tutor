# EduSmart AI: Personalized AI Tutor System

Welcome to the **EduSmart AI Personalized Tutor** project repository, developed for **EduSmart AI**'s K-12 intelligent tutoring initiative.

This project delivers an interactive, memory-aware, and human-like AI tutor that retrieves syllabus-specific curriculum content, grounds responses using Hugging Face LLMs and embeddings, maintains conversational memory across learning sessions, and enforces stringent child privacy and data safety guardrails (COPPA/FERPA compliant).

---

## Architecture Overview

```
                          ┌───────────────────────────┐
                          │   Student / Teacher UI    │
                          │   (Streamlit / Terminal)  │
                          └─────────────┬─────────────┘
                                        │
                                        ▼
                          ┌───────────────────────────┐
                          │ Child Safety Guardrail    │
                          │ - PII Redactor (Email/Tel)│
                          │ - Safe Content Auditor    │
                          └─────────────┬─────────────┘
                                        │ (Sanitized Prompt)
                                        ▼
┌─────────────────────────┐       ┌───────────────────────────┐
│ Curriculum Repository   │       │ LangChain LCEL Pipeline   │
│ - Science Syllabus      ├──────►│ - RunnableParallel        │
│ - Math Syllabus         │       │ - Chroma Vector Retriever │
│ - History Syllabus      │       │ - Session Memory History  │
└─────────────────────────┘       └─────────────┬─────────────┘
                                                │
                                                ▼
                                  ┌───────────────────────────┐
                                  │   Hugging Face Endpoint   │
                                  │  Mistral-7B-Instruct-v0.3 │
                                  │   (Empathetic Persona)    │
                                  └─────────────┬─────────────┘
                                                │
                                                ▼
                                  ┌───────────────────────────┐
                                  │ Socratic Student Guidance │
                                  │ (Analogy + Concept Check) │
                                  └───────────────────────────┘
```

---

## Key Features & Evaluation Goals

| # | Evaluation Goal | Implementation in EduSmart AI |
|---|----------------|-------------------------------|
| **1** | **Human-like, adaptive responses** | Socratic pedagogical system prompt using `mistralai/Mistral-7B-Instruct-v0.3`. Breaks down complex concepts into relatable analogies and ends with active comprehension checks. |
| **2** | **Syllabus-specific vector search** | LangChain RAG pipeline using Chroma vector store and `sentence-transformers/all-MiniLM-L6-v2` dense embeddings with chunk citation. |
| **3** | **Conversational memory** | Session-level `ChatMessageHistory` buffer injected via LangChain Expression Language (`RunnableParallel`), tracking follow-up context and student progression. |
| **4** | **Scalability & multi-subject expansion** | Metadata-filtered Chroma collections supporting multi-subject curriculum expansion (`science`, `mathematics`, `history`) without cross-contamination. |
| **5** | **Privacy & child safety guardrails** | Real-time PII regex masking (emails, phone numbers, student IDs) and safety guardrail auditor blocking exam cheating, self-harm, and inappropriate topics. |

---

## Directory Structure

```
14) EduSmart-AI-Personalized-Tutor/
├── syllabus_data/
│   ├── science.md          # Middle School Science (Photosynthesis, Respiration, Solar System)
│   ├── mathematics.md      # Middle School Math (Linear Equations, Variables, Ratios)
│   └── history.md          # World History (Ancient River Valley Civilizations, Indus Valley)
├── chroma_db/              # Persistent Chroma vector store
├── safety_guardrails.py    # Child safety auditor & PII masking regex
├── vector_store.py         # Chroma ingestion & similarity search with metadata filtering
├── tutor_engine.py         # LangChain LCEL RAG chain, memory manager & Hugging Face LLM
├── cli_interactive.py      # Automated verification script testing all 5 goals
├── app.py                  # Full interactive Streamlit web dashboard
└── README.md               # Technical documentation & Role-Play conversation guide
```

---

## How to Run

### 1. Run the Automated 5-Goal Verification Suite
```powershell
& "d:\AI-ML Journy\venv\python.exe" cli_interactive.py
```

### 2. Launch the Streamlit Web UI
```powershell
& "d:\AI-ML Journy\venv\Scripts\streamlit.exe" run app.py
```
Open your browser at `http://localhost:8501`.

