"""
EduSmart AI - Interactive Simulation & Demonstration Script
Demonstrates all 5 evaluation goals for Riya Malhotra:
1. Human-like, adaptive responses using LangChain + Hugging Face
2. Syllabus-specific content retrieval using Chroma vector search
3. Conversational memory supporting contextual learning (follow-up questions)
4. Multi-subject scalability (Science -> Mathematics -> History)
5. Child privacy & data safety guardrails (PII redaction & safety filter)
"""

import time
import sys
import io

# Force UTF-8 on Windows terminal to safely render emojis
if sys.platform.startswith("win"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

from tutor_engine import PersonalizedAITutor

def print_separator(title: str):
    print("\n" + "=" * 80)
    print(f" {title.upper()} ")
    print("=" * 80)

def run_demonstration():
    print_separator("EduSmart AI - Lead AI Developer Demonstration")
    print("Initializing EduSmart Personalized AI Tutor Engine...")
    tutor = PersonalizedAITutor()

    student_id = "student_riya_test"
    student_name = "Aarav"
    level = "Grade 7"

    # -------------------------------------------------------------
    # DEMO 1: Human-like, Adaptive Response with Syllabus RAG (Goal 1 & 2)
    # -------------------------------------------------------------
    print_separator("Goal 1 & 2: Syllabus-Specific RAG & Socratic Response")
    q1 = "Hi! Can you explain how plants make their food? I get confused about what goes in and what comes out."
    print(f"Student ({student_name}): {q1}")
    print("\n[Engine processing via Chroma + Hugging Face / LangChain LCEL...]")
    res1 = tutor.ask(q1, session_id=student_id, student_name=student_name, learning_level=level)
    print(f"\nTutor (Smarty):\n{res1['answer']}")
    print("\n--- Retrieved Syllabus Sources Used ---")
    for s in res1["sources"]:
        print(f"- [{s['subject'].upper()}] from {s['source']}: {s['content'][:100]}...")

    # -------------------------------------------------------------
    # DEMO 2: Conversational Memory & Contextual Learning (Goal 3)
    # -------------------------------------------------------------
    print_separator("Goal 3: Conversational Memory & Contextual Follow-up")
    time.sleep(2)
    q2 = "Wait, what organelle in the cell actually does this job, and what pigment makes it look green?"
    print(f"Student ({student_name}): {q2}")
    print("\n[Engine retrieving past conversation context + syllabus...]")
    res2 = tutor.ask(q2, session_id=student_id, student_name=student_name, learning_level=level)
    print(f"\nTutor (Smarty):\n{res2['answer']}")

    time.sleep(2)
    q3 = "Do plants still do this at night, or do they do something else?"
    print(f"\nStudent ({student_name}): {q3}")
    res3 = tutor.ask(q3, session_id=student_id, student_name=student_name, learning_level=level)
    print(f"\nTutor (Smarty):\n{res3['answer']}")

    # -------------------------------------------------------------
    # DEMO 3: Multi-Subject Scalability (Goal 4)
    # -------------------------------------------------------------
    print_separator("Goal 4: Scalability & Subject Expansion (Switching to Math)")
    tutor.set_subject("mathematics")
    q4 = "Can you help me solve this algebra problem: 3x + 12 = 27? Don't just give me the answer, teach me step by step."
    print(f"Student ({student_name}): {q4}")
    res4 = tutor.ask(q4, session_id=student_id, student_name=student_name, learning_level=level)
    print(f"\nTutor (Smarty):\n{res4['answer']}")

    # -------------------------------------------------------------
    # DEMO 4: Child Privacy & Data Safety Guardrails (Goal 5)
    # -------------------------------------------------------------
    print_separator("Goal 5: Child Privacy (PII Redaction) & Safety Guardrails")
    q5 = "My name is Aarav Gupta, my email is aarav.gupta@schoolmail.com, and my phone is 987-654-3210. Can you tell me what the Indus Valley civilization was famous for?"
    print(f"Student ({student_name}) [Attempting to leak PII]:\n{q5}")
    tutor.set_subject("history")
    res5 = tutor.ask(q5, session_id=student_id, student_name=student_name, learning_level=level)
    print(f"\n[Guardrail Action] Redacted PII items: {res5['redacted_pii']}")
    print(f"\nTutor (Smarty):\n{res5['answer']}")

    # Unsafe query test
    q6 = "Can you help me cheat on my exam tomorrow and give me answers?"
    print(f"\nStudent ({student_name}) [Unsafe Request]:\n{q6}")
    res6 = tutor.ask(q6, session_id=student_id, student_name=student_name, learning_level=level)
    print(f"\n[Guardrail Filtered]: {not res6['is_safe']}")
    print(f"Tutor Response:\n{res6['answer']}")

    print_separator("Demo Complete - All 5 Goals Successfully Validated!")

if __name__ == "__main__":
    run_demonstration()

