"""
EduSmart AI - Streamlit Web Application
Personalized AI Tutor for K-12 Students
Product Demo for Riya Malhotra & EduSmart Leadership
"""

import os
import streamlit as st
from tutor_engine import PersonalizedAITutor

st.set_page_config(
    page_title="EduSmart AI - Personalized Tutor",
    page_icon="🎓",
    layout="wide"
)

# Custom Styling
st.markdown("""
<style>
    .main-title {
        color: #1E3A8A;
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        color: #4B5563;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }
    .stChatMessage {
        border-radius: 12px;
        padding: 12px;
        margin-bottom: 8px;
    }
    .badge {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 12px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-right: 5px;
    }
    .badge-rag { background-color: #DBEAFE; color: #1E40AF; }
    .badge-memory { background-color: #D1FAE5; color: #065F46; }
    .badge-safety { background-color: #FEF3C7; color: #92400E; }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def get_tutor_engine():
    return PersonalizedAITutor(subject="all")

# Initialize Session State
if "tutor" not in st.session_state:
    st.session_state.tutor = get_tutor_engine()

if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = [
        {"role": "assistant", "content": "👋 Hi there! I'm **Smarty**, your personal AI tutor at EduSmart AI! What exciting topic are we exploring today? (Science, Math, or History?)"}
    ]

if "session_id" not in st.session_state:
    st.session_state.session_id = "student_session_001"

# Sidebar Controls
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/graduation-cap.png", width=64)
    st.title("EduSmart Controls")
    
    st.markdown("### 👤 Student Profile")
    student_name = st.text_input("Student Name", value="Aarav")
    grade_level = st.selectbox(
        "Learning Level",
        ["Grade 6 (Foundational)", "Grade 7 (Intermediate)", "Grade 8 (Advanced)"]
    )
    
    st.markdown("### 📚 Subject Scope")
    subject = st.selectbox(
        "Target Subject (Chroma Vector Filter)",
        ["all", "science", "mathematics", "history"],
        index=0
    )
    st.session_state.tutor.set_subject(subject)
    
    st.markdown("---")
    st.markdown("### 🛡️ Safety & Architecture")
    st.markdown("""
    - **LLM**: Hugging Face `Mistral-7B-Instruct`
    - **Embeddings**: `sentence-transformers/all-MiniLM-L6-v2`
    - **Vector DB**: Chroma (Syllabus RAG)
    - **Framework**: LangChain LCEL + RunnableMap
    - **Safety**: COPPA / FERPA PII Masking & Moderation
    """)
    
    if st.button("🧹 Clear Chat History"):
        st.session_state.chat_messages = [
            {"role": "assistant", "content": f"Hi {student_name}! Fresh start—what shall we learn next?"}
        ]
        st.session_state.tutor.session_memories[st.session_state.session_id].clear()
        st.rerun()

# Main Layout Header
st.markdown('<div class="main-title">EduSmart AI • Personalized Learning Companion</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Empathetic, memory-aware tutoring powered by LangChain LCEL and Hugging Face</div>', unsafe_allow_html=True)

# Status Badges
st.markdown("""
<span class="badge badge-rag">✓ RAG Active (Chroma)</span>
<span class="badge badge-memory">✓ Session Memory Online</span>
<span class="badge badge-safety">✓ Minor Protection Guardrails Enabled</span>
""", unsafe_allow_html=True)
st.write("")

# Render Chat History
for message in st.session_state.chat_messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "sources" in message and message["sources"]:
            with st.expander("📖 Curriculum Citations (RAG Documents)"):
                for s in message["sources"]:
                    st.caption(f"**Subject:** {s['subject'].upper()} | **Source:** `{s['source']}`")
                    st.write(s["content"])
        if "redacted" in message and message["redacted"]:
            st.info(f"🔒 Privacy Guardrail: Masked PII entities {message['redacted']}")

# Chat Input Box
user_prompt = st.chat_input("Ask a question about your lessons...")

if user_prompt:
    # 1. Display user message
    st.session_state.chat_messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.markdown(user_prompt)

    # 2. Get Tutor Response
    with st.chat_message("assistant"):
        with st.spinner("Thinking like a helpful teacher..."):
            response = st.session_state.tutor.ask(
                question=user_prompt,
                session_id=st.session_state.session_id,
                student_name=student_name,
                learning_level=grade_level
            )
            st.markdown(response["answer"])

            if response["sources"]:
                with st.expander("📖 Curriculum Citations (RAG Documents)"):
                    for s in response["sources"]:
                        st.caption(f"**Subject:** {s['subject'].upper()} | **Source:** `{s['source']}`")
                        st.write(s["content"])

            if response["redacted_pii"]:
                st.info(f"🔒 Privacy Guardrail: Masked PII entities {response['redacted_pii']}")

            # Save to session
            st.session_state.chat_messages.append({
                "role": "assistant",
                "content": response["answer"],
                "sources": response["sources"],
                "redacted": response["redacted_pii"]
            })

