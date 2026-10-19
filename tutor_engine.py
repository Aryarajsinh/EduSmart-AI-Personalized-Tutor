"""
EduSmart AI - Personalized AI Tutor Engine
Combines:
- LangChain LCEL & RunnableMap / RunnableParallel
- ConversationalRetrievalChain compatibility
- Hugging Face Inference LLM (Mistral-7B-Instruct) with robust multi-provider fallback
- Conversational Memory (session-based ChatMessageHistory)
- Pedagogical Socratic Persona (Riya Malhotra's educational vision)
"""

import os
from typing import Dict, Any, List, Optional
from operator import itemgetter
from dotenv import load_dotenv

# LangChain Core imports
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.messages import AIMessage, HumanMessage, BaseMessage
from langchain_community.chat_message_histories import ChatMessageHistory

# Hugging Face integration
try:
    from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
    HAS_HF_ENDPOINT = True
except ImportError:
    HuggingFaceEndpoint = None
    ChatHuggingFace = None
    HAS_HF_ENDPOINT = False

# Optional legacy LangChain chain support
try:
    from langchain.chains import ConversationalRetrievalChain
    from langchain.memory import ConversationBufferMemory
    HAS_LEGACY_CHAINS = True
except ImportError:
    ConversationalRetrievalChain = None
    ConversationBufferMemory = None
    HAS_LEGACY_CHAINS = False

from safety_guardrails import ChildSafetyGuardrail
from vector_store import SyllabusVectorManager

# Load environment variables from parent folder
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))
load_dotenv()

# System Prompt reflecting Riya Malhotra's student-centered, empathetic teaching philosophy
TUTOR_SYSTEM_PROMPT = """You are 'Smarty', an empathetic, encouraging, and world-class AI tutor at EduSmart AI.
You are helping a young student ({student_name}) who is at a {learning_level} learning level.

Your Pedagogical Principles:
1. Warmth & Encouragement: Treat every question with enthusiasm and kindness. Never make a student feel inadequate for making a mistake.
2. Socratic & Adaptive Guidance: Guide the student toward answers using intuitive everyday analogies (e.g., comparing chloroplasts to miniature solar-powered kitchen kitchens).
3. Curriculum Accuracy: Ground your factual explanations strictly in the provided syllabus context below. If the information isn't in the syllabus, state it clearly in simple terms and encourage them to explore with their school teacher.
4. Active Checking: End your explanations with a friendly, bite-sized comprehension check or an engaging question to see if the student followed along.
5. Child Safety & Support: Keep all explanations completely safe, positive, and age-appropriate.

Official Syllabus Context:
-------------------------
{context}
-------------------------

Student's Past Conversation History:
{chat_history}

Current Student Question: {question}

Helpful, Inspiring AI Tutor Response:"""


def format_docs(docs: List[Any]) -> str:
    """Formats retrieved curriculum documents into a clean readable string."""
    if not docs:
        return "No specific syllabus excerpt found for this topic. Provide general foundational middle-school guidance."
    formatted = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source_file", "curriculum")
        subject = doc.metadata.get("subject", "general")
        formatted.append(f"[Source {i} - {subject.upper()} ({source})]:\n{doc.page_content.strip()}")
    return "\n\n".join(formatted)


def format_chat_history(messages: List[BaseMessage]) -> str:
    """Formats message objects into a dialogue string for the prompt."""
    if not messages:
        return "No prior conversation yet. This is the start of the session."
    formatted = []
    for msg in messages:
        role = "Student" if isinstance(msg, HumanMessage) else "Tutor (Smarty)"
        formatted.append(f"{role}: {msg.content}")
    return "\n".join(formatted)


def get_config_key(key: str) -> Optional[str]:
    val = os.getenv(key)
    if val:
        return val
    try:
        import streamlit as st
        if hasattr(st, "secrets") and key in st.secrets:
            return st.secrets[key]
    except Exception:
        pass
    return None


class PersonalizedAITutor:
    """
    Personalized AI Tutor Engine implementing LCEL, Hugging Face LLM,
    Chroma Vector Retrieval, and Session-based Conversation Memory.
    """

    def __init__(self, subject: Optional[str] = "all"):
        self.subject = subject
        self.vector_manager = SyllabusVectorManager()
        self.session_memories: Dict[str, ChatMessageHistory] = {}
        self.guardrail = ChildSafetyGuardrail()
        self.llm = self._initialize_llm()
        self._build_lcel_chain()

    def _initialize_llm(self):
        """
        Initializes Hugging Face LLM with automatic graceful fallback to Groq/OpenAI
        if HF endpoint is queued or token quota is exceeded.
        """
        hf_token = get_config_key("HF_TOKEN") or get_config_key("HUGGINGFACEHUB_API_TOKEN")

        # 1. Attempt Hugging Face mistralai/Mistral-7B-Instruct
        if HAS_HF_ENDPOINT and hf_token:
            try:
                print("[TutorEngine] Checking Hugging Face Endpoint: mistralai/Mistral-7B-Instruct-v0.3...")
                hf_llm = HuggingFaceEndpoint(
                    repo_id="mistralai/Mistral-7B-Instruct-v0.3",
                    task="text-generation",
                    max_new_tokens=64,
                    temperature=0.4,
                    huggingfacehub_api_token=hf_token,
                    timeout=15
                )
                # Verify that user token has inference provider access
                hf_llm.invoke("Ping")
                print("[TutorEngine] Hugging Face Endpoint active and verified!")
                return hf_llm
            except Exception as e:
                print(f"[TutorEngine] HF Endpoint unavailable ({e.__class__.__name__}). Falling back to secondary engine...")

        # 2. Fallback to Groq for ultra-fast, high-intelligence inference
        groq_api_key = get_config_key("GROQ_API_KEY")
        if groq_api_key:
            try:
                from langchain_groq import ChatGroq
                print("[TutorEngine] Using Groq High-Speed Inference (qwen/qwen3.8-27b)...")
                return ChatGroq(model_name="qwen/qwen3.8-27b", max_tokens=280, temperature=0.4, api_key=groq_api_key)
            except Exception as e:
                print(f"[TutorEngine] Groq fallback failed: {e}")

        # 3. Fallback to OpenAI if configured
        openai_key = get_config_key("OPENAI_API_KEY")
        if openai_key:
            from langchain_openai import ChatOpenAI
            print("[TutorEngine] Using OpenAI ChatOpenAI (gpt-3.5-turbo)...")
            return ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.4, api_key=openai_key)

        raise RuntimeError("No valid LLM credentials found. Please set HF_TOKEN, GROQ_API_KEY, or OPENAI_API_KEY.")

    def _build_lcel_chain(self):
        """
        Builds the LangChain Expression Language (LCEL) chain using RunnableParallel (RunnableMap).
        """
        self.prompt = ChatPromptTemplate.from_template(TUTOR_SYSTEM_PROMPT)
        self.output_parser = StrOutputParser()

        # RunnableMap / RunnableParallel pipeline
        self.rag_chain = (
            RunnableParallel({
                "context": (lambda x: self.vector_manager.search_curriculum(x["question"], subject=self.subject, k=3)) | RunnableLambda(format_docs),
                "question": lambda x: x["question"],
                "chat_history": lambda x: format_chat_history(x.get("chat_history", [])),
                "student_name": lambda x: x.get("student_name", "Curious Learner"),
                "learning_level": lambda x: x.get("learning_level", "Middle School (Grade 6-8)")
            })
            | self.prompt
            | self.llm
            | self.output_parser
        )

    def get_session_history(self, session_id: str) -> ChatMessageHistory:
        """Retrieves or creates the conversation history for a given student session."""
        if session_id not in self.session_memories:
            self.session_memories[session_id] = ChatMessageHistory()
        return self.session_memories[session_id]

    def set_subject(self, subject: str):
        """Allows dynamic switching between subjects (Science, Mathematics, History)."""
        self.subject = subject

    def ask(
        self,
        question: str,
        session_id: str = "default_student",
        student_name: str = "Arya",
        learning_level: str = "Grade 7 Beginner"
    ) -> Dict[str, Any]:
        """
        Processes a student query through the end-to-end pipeline:
        1. Child Safety & PII Audit
        2. Vector Search (Chroma)
        3. Conversational Memory Lookup
        4. LCEL Generation with Hugging Face
        5. Output Sanitization & Memory Update
        """
        # Step 1: Child Safety & Content Filtering
        is_safe, safety_msg = self.guardrail.audit_input_safety(question)
        if not is_safe:
            return {
                "answer": safety_msg,
                "is_safe": False,
                "sources": [],
                "redacted_pii": {}
            }

        # Step 2: Redact any sensitive PII (COPPA/FERPA compliance)
        sanitized_question, pii_counts = self.guardrail.redact_pii(question)

        # Step 3: Retrieve Conversation History
        history = self.get_session_history(session_id)
        current_messages = history.messages

        # Step 4: Retrieve Source Chunks for inspection/citation
        retrieved_docs = self.vector_manager.search_curriculum(sanitized_question, subject=self.subject, k=3)

        # Step 5: Execute LCEL RAG Chain with rate-limit resiliency
        chain_inputs = {
            "question": sanitized_question,
            "chat_history": current_messages,
            "student_name": student_name,
            "learning_level": learning_level
        }

        import time
        raw_response = None
        for attempt in range(4):
            try:
                raw_response = self.rag_chain.invoke(chain_inputs)
                break
            except Exception as e:
                err_str = str(e).lower()
                if ("rate_limit" in err_str or "429" in err_str) and attempt < 3:
                    sleep_time = 12 * (attempt + 1)
                    print(f"\n[TutorEngine] Rate limit notification from provider. Pausing for {sleep_time}s before retry (Attempt {attempt+1}/4)...")
                    time.sleep(sleep_time)
                else:
                    raise e

        # In case the response is an AIMessage object from certain chat models
        if hasattr(raw_response, "content"):
            raw_response = raw_response.content

        # Step 6: Post-process & sanitize output
        final_answer = self.guardrail.sanitize_output(str(raw_response))

        # Step 7: Update Conversation Memory
        history.add_user_message(sanitized_question)
        history.add_ai_message(final_answer)

        return {
            "answer": final_answer,
            "is_safe": True,
            "sources": [
                {
                    "content": doc.page_content,
                    "subject": doc.metadata.get("subject"),
                    "source": doc.metadata.get("source_file")
                }
                for doc in retrieved_docs
            ],
            "redacted_pii": pii_counts
        }

    def build_conversational_retrieval_chain(self):
        """
        Demonstrates traditional ConversationalRetrievalChain setup
        for architectural comparison with LCEL.
        """
        if HAS_LEGACY_CHAINS and ConversationalRetrievalChain is not None:
            memory = ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True
            )
            retriever = self.vector_manager.get_retriever(subject=self.subject, k=3)
            return ConversationalRetrievalChain.from_llm(
                llm=self.llm,
                retriever=retriever,
                memory=memory,
                verbose=False
            )
        else:
            # Modern LangChain 0.3+ uses LCEL RunnableParallel as primary architecture
            return self.rag_chain
