"""
Medical Pharmacy Assistant
Streamlit Front-End

Pipeline:
User Question
    ↓
retrieve_chunks()
    ↓
Cohere Reranking
    ↓
Relevance Check
    ↓
Context + Strict Prompt
    ↓
Gemini 2.5 Flash
    ↓
Answer + Sources
"""

import sys
import os
import html
import time

import streamlit as st

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import cohere
from sentence_transformers import SentenceTransformer
from langchain_google_genai import ChatGoogleGenerativeAI
import weaviate
from weaviate.auth import AuthApiKey

from rag import rag_answer, ConversationMemory


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="MedGuide AI",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CUSTOM CSS
# ============================================================

CSS = """
<style>

/* =========================================================
   GLOBAL
========================================================= */

@import url(
'https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap'
);

:root {
    --bg: #071A1B;
    --bg-2: #0B2526;
    --card: rgba(255,255,255,0.055);
    --card-hover: rgba(255,255,255,0.09);

    --primary: #39D6B4;
    --primary-dark: #1EAA91;
    --primary-soft: rgba(57,214,180,0.12);

    --cyan: #55D9FF;
    --purple: #A78BFA;

    --text: #F3FAF8;
    --muted: #A4B8B5;
    --border: rgba(255,255,255,0.10);

    --danger: #FF7B7B;
    --warning: #F5C76B;
}

html,
body,
[class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

.stApp {
    background:
        radial-gradient(
            circle at 10% 10%,
            rgba(57,214,180,0.09),
            transparent 30%
        ),
        radial-gradient(
            circle at 90% 15%,
            rgba(85,217,255,0.07),
            transparent 28%
        ),
        linear-gradient(
            135deg,
            #071A1B 0%,
            #081F20 50%,
            #061719 100%
        );

    color: var(--text);
}

/* Main content */

.block-container {
    max-width: 1200px;
    padding-top: 2rem;
    padding-bottom: 7rem;
}


/* =========================================================
   SIDEBAR
========================================================= */

section[data-testid="stSidebar"] {
    background:
        linear-gradient(
            180deg,
            #061819 0%,
            #0A2324 100%
        );

    border-right: 1px solid var(--border);
}

section[data-testid="stSidebar"] * {
    color: var(--text) !important;
}

.sidebar-logo {
    padding: 0.5rem 0 1.8rem;
}

.logo-icon {
    display: inline-flex;
    align-items: center;
    justify-content: center;

    width: 48px;
    height: 48px;

    border-radius: 15px;

    background: linear-gradient(
        135deg,
        var(--primary),
        var(--cyan)
    );

    color: #05201C;
    font-size: 1.45rem;
    font-weight: 800;

    box-shadow:
        0 8px 30px rgba(57,214,180,0.22);
}

.logo-title {
    margin-top: 0.75rem;

    font-family: 'Manrope', sans-serif;
    font-size: 1.25rem;
    font-weight: 800;

    letter-spacing: -0.5px;
}

.logo-subtitle {
    color: var(--muted) !important;
    font-size: 0.78rem;
    line-height: 1.5;
    margin-top: 0.2rem;
}


/* Sidebar cards */

.sidebar-card {
    padding: 1rem;

    margin: 0.8rem 0;

    border: 1px solid var(--border);
    border-radius: 16px;

    background: var(--card);

    backdrop-filter: blur(12px);
}

.sidebar-card-title {
    font-size: 0.75rem;
    font-weight: 700;

    color: var(--primary) !important;

    text-transform: uppercase;
    letter-spacing: 1px;

    margin-bottom: 0.55rem;
}

.sidebar-card-text {
    color: var(--muted) !important;

    font-size: 0.78rem;
    line-height: 1.6;
}


/* System status */

.system-status {
    display: flex;
    align-items: center;
    gap: 8px;

    margin: 1rem 0;

    padding: 0.65rem 0.8rem;

    border-radius: 12px;

    background: rgba(57,214,180,0.07);
    border: 1px solid rgba(57,214,180,0.15);

    font-size: 0.78rem;
}

.status-dot {
    width: 8px;
    height: 8px;

    border-radius: 50%;

    background: var(--primary);

    box-shadow:
        0 0 0 5px rgba(57,214,180,0.08),
        0 0 15px rgba(57,214,180,0.7);

    animation: pulse 2s infinite;
}

@keyframes pulse {
    0% {
        box-shadow:
            0 0 0 0 rgba(57,214,180,0.4);
    }

    70% {
        box-shadow:
            0 0 0 8px rgba(57,214,180,0);
    }

    100% {
        box-shadow:
            0 0 0 0 rgba(57,214,180,0);
    }
}


/* =========================================================
   HERO
========================================================= */

.hero {
    position: relative;

    padding: 2.7rem 2.4rem 2.4rem;

    margin-bottom: 1.4rem;

    border: 1px solid var(--border);
    border-radius: 28px;

    background:
        linear-gradient(
            135deg,
            rgba(57,214,180,0.09),
            rgba(85,217,255,0.035)
        );

    backdrop-filter: blur(18px);

    overflow: hidden;
}

.hero::before {
    content: "";

    position: absolute;

    width: 260px;
    height: 260px;

    right: -90px;
    top: -120px;

    border-radius: 50%;

    background: rgba(57,214,180,0.10);

    filter: blur(10px);

    animation: floatOrb 7s ease-in-out infinite;
}

.hero::after {
    content: "";

    position: absolute;

    width: 180px;
    height: 180px;

    left: -80px;
    bottom: -100px;

    border-radius: 50%;

    background: rgba(85,217,255,0.07);

    filter: blur(15px);

    animation: floatOrb2 8s ease-in-out infinite;
}

@keyframes floatOrb {
    0%, 100% {
        transform: translateY(0px);
    }

    50% {
        transform: translateY(20px);
    }
}

@keyframes floatOrb2 {
    0%, 100% {
        transform: translateX(0px);
    }

    50% {
        transform: translateX(25px);
    }
}

.hero-badge {
    display: inline-flex;
    align-items: center;
    gap: 7px;

    padding: 0.4rem 0.8rem;

    border-radius: 999px;

    background: var(--primary-soft);

    border: 1px solid rgba(57,214,180,0.20);

    color: var(--primary);

    font-size: 0.72rem;
    font-weight: 700;

    letter-spacing: 0.6px;

    text-transform: uppercase;
}

.hero-title {
    position: relative;
    z-index: 2;

    margin-top: 1.2rem;

    font-family: 'Manrope', sans-serif;

    font-size: clamp(2rem, 4vw, 3.4rem);

    font-weight: 800;

    line-height: 1.05;

    letter-spacing: -2px;

    color: var(--text);
}

.hero-title span {
    background:
        linear-gradient(
            90deg,
            var(--primary),
            var(--cyan)
        );

    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.hero-description {
    position: relative;
    z-index: 2;

    max-width: 650px;

    margin-top: 1rem;

    color: var(--muted);

    font-size: 0.95rem;

    line-height: 1.7;
}


/* =========================================================
   QUICK QUESTIONS
========================================================= */

.quick-title {
    margin: 1.3rem 0 0.6rem;

    color: var(--muted);

    font-size: 0.72rem;
    font-weight: 700;

    letter-spacing: 1px;
    text-transform: uppercase;
}

.quick-card {
    min-height: 95px;

    padding: 1rem;

    border-radius: 17px;

    background: var(--card);

    border: 1px solid var(--border);

    transition:
        transform 0.25s ease,
        border-color 0.25s ease,
        background 0.25s ease;

    cursor: pointer;
}

.quick-card:hover {
    transform: translateY(-4px);

    background: var(--card-hover);

    border-color: rgba(57,214,180,0.35);
}

.quick-icon {
    font-size: 1.25rem;
}

.quick-name {
    margin-top: 0.45rem;

    font-size: 0.78rem;
    font-weight: 700;
}

.quick-desc {
    margin-top: 0.2rem;

    color: var(--muted);

    font-size: 0.68rem;
}


/* =========================================================
   CHAT
========================================================= */

.chat-user {
    display: flex;
    justify-content: flex-end;

    margin: 1.2rem 0;
}

.user-bubble {
    max-width: 72%;

    padding: 0.9rem 1.15rem;

    border-radius: 20px 20px 5px 20px;

    background:
        linear-gradient(
            135deg,
            #159D86,
            #1EAA91
        );

    color: white;

    font-size: 0.92rem;
    line-height: 1.6;

    box-shadow:
        0 8px 25px rgba(0,0,0,0.18);
}

.chat-assistant {
    display: flex;
    justify-content: flex-start;

    margin: 1.4rem 0;
}

.assistant-card {
    max-width: 88%;

    padding: 1.25rem;

    border-radius: 20px;

    background:
        linear-gradient(
            135deg,
            rgba(255,255,255,0.065),
            rgba(255,255,255,0.035)
        );

    border: 1px solid var(--border);

    box-shadow:
        0 12px 35px rgba(0,0,0,0.14);

    animation: answerAppear 0.45s ease;
}

@keyframes answerAppear {
    from {
        opacity: 0;
        transform: translateY(8px);
    }

    to {
        opacity: 1;
        transform: translateY(0);
    }
}

.assistant-label {
    display: flex;
    align-items: center;
    gap: 8px;

    margin-bottom: 0.8rem;

    color: var(--primary);

    font-size: 0.7rem;
    font-weight: 800;

    letter-spacing: 0.8px;

    text-transform: uppercase;
}

.ai-orb {
    width: 22px;
    height: 22px;

    display: inline-flex;
    align-items: center;
    justify-content: center;

    border-radius: 7px;

    background:
        linear-gradient(
            135deg,
            var(--primary),
            var(--cyan)
        );

    color: #05201C;

    font-size: 0.75rem;

    animation: orbGlow 2.5s infinite;
}

@keyframes orbGlow {
    0%, 100% {
        box-shadow: 0 0 0 rgba(57,214,180,0);
    }

    50% {
        box-shadow: 0 0 18px rgba(57,214,180,0.35);
    }
}

.answer-text {
    color: #E8F2EF;

    font-size: 0.93rem;

    line-height: 1.75;
}

.no-answer {
    border-color: rgba(245,199,107,0.25);

    background:
        rgba(245,199,107,0.06);
}

.no-answer .assistant-label {
    color: var(--warning);
}


/* =========================================================
   SOURCES
========================================================= */

.sources-header {
    margin-top: 1rem;
    margin-bottom: 0.65rem;

    color: var(--muted);

    font-size: 0.68rem;
    font-weight: 700;

    letter-spacing: 0.9px;

    text-transform: uppercase;
}

.source-card {
    padding: 0.8rem;

    margin-bottom: 0.5rem;

    border-radius: 13px;

    background: rgba(255,255,255,0.035);

    border: 1px solid rgba(255,255,255,0.07);

    transition:
        background 0.2s ease,
        border-color 0.2s ease;
}

.source-card:hover {
    background: rgba(255,255,255,0.06);

    border-color: rgba(57,214,180,0.20);
}

.source-drug {
    font-size: 0.78rem;
    font-weight: 700;
}

.source-section {
    margin-top: 0.2rem;

    color: var(--muted);

    font-size: 0.68rem;
}

.score-pill {
    display: inline-block;

    padding: 0.18rem 0.45rem;

    margin-left: 0.35rem;

    border-radius: 999px;

    background: var(--primary-soft);

    color: var(--primary);

    font-size: 0.63rem;
    font-weight: 700;
}


/* =========================================================
   DISCLAIMER
========================================================= */

.disclaimer {
    display: flex;
    gap: 0.7rem;
    align-items: flex-start;

    padding: 0.85rem 1rem;

    margin: 1.1rem 0;

    border-radius: 15px;

    background: rgba(245,199,107,0.055);

    border: 1px solid rgba(245,199,107,0.15);

    color: #CFC3A2;

    font-size: 0.72rem;

    line-height: 1.55;
}

.disclaimer-icon {
    color: var(--warning);

    font-size: 1rem;
}


/* =========================================================
   CHAT INPUT
========================================================= */

[data-testid="stChatInput"] {
    border-top: none !important;
}

[data-testid="stChatInput"] > div {
    background: rgba(8,30,31,0.88) !important;

    border: 1px solid rgba(57,214,180,0.18) !important;

    border-radius: 18px !important;

    box-shadow:
        0 10px 35px rgba(0,0,0,0.25);
}

[data-testid="stChatInput"] textarea {
    color: var(--text) !important;

    font-family: 'DM Sans', sans-serif !important;
}

[data-testid="stChatInput"] textarea::placeholder {
    color: #718884 !important;
}


/* =========================================================
   BUTTONS
========================================================= */

.stButton > button {
    width: 100%;

    border-radius: 12px;

    border: 1px solid var(--border);

    background: rgba(255,255,255,0.045);

    color: var(--text);

    font-size: 0.78rem;

    transition:
        all 0.2s ease;
}

.stButton > button:hover {
    border-color: rgba(57,214,180,0.4);

    background: var(--primary-soft);

    color: var(--primary);
}


/* =========================================================
   METRICS
========================================================= */

.metric-card {
    text-align: center;

    padding: 0.75rem;

    border-radius: 14px;

    background: rgba(255,255,255,0.035);

    border: 1px solid var(--border);
}

.metric-number {
    color: var(--primary);

    font-family: 'Manrope', sans-serif;

    font-size: 1rem;

    font-weight: 800;
}

.metric-label {
    color: var(--muted);

    font-size: 0.62rem;

    margin-top: 0.15rem;
}


/* =========================================================
   STREAMLIT EXPANDER
========================================================= */

[data-testid="stExpander"] {
    background: rgba(255,255,255,0.025) !important;

    border: 1px solid var(--border) !important;

    border-radius: 14px !important;
}


/* =========================================================
   SCROLLBAR
========================================================= */

::-webkit-scrollbar {
    width: 7px;
}

::-webkit-scrollbar-track {
    background: #071A1B;
}

::-webkit-scrollbar-thumb {
    background: #1E4A47;

    border-radius: 10px;
}

::-webkit-scrollbar-thumb:hover {
    background: #287067;
}

</style>
"""

st.markdown(CSS, unsafe_allow_html=True)


# ============================================================
# CACHED RESOURCES
# ============================================================

@st.cache_resource(
    show_spinner="Connecting to the medical knowledge base..."
)
def get_weaviate_collection():

    client = weaviate.connect_to_weaviate_cloud(
        cluster_url=st.secrets["WEAVIATE_URL"],
        auth_credentials=AuthApiKey(
            st.secrets["WEAVIATE_API_KEY"]
        ),
    )

    collection = client.collections.get("MedicalChunk")

    return collection


@st.cache_resource(
    show_spinner="Loading medical embedding model..."
)
def get_query_model():

    return SentenceTransformer(
        "BAAI/bge-base-en-v1.5"
    )


@st.cache_resource(show_spinner=False)
def get_cohere_client():

    return cohere.Client(
        st.secrets["COHERE_API_KEY"]
    )


@st.cache_resource(show_spinner=False)
def get_llm():

    return ChatGoogleGenerativeAI(
        model="gemini-3.1-flash-lite",
        temperature=0,
        google_api_key=st.secrets["GOOGLE_API_KEY"],
    )


# Load resources

collection = get_weaviate_collection()
query_model = get_query_model()
co = get_cohere_client()
llm = get_llm()


# ============================================================
# SESSION STATE
# ============================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "memory" not in st.session_state:
    st.session_state.memory = ConversationMemory()

if "quick_question" not in st.session_state:
    st.session_state.quick_question = None


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        """
        <div class="sidebar-logo">

            <div class="logo-icon">✚</div>

            <div class="logo-title">
                MedGuide AI
            </div>

            <div class="logo-subtitle">
                Medical Pharmacy Assistant
            </div>

        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="system-status">
            <span class="status-dot"></span>
            Knowledge base connected
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="sidebar-card">

            <div class="sidebar-card-title">
                How it works
            </div>

            <div class="sidebar-card-text">
                Your question is searched against official
                medical drug labeling using hybrid retrieval.
                Relevant evidence is then reranked and passed
                to the language model.
            </div>

        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="sidebar-card">

            <div class="sidebar-card-title">
                Knowledge Coverage
            </div>

            <div class="sidebar-card-text">
                • Indications & Uses<br>
                • Active Ingredients<br>
                • Adverse Reactions<br>
                • Warnings<br>
                • Contraindications<br>
                • Drug Interactions<br>
                • Dosage & Administration
            </div>

        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.expander("⚙ Advanced settings"):

        top_k = st.slider(
            "Sources per answer",
            min_value=3,
            max_value=8,
            value=5,
        )

        min_score = st.slider(
            "Minimum relevance score",
            min_value=0.0,
            max_value=1.0,
            value=0.3,
            step=0.05,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    if st.button("＋ Start new conversation"):

        st.session_state.messages = []

        st.session_state.memory = ConversationMemory()

        st.rerun()


# ============================================================
# HERO
# ============================================================

if not st.session_state.messages:

    st.markdown(
        """
        <div class="hero">

            <div class="hero-badge">
                ✦ AI-powered medical retrieval
            </div>

            <div class="hero-title">
                Your Medical
                <span>Knowledge Companion.</span>
            </div>

            <div class="hero-description">
                Ask about medications and receive answers grounded
                in official drug labeling. Our retrieval pipeline
                finds relevant evidence before the AI generates
                an answer.
            </div>

        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# QUICK QUESTIONS
# ============================================================

if not st.session_state.messages:

    st.markdown(
        '<div class="quick-title">Try asking about</div>',
        unsafe_allow_html=True,
    )

    q1, q2, q3, q4 = st.columns(4)

    quick_questions = [
        (
            q1,
            "⚠️",
            "Warnings",
            "What are the warnings for sertraline?"
        ),
        (
            q2,
            "◈",
            "Side Effects",
            "What are the adverse reactions of sertraline?"
        ),
        (
            q3,
            "⚕",
            "Interactions",
            "What are the drug interactions of sertraline?"
        ),
        (
            q4,
            "◉",
            "Dosage",
            "What is the dosage and administration?"
        ),
    ]

    for col, icon, title, question_text in quick_questions:

        with col:

            st.markdown(
                f"""
                <div class="quick-card">

                    <div class="quick-icon">
                        {icon}
                    </div>

                    <div class="quick-name">
                        {title}
                    </div>

                    <div class="quick-desc">
                        Explore medical label information
                    </div>

                </div>
                """,
                unsafe_allow_html=True,
            )

            if st.button(
                f"Ask about {title}",
                key=f"quick_{title}",
            ):

                st.session_state.quick_question = question_text

                st.rerun()


# ============================================================
# DISCLAIMER
# ============================================================

st.markdown(
    """
    <div class="disclaimer">

        <div class="disclaimer-icon">⚠</div>

        <div>
            <b>Medical information only.</b>
            This assistant provides information retrieved from
            medication labeling. It does not diagnose conditions
            or replace advice from a qualified physician or
            pharmacist.
        </div>

    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# RENDER SOURCES
# ============================================================

def render_sources(sources):

    if not sources:
        return

    st.markdown(
        '<div class="sources-header">Evidence used for this answer</div>',
        unsafe_allow_html=True,
    )

    for i, source in enumerate(sources, start=1):

        drug = (
            source.get("drug")
            or source.get("brand")
            or source.get("generic_name")
            or "Unknown medication"
        )

        section = (
            source.get("section")
            or "Medical information"
        )

        section = (
            section
            .replace("_", " ")
            .title()
        )

        score = source.get("rerank_score")

        score_html = ""

        if score is not None:

            score_html = (
                f'<span class="score-pill">'
                f'Relevance {score:.2f}'
                f'</span>'
            )

        safe_drug = html.escape(str(drug))
        safe_section = html.escape(str(section))

        st.markdown(
            f"""
            <div class="source-card">

                <div class="source-drug">
                    {i}. {safe_drug}
                    {score_html}
                </div>

                <div class="source-section">
                    Section: {safe_section}
                </div>

            </div>
            """,
            unsafe_allow_html=True,
        )


# ============================================================
# RENDER CHAT HISTORY
# ============================================================

for message in st.session_state.messages:

    if message["role"] == "user":

        safe_content = html.escape(
            message["content"]
        )

        st.markdown(
            f"""
            <div class="chat-user">

                <div class="user-bubble">
                    {safe_content}
                </div>

            </div>
            """,
            unsafe_allow_html=True,
        )

    else:

        answer = message["content"]

        no_answer = answer.strip().lower().startswith(
            "i don't know"
        )

        card_class = (
            "assistant-card no-answer"
            if no_answer
            else "assistant-card"
        )

        safe_answer = html.escape(answer)

        # Preserve new lines
        safe_answer = safe_answer.replace(
            "\n",
            "<br>"
        )

        st.markdown(
            f"""
            <div class="chat-assistant">

                <div class="{card_class}">

                    <div class="assistant-label">

                        <span class="ai-orb">
                            ✦
                        </span>

                        MedGuide AI
                    </div>

                    <div class="answer-text">
                        {safe_answer}
                    </div>

                </div>

            </div>
            """,
            unsafe_allow_html=True,
        )

        if message.get("sources"):

            with st.expander(
                f"🔎 View {len(message['sources'])} evidence sources"
            ):

                render_sources(
                    message["sources"]
                )


# ============================================================
# PROCESS QUESTION
# ============================================================

question = st.chat_input(
    "Ask about a medication, warning, dosage, interaction..."
)


# Quick question button

if st.session_state.quick_question:

    question = st.session_state.quick_question

    st.session_state.quick_question = None


# ============================================================
# RAG PIPELINE
# ============================================================

if question:

    # Save user message

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question,
        }
    )

    # Loading animation

    with st.status(
        "✦ Searching medical evidence...",
        expanded=True,
    ) as status:

        st.write(
            "Searching the medical knowledge base..."
        )

        time.sleep(0.15)

        st.write(
            "Comparing semantic and keyword matches..."
        )

        time.sleep(0.15)

        st.write(
            "Reranking the most relevant evidence..."
        )

        # Actual RAG call

        result = rag_answer(
            question,
            collection=collection,
            query_model=query_model,
            co=co,
            llm=llm,
            memory=st.session_state.memory,
            top_k=top_k,
            min_rerank_score=min_score,
        )

        status.update(
            label="✓ Evidence retrieved",
            state="complete",
            expanded=False,
        )

    # Save assistant response

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": result["answer"],
            "sources": result["sources"],
        }
    )

    st.rerun()
