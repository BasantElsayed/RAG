"""
app.py

Medical Pharmacy Assistant -- Streamlit front-end.

Pipeline: retrieve_chunks() (retrieval.py, Weaviate hybrid search)
       -> retrieve_and_rerank() (rag.py, Cohere rerank)
       -> relevance check -> context -> strict medical prompt
       -> Gemini -> answer + citations (rag.py)

This file only wires the UI together -- all RAG logic lives in
retrieval.py and rag.py, unchanged from the notebook.
"""

import sys
import os

import streamlit as st

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import cohere
from sentence_transformers import SentenceTransformer
from langchain_google_genai import ChatGoogleGenerativeAI
import weaviate
from weaviate.auth import AuthApiKey

from rag import rag_answer, ConversationMemory


# ----------------------------------------------------------------------
# Page config
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="Medical Pharmacy Assistant",
    page_icon="\U0001F48A",  # pill emoji
    layout="wide",
    initial_sidebar_state="expanded",
)


# ----------------------------------------------------------------------
# Theme -- clean hospital look: white canvas, one blue, lots of
# whitespace, one font family. Nothing decorative competes with the
# content; the only job of the styling is to make answers easy to
# read and sources easy to scan.
# ----------------------------------------------------------------------
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

:root {
    --bg:        #FFFFFF;
    --bg-soft:   #F7F9FB;
    --ink:       #0F1E2C;
    --ink-muted: #64748B;
    --blue:      #1565C0;
    --blue-dark: #0D4A94;
    --blue-tint: #EAF2FC;
    --border:    #E3E8ED;
    --red:       #C0392B;
    --red-tint:  #FBEAE8;
}

html, body, [class*="css"]  {
    font-family: 'Inter', sans-serif;
    color: var(--ink);
}

.stApp {
    background: var(--bg);
}

header[data-testid="stHeader"] {
    background: transparent;
}

/* ---------------- Sidebar ---------------- */
section[data-testid="stSidebar"] {
    background: var(--bg-soft);
    border-right: 1px solid var(--border);
}
section[data-testid="stSidebar"] * {
    color: var(--ink) !important;
}
section[data-testid="stSidebar"] hr {
    border-color: var(--border);
}

.brand-mark {
    font-weight: 700;
    font-size: 1.3rem;
    line-height: 1.3;
    color: var(--blue-dark) !important;
    margin-bottom: 0.3rem;
}
.brand-sub {
    font-size: 0.86rem;
    color: var(--ink-muted) !important;
    line-height: 1.5;
    margin-bottom: 1.4rem;
}
.side-label {
    font-size: 0.78rem;
    font-weight: 600;
    color: var(--ink) !important;
    margin-bottom: 0.35rem;
}
.side-block {
    background: var(--bg);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 0.8rem 0.9rem;
    margin-bottom: 1.1rem;
    font-size: 0.85rem;
    line-height: 1.55;
}

/* ---------------- Header strip ---------------- */
.app-header {
    padding: 1rem 0.2rem 1rem 0.2rem;
    border-bottom: 1px solid var(--border);
    margin-bottom: 1.5rem;
}
.app-title {
    font-weight: 700;
    font-size: 1.7rem;
    color: var(--ink);
}
.app-tagline {
    font-size: 0.92rem;
    color: var(--ink-muted);
    margin-top: 0.25rem;
}

/* ---------------- Disclaimer ---------------- */
.disclaimer {
    background: var(--blue-tint);
    border-radius: 8px;
    padding: 0.75rem 1rem;
    font-size: 0.85rem;
    color: var(--blue-dark);
    margin-bottom: 1.5rem;
    line-height: 1.55;
}

/* ---------------- Chat bubbles ---------------- */
.msg-row {
    display: flex;
    margin-bottom: 1.1rem;
}
.msg-row.user {
    justify-content: flex-end;
}
.msg-row.assistant {
    justify-content: flex-start;
}

.bubble-user {
    background: var(--blue);
    color: #FFFFFF;
    padding: 0.7rem 1.05rem;
    border-radius: 14px 14px 2px 14px;
    max-width: 68%;
    font-size: 0.96rem;
    line-height: 1.55;
}

.bubble-assistant {
    background: var(--bg-soft);
    border: 1px solid var(--border);
    padding: 0.85rem 1.1rem;
    border-radius: 2px 14px 14px 14px;
    max-width: 78%;
    font-size: 0.96rem;
    line-height: 1.65;
}

.bubble-assistant.no-answer {
    background: var(--red-tint);
    border-color: #E9C6C1;
    color: #7A281C;
}

/* ---------------- Source chips ---------------- */
.source-chip {
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
    background: var(--blue-tint);
    color: var(--blue-dark);
    border-radius: 999px;
    padding: 0.28rem 0.75rem;
    font-size: 0.78rem;
    margin: 0.2rem 0.35rem 0.2rem 0;
}
.source-chip .score {
    font-weight: 600;
}

/* ---------------- Chat input ---------------- */
[data-testid="stChatInput"] textarea {
    font-family: 'Inter', sans-serif;
}

/* Buttons */
.stButton > button {
    background: var(--blue);
    color: #FFFFFF;
    border: none;
    border-radius: 8px;
    font-size: 0.85rem;
    padding: 0.45rem 0.95rem;
}
.stButton > button:hover {
    background: var(--blue-dark);
    color: #FFFFFF;
}
</style>
"""

st.markdown(CSS, unsafe_allow_html=True)


# ----------------------------------------------------------------------
# Cached resources -- loaded once per app lifetime, not per interaction
# ----------------------------------------------------------------------
@st.cache_resource(show_spinner="Connecting to the medical knowledge base...")
def get_weaviate_collection():
    client = weaviate.connect_to_weaviate_cloud(
        cluster_url=st.secrets["WEAVIATE_URL"],
        auth_credentials=AuthApiKey(st.secrets["WEAVIATE_API_KEY"]),
    )
    collection = client.collections.get("MedicalChunk")
    return collection


@st.cache_resource(show_spinner="Loading the embedding model...")
def get_query_model():
    return SentenceTransformer("BAAI/bge-base-en-v1.5")


@st.cache_resource(show_spinner=False)
def get_cohere_client():
    return cohere.Client(st.secrets["COHERE_API_KEY"])


@st.cache_resource(show_spinner=False)
def get_llm():
    return ChatGoogleGenerativeAI(
        model="gemini-3.1-flash-lite",
        temperature=0,
        google_api_key=st.secrets["GOOGLE_API_KEY"],
    )


collection = get_weaviate_collection()
query_model = get_query_model()
co = get_cohere_client()
llm = get_llm()


# ----------------------------------------------------------------------
# Session state
# ----------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []  # list of {"role": ..., "content": ..., "sources": [...]}

if "memory" not in st.session_state:
    st.session_state.memory = ConversationMemory()


# ----------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------
with st.sidebar:
    st.markdown('<div class="brand-mark">Medical Pharmacy Assistant</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="brand-sub">Answers grounded in official FDA drug '
        'labeling -- retrieved, cited, never invented.</div>',
        unsafe_allow_html=True,
    )

    st.markdown('<div class="side-label">How it works</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="side-block">Your question is matched against FDA drug '
        'labels using hybrid search, reranked for relevance, then answered '
        'strictly from the retrieved text -- with sources attached to every '
        'answer.</div>',
        unsafe_allow_html=True,
    )

    st.markdown('<div class="side-label">Covers</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="side-block">Indications, active ingredients, adverse '
        'reactions, warnings, contraindications, drug interactions, dosage '
        '&amp; administration.</div>',
        unsafe_allow_html=True,
    )

    with st.expander("Advanced settings"):
        top_k = st.slider("Sources per answer", min_value=3, max_value=8, value=5)
        min_score = st.slider(
            "Minimum relevance score", min_value=0.0, max_value=1.0, value=0.3, step=0.05
        )

    st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
    if st.button("Start new conversation"):
        st.session_state.messages = []
        st.session_state.memory = ConversationMemory()
        st.rerun()


# ----------------------------------------------------------------------
# Header
# ----------------------------------------------------------------------
st.markdown(
    """
    <div class="app-header">
        <div class="app-title">Medical Pharmacy Assistant</div>
        <div class="app-tagline">Ask about any medication covered in the FDA drug label corpus.</div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="disclaimer">This assistant relays information from '
    'official medication labels. It does not diagnose and does not '
    'replace advice from a physician or pharmacist.</div>',
    unsafe_allow_html=True,
)


# ----------------------------------------------------------------------
# Render chat history
# ----------------------------------------------------------------------
def render_sources(sources):
    if not sources:
        return
    chips = []
    for s in sources:
        drug = s.get("drug") or s.get("brand") or "Unknown drug"
        section = (s.get("section") or "").replace("_", " ").title()
        score = s.get("rerank_score")
        score_txt = f'<span class="score">{score:.2f}</span>' if score is not None else ""
        chips.append(
            f'<span class="source-chip">{drug} &middot; {section} {score_txt}</span>'
        )
    st.markdown("".join(chips), unsafe_allow_html=True)


for msg in st.session_state.messages:
    role_class = "user" if msg["role"] == "user" else "assistant"
    st.markdown(f'<div class="msg-row {role_class}">', unsafe_allow_html=True)

    if msg["role"] == "user":
        st.markdown(f'<div class="bubble-user">{msg["content"]}</div>', unsafe_allow_html=True)
    else:
        no_answer = msg["content"].strip().startswith("I don't know")
        bubble_class = "bubble-assistant no-answer" if no_answer else "bubble-assistant"
        st.markdown(f'<div class="{bubble_class}">{msg["content"]}</div>', unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

    if msg["role"] == "assistant" and msg.get("sources"):
        with st.expander(f"Sources ({len(msg['sources'])})"):
            render_sources(msg["sources"])


# ----------------------------------------------------------------------
# Chat input
# ----------------------------------------------------------------------
question = st.chat_input("Ask about a medication, e.g. \"What are the warnings for sertraline?\"")

if question:
    st.session_state.messages.append({"role": "user", "content": question})

    with st.spinner("Searching the medical label corpus..."):
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

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": result["answer"],
            "sources": result["sources"],
        }
    )

    st.rerun()
