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
# Theme -- apothecary palette: deep pharmacy teal + amber label accent,
# warm paper background. Deliberately not the generic "clinical blue on
# white" look, and not the generic cream/terracotta AI-default either.
# ----------------------------------------------------------------------
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400;9..144,500;9..144,600&family=IBM+Plex+Sans:wght@400;500;600&display=swap');

:root {
    --bg:        #F3F4F0;
    --panel:     #FFFFFF;
    --ink:       #1C2622;
    --ink-muted: #5B6B66;
    --teal:      #1F4B43;
    --teal-dark: #14332C;
    --teal-tint: #E7EFEC;
    --amber:     #B8763E;
    --amber-tint:#F3E6D6;
    --rust:      #A85D3B;
    --rust-tint: #F4E3DA;
    --border:    #DCE3DF;
}

html, body, [class*="css"]  {
    font-family: 'IBM Plex Sans', sans-serif;
    color: var(--ink);
}

.stApp {
    background: var(--bg);
}

/* Hide default streamlit chrome that fights the custom header */
header[data-testid="stHeader"] {
    background: transparent;
}

/* ---------------- Sidebar ---------------- */
section[data-testid="stSidebar"] {
    background: var(--teal-dark);
}
section[data-testid="stSidebar"] * {
    color: #EAF1EE !important;
}
section[data-testid="stSidebar"] hr {
    border-color: rgba(234, 241, 238, 0.18);
}

.brand-mark {
    font-family: 'Fraunces', serif;
    font-weight: 600;
    font-size: 1.55rem;
    letter-spacing: 0.2px;
    line-height: 1.15;
    color: #F4EFE6 !important;
    margin-bottom: 0.15rem;
}
.brand-sub {
    font-size: 0.86rem;
    color: #C9D8D2 !important;
    line-height: 1.45;
    margin-bottom: 1.1rem;
}
.side-block {
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 10px;
    padding: 0.85rem 0.95rem;
    margin-bottom: 0.9rem;
    font-size: 0.85rem;
    line-height: 1.55;
}
.side-block b { color: #F4EFE6 !important; }

/* ---------------- Header strip ---------------- */
.app-header {
    display: flex;
    align-items: baseline;
    justify-content: space-between;
    padding: 0.9rem 0.2rem 0.85rem 0.2rem;
    border-bottom: 2px solid var(--teal);
    margin-bottom: 1.4rem;
}
.app-title {
    font-family: 'Fraunces', serif;
    font-weight: 600;
    font-size: 2.0rem;
    color: var(--teal-dark);
    letter-spacing: 0.1px;
}
.app-tagline {
    font-size: 0.92rem;
    color: var(--ink-muted);
    max-width: 380px;
    text-align: right;
    line-height: 1.5;
}

/* ---------------- Disclaimer strip ---------------- */
.disclaimer {
    background: var(--amber-tint);
    border: 1px solid #E4C79A;
    border-radius: 10px;
    padding: 0.7rem 1rem;
    font-size: 0.85rem;
    color: #6B4A22;
    margin-bottom: 1.3rem;
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
    background: var(--teal);
    color: #F4F8F6;
    padding: 0.7rem 1.05rem;
    border-radius: 16px 16px 3px 16px;
    max-width: 68%;
    font-size: 0.96rem;
    line-height: 1.55;
}

.bubble-assistant {
    background: var(--panel);
    border: 1px solid var(--border);
    border-left: 4px solid var(--teal);
    padding: 0.85rem 1.1rem;
    border-radius: 3px 12px 12px 12px;
    max-width: 78%;
    font-size: 0.96rem;
    line-height: 1.65;
}

.bubble-assistant.no-answer {
    border-left: 4px solid var(--rust);
    background: var(--rust-tint);
    color: #5E3521;
}

/* ---------------- Source chips ---------------- */
.source-chip {
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
    background: var(--teal-tint);
    border: 1px solid #C9DBD4;
    color: var(--teal-dark);
    border-radius: 999px;
    padding: 0.28rem 0.75rem;
    font-size: 0.78rem;
    margin: 0.2rem 0.35rem 0.2rem 0;
}
.source-chip .score {
    color: var(--amber);
    font-weight: 600;
}

/* ---------------- Chat input ---------------- */
[data-testid="stChatInput"] textarea {
    font-family: 'IBM Plex Sans', sans-serif;
}

/* Buttons */
.stButton > button {
    background: var(--teal);
    color: #F4F8F6;
    border: none;
    border-radius: 8px;
    font-size: 0.85rem;
    padding: 0.4rem 0.9rem;
}
.stButton > button:hover {
    background: var(--teal-dark);
    color: #F4F8F6;
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
    st.markdown('<div class="brand-mark">Medical Pharmacy<br/>Assistant</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="brand-sub">Answers grounded in official FDA drug '
        'labeling -- retrieved, cited, never invented.</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="side-block"><b>How it works</b><br/>'
        'Your question is matched against FDA drug labels using hybrid '
        'search, reranked for relevance, then answered strictly from the '
        'retrieved text -- with sources attached to every answer.</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="side-block"><b>Covers</b><br/>'
        'Indications, active ingredients, adverse reactions, warnings, '
        'contraindications, drug interactions, dosage &amp; administration.</div>',
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
