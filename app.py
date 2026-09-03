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
# Theme -- "package insert" clinical identity.
#
# Grounded in the actual subject matter: FDA drug labels. The federal
# government's own design system (USWDS) is built on Public Sans, so
# that's the UI typeface here -- it's not a generic choice, it's the
# same face the source documents' publisher uses. Source Serif 4 gives
# the brand mark and drug names an editorial, medical-journal weight.
# The boxed disclaimer echoes a real FDA "boxed warning." Source chips
# are drawn as specimen/label tags (hairline border, square corner,
# printed feel) rather than soft SaaS pills.
# ----------------------------------------------------------------------
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Source+Serif+4:opsz,wght@8..60,500;8..60,600;8..60,700&family=Public+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@500;600&display=swap');

:root {
    --paper:      #F4F7F6;
    --panel:      #FFFFFF;
    --ink:        #122A33;
    --ink-muted:  #5B6E72;
    --blue:       #0B5FA5;
    --blue-deep:  #0A3A5C;
    --blue-tint:  #E4EEF6;
    --rx-red:     #A6392B;
    --rx-red-tint:#F6E6E2;
    --border:     #D7E0DE;
    --mono:       'IBM Plex Mono', monospace;
}

html, body, [class*="css"]  {
    font-family: 'Public Sans', sans-serif;
    color: var(--ink);
}

.stApp {
    background: var(--paper);
}

header[data-testid="stHeader"] {
    background: transparent;
}

/* ---------------- Sidebar ---------------- */
section[data-testid="stSidebar"] {
    background: var(--blue-deep);
    border-right: 1px solid rgba(0,0,0,0.15);
}
section[data-testid="stSidebar"] * {
    color: #EAF2F8 !important;
}
section[data-testid="stSidebar"] hr {
    border-color: rgba(234, 242, 248, 0.16);
}

.brand-row {
    display: flex;
    align-items: center;
    gap: 0.55rem;
    margin-bottom: 0.3rem;
}
.rx-mark {
    width: 30px;
    height: 30px;
    flex-shrink: 0;
}
.brand-mark {
    font-family: 'Source Serif 4', serif;
    font-weight: 600;
    font-size: 1.5rem;
    letter-spacing: 0.1px;
    line-height: 1.12;
    color: #F7FAFC !important;
}
.brand-sub {
    font-size: 0.85rem;
    color: #B9CEDA !important;
    line-height: 1.5;
    margin: 0.5rem 0 1.15rem 0;
}
.side-label {
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.06em;
    color: #8FB2C9 !important;
    margin-bottom: 0.3rem;
}
.side-block {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.14);
    border-left: 2px solid #4F8FBF;
    border-radius: 3px;
    padding: 0.8rem 0.95rem;
    margin-bottom: 0.9rem;
    font-size: 0.85rem;
    line-height: 1.55;
}

/* ---------------- Header strip ---------------- */
.app-header {
    display: flex;
    align-items: baseline;
    justify-content: space-between;
    padding: 0.9rem 0.2rem 0.85rem 0.2rem;
    border-bottom: 2px solid var(--blue);
    margin-bottom: 1.4rem;
}
.app-title {
    font-family: 'Source Serif 4', serif;
    font-weight: 600;
    font-size: 2.05rem;
    color: var(--blue-deep);
    letter-spacing: 0.1px;
}
.app-tagline {
    font-size: 0.92rem;
    color: var(--ink-muted);
    max-width: 380px;
    text-align: right;
    line-height: 1.5;
}

/* ---------------- Boxed warning (disclaimer) ---------------- */
.disclaimer {
    background: var(--blue-deep);
    border: 1px solid var(--blue-deep);
    border-radius: 3px;
    padding: 0.75rem 1rem;
    font-size: 0.85rem;
    color: #EAF2F8;
    margin-bottom: 1.3rem;
    line-height: 1.55;
    display: flex;
    gap: 0.65rem;
    align-items: flex-start;
}
.disclaimer .tag {
    font-family: var(--mono);
    font-size: 0.68rem;
    font-weight: 600;
    letter-spacing: 0.06em;
    color: #9FC3DC;
    border: 1px solid #4F8FBF;
    border-radius: 2px;
    padding: 0.1rem 0.4rem;
    flex-shrink: 0;
    margin-top: 0.1rem;
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
    color: #F5FAFF;
    padding: 0.7rem 1.05rem;
    border-radius: 10px 10px 2px 10px;
    max-width: 68%;
    font-size: 0.96rem;
    line-height: 1.55;
}

.bubble-assistant {
    background: var(--panel);
    border: 1px solid var(--border);
    border-left: 3px solid var(--blue);
    padding: 0.85rem 1.1rem;
    border-radius: 2px 10px 10px 2px;
    max-width: 78%;
    font-size: 0.96rem;
    line-height: 1.65;
}

.bubble-assistant.no-answer {
    border-left: 3px solid var(--rx-red);
    background: var(--rx-red-tint);
    color: #5C231A;
}

/* ---------------- Source chips (specimen labels) ---------------- */
.source-chip {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    background: var(--panel);
    border: 1px solid var(--border);
    border-left: 2px solid var(--blue);
    color: var(--ink);
    border-radius: 2px;
    padding: 0.28rem 0.7rem;
    font-size: 0.78rem;
    margin: 0.2rem 0.35rem 0.2rem 0;
}
.source-chip .drug-name {
    font-weight: 600;
}
.source-chip .score {
    font-family: var(--mono);
    color: var(--blue);
    font-weight: 600;
}

/* ---------------- Chat input ---------------- */
[data-testid="stChatInput"] textarea {
    font-family: 'Public Sans', sans-serif;
}

/* Buttons */
.stButton > button {
    background: var(--blue);
    color: #F5FAFF;
    border: none;
    border-radius: 4px;
    font-size: 0.85rem;
    padding: 0.4rem 0.9rem;
}
.stButton > button:hover {
    background: var(--blue-deep);
    color: #F5FAFF;
}
</style>
"""

st.markdown(CSS, unsafe_allow_html=True)

RX_MARK_SVG = """
<svg class="rx-mark" viewBox="0 0 32 32" fill="none" xmlns="http://www.w3.org/2000/svg">
    <circle cx="16" cy="16" r="15" stroke="#4F8FBF" stroke-width="1.4"/>
    <path d="M11 22V10h4.2a3.6 3.6 0 0 1 0 7.2H11" stroke="#F7FAFC" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"/>
    <path d="M15 17.2 20 22" stroke="#F7FAFC" stroke-width="1.6" stroke-linecap="round"/>
</svg>
"""


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
    st.markdown(
        f'<div class="brand-row">{RX_MARK_SVG}'
        '<div class="brand-mark">Medical Pharmacy<br/>Assistant</div></div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="brand-sub">Answers grounded in official FDA drug '
        'labeling -- retrieved, cited, never invented.</div>',
        unsafe_allow_html=True,
    )

    st.markdown('<div class="side-label">HOW IT WORKS</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="side-block">Your question is matched against FDA drug '
        'labels using hybrid search, reranked for relevance, then answered '
        'strictly from the retrieved text -- with sources attached to every '
        'answer.</div>',
        unsafe_allow_html=True,
    )

    st.markdown('<div class="side-label">COVERS</div>', unsafe_allow_html=True)
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
    '<div class="disclaimer"><span class="tag">NOTE</span>'
    '<span>This assistant relays information from official medication '
    'labels. It does not diagnose and does not replace advice from a '
    'physician or pharmacist.</span></div>',
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
            f'<span class="source-chip"><span class="drug-name">{drug}</span>'
            f' &middot; {section} {score_txt}</span>'
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


# # ----------------------------------------------------------------------
# # Chat input
# # ----------------------------------------------------------------------
# question = st.chat_input("Ask about a medication, e.g. \"What are the warnings for sertraline?\"")

# if question:
#     st.session_state.messages.append({"role": "user", "content": question})

#     with st.spinner("Searching the medical label corpus..."):
#         result = rag_answer(
#             question,
#             collection=collection,
#             query_model=query_model,
#             co=co,
#             llm=llm,
#             memory=st.session_state.memory,
#             top_k=top_k,
#             min_rerank_score=min_score,
#         )

#     st.session_state.messages.append(
#         {
#             "role": "assistant",
#             "content": result["answer"],
#             "sources": result["sources"],
#         }
#     )

#     st.rerun()

# ----------------------------------------------------------------------
# Chat input & Execution
# ----------------------------------------------------------------------
question = st.chat_input("Ask about a medication, e.g. \"What are the warnings for sertraline?\"")

if question:
    # 1. Append and Render User Message Immediately
    st.session_state.messages.append({"role": "user", "content": question})
    st.markdown(f'<div class="msg-row user"><div class="bubble-user">{question}</div></div>', unsafe_allow_html=True)

    # 2. Process Assistant Response
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

    # 3. Append Assistant Message to State
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": result["answer"],
            "sources": result["sources"],
        }
    )

    # 4. Render Assistant Message Immediately
    no_answer = result["answer"].strip().startswith("I don't know")
    bubble_class = "bubble-assistant no-answer" if no_answer else "bubble-assistant"
    
    st.markdown(f'<div class="msg-row assistant"><div class="{bubble_class}">{result["answer"]}</div></div>', unsafe_allow_html=True)

    if result.get("sources"):
        with st.expander(f"Sources ({len(result['sources'])})"):
            render_sources(result["sources"])
            
    # Note: No st.rerun() is called here.
