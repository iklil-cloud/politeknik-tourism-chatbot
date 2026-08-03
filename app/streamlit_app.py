from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config import CHAPTERS_DIR, EVAL_DATA_PATH
from app.rag_pipeline import SejarahRAG


st.set_page_config(page_title="Chatbot Pendidikan Pelancongan", page_icon="🧭", layout="wide")


@st.cache_resource(show_spinner=False)
def load_rag() -> SejarahRAG:
    rag = SejarahRAG(CHAPTERS_DIR)
    rag.ingest(reset=False)
    return rag


@st.cache_data(show_spinner=False)
def load_questions() -> pd.DataFrame:
    if EVAL_DATA_PATH.exists():
        return pd.read_csv(EVAL_DATA_PATH)
    return pd.DataFrame(columns=["question", "reference_answer"])


def render_sources(sources: list[object]) -> None:
    if not sources:
        return
    st.markdown("### Rujukan")
    for idx, source in enumerate(sources, start=1):
        title = source.source
        if source.title:
            title = f"{title} - {source.title}"
        with st.expander(f"Sumber {idx}: {title}"):
            st.write(source.text)

if "messages" not in st.session_state:
    st.session_state.messages = []

rag = load_rag()
sample_questions = load_questions()

st.title("Chatbot Pendidikan Pelancongan Politeknik")
st.caption("RAG berasaskan dokumen dalam Data untuk membantu pelajar Pengurusan Pelancongan memahami konsep pelancongan dengan lebih mudah.")
# st.sidebar.markdown("### Status Sistem")
# st.sidebar.write(f"Backend stor vektor: `{rag.backend}`")
# st.sidebar.write("Model jawapan: `qwen2.5:3b`")
# st.sidebar.write("Model embedding: `nomic-embed-text:latest`")

tab_chat, tab_summary = st.tabs(["Soal Jawab", "Ringkasan Bab"])

with tab_chat:

    # 1. Clear button (put at top)
    if st.button("🗑️ Padam Chat"):
        st.session_state.messages = []
        st.rerun()

    # 2. Show chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    # 3. Chat input (ALWAYS LAST)
    question = st.chat_input("Tanya tentang konsep, geografi, amalan atau pengurusan pelancongan")

    # 4. Handle input
    if question:
        st.session_state.messages.append({"role": "user", "content": question})

        with st.chat_message("user"):
            st.write(question)

        with st.spinner("Menjana jawapan..."):
            result = rag.answer_question(question)

        answer = result["answer"]

        st.session_state.messages.append({"role": "assistant", "content": answer})

        with st.chat_message("assistant"):
            st.write(answer)
            render_sources(result["sources"])

with tab_summary:
    chapters = rag.list_chapters()
    selected_chapter = st.selectbox("Pilih bab", chapters if chapters else ["Tiada bab ditemui"])
    if st.button("Jana Ringkasan Bab"):
        with st.spinner("Menyusun ringkasan ulang kaji..."):
            result = rag.summarize_chapter(selected_chapter)
        st.markdown("### Ringkasan")
        st.write(result["summary"])
        render_sources(result["sources"])

# with tab_practice:
#     st.markdown("Gunakan set soalan ini untuk semakan pantas atau penilaian awal chatbot.")
#     if sample_questions.empty:
#         st.info("Fail `data/test_questions.csv` belum diisi.")
#     else:
#         st.dataframe(sample_questions, use_container_width=True, hide_index=True)