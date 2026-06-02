import os
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import sys
from pathlib import Path
from dotenv import load_dotenv
import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq

load_dotenv()

# Resolve paths relative to the repository root so the app works correctly
# regardless of the working directory at startup (e.g. Streamlit Cloud).
_REPO_ROOT = Path(__file__).parent.parent
DATA_DIR = _REPO_ROOT / "data"
INDEX_DIR = DATA_DIR / "faiss_index"

# Make sure the repo root is on sys.path so that ingest.py can be imported.
_repo_root_str = str(_REPO_ROOT)
if _repo_root_str not in sys.path:
    sys.path.insert(0, _repo_root_str)

st.set_page_config(page_title="Evaluation Policy Assistant", page_icon="🇨🇦", layout="wide")
st.title("Evaluation Policy Q&A Assistant")
st.caption("Ask questions about Canada's Policy on Results (Treasury Board, 2016) in English or French.")

lang = st.sidebar.radio("Language", ["English", "Francais"])
is_french = lang == "Francais"
st.sidebar.markdown("Built with LangChain, Groq, FAISS, Streamlit")

@st.cache_resource(show_spinner="Loading knowledge base...")
def load_vectorstore():
    if not INDEX_DIR.exists():
        with st.spinner("Knowledge base not found -- building it now from source PDFs (this takes a minute on first run)..."):
            try:
                from ingest import build_index
                build_index(data_dir=DATA_DIR, index_dir=INDEX_DIR)
            except Exception as exc:
                st.error(
                    f"Failed to build the knowledge base automatically: {exc}\n\n"
                    "Please run `python ingest.py` from the repository root and redeploy."
                )
                st.stop()
    emb = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.load_local(str(INDEX_DIR), emb, allow_dangerous_deserialization=True)

vs = load_vectorstore()
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.1, api_key=os.getenv("GROQ_API_KEY"))

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

prompt = st.chat_input("Ask your question here...")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            # --- Improved Retrieval ---
            # 1. Get more candidates initially (MMR for diversity)
            docs = vs.max_marginal_relevance_search(prompt, k=8, fetch_k=20)

            # 2. Filter by language if possible (metadata-based)
            target_lang = "fr" if is_french else "en"
            filtered = [d for d in docs if d.metadata.get("language") == target_lang]
            # Fallback: if filtering removes everything, use all docs
            if not filtered:
                filtered = docs
            docs = filtered[:5]  # keep top 5 after filtering

            # 3. Build a cleaner context with source attribution
            context_parts = []
            for i, d in enumerate(docs, 1):
                page = d.metadata.get("page", "?")
                source = d.metadata.get("source", "unknown")
                context_parts.append(f"[Source {i} - Page {page} from {source}]:\n{d.page_content}")
            context = "\n\n".join(context_parts)

            # 4. Better structured prompt
            if is_french:
                system = (
                    "Tu es un assistant spécialisé dans la Politique sur les résultats du Canada (Conseil du Trésor, 2016). "
                    "Réponds UNIQUEMENT à partir du contexte fourni ci-dessous. "
                    "Si le contexte ne contient pas la réponse, dis-le poliment. "
                    "Cite les sections et numéros de page quand c'est possible. "
                    "Réponds en français."
                )
            else:
                system = (
                    "You are an expert assistant on Canada's Policy on Results (Treasury Board, 2016). "
                    "Answer ONLY from the context provided below. "
                    "If the context doesn't contain the answer, say so politely. "
                    "Cite sections and page numbers when possible. "
                    "Answer in plain English."
                )

            full_prompt = f"{system}\n\n---\nCONTEXT:\n{context}\n---\n\nQUESTION: {prompt}\n\nANSWER:"
            answer = llm.invoke(full_prompt).content

        st.markdown(answer)
        with st.expander("📄 Sources"):
            for i, doc in enumerate(docs, 1):
                page = doc.metadata.get("page", "?")
                source = doc.metadata.get("source", "unknown")
                lang_label = "🇫🇷 FR" if doc.metadata.get("language") == "fr" else "🇬🇧 EN"
                st.markdown(f"**[{i}] Page {page}** — {lang_label} ({source})")
                st.markdown(f"> {doc.page_content}")
                st.divider()
        st.session_state.messages.append({"role": "assistant", "content": answer})
