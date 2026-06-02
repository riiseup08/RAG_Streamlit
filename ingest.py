from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

_REPO_ROOT = Path(__file__).parent
DATA_DIR = _REPO_ROOT / "data"
INDEX_DIR = DATA_DIR / "faiss_index"

DOCS = {
    "policy_on_results_EN.pdf": "en",
    "policy_on_results_FR.pdf": "fr",
}


def build_index(data_dir: Path = DATA_DIR, index_dir: Path = INDEX_DIR) -> None:
    """Build the FAISS vector index from the source PDFs and save it to *index_dir*.

    Parameters
    ----------
    data_dir:
        Directory that contains the source PDF files.
    index_dir:
        Directory where the FAISS index will be saved.
    """
    all_docs = []
    # Smaller chunks for more precise retrieval
    splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)

    for fname, lang in DOCS.items():
        fpath = str(data_dir / fname)
        print(f"Loading {fname}...")
        loader = PyPDFLoader(fpath)
        pages = loader.load()
        for p in pages:
            p.metadata["language"] = lang
            p.metadata["source"] = fname
        chunks = splitter.split_documents(pages)
        all_docs.extend(chunks)
        print(f"  -> {len(chunks)} chunks")

    print(f"\nTotal chunks: {len(all_docs)}")
    print("Building FAISS index...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(all_docs, embeddings)
    vectorstore.save_local(str(index_dir))
    print(f"Done! Index saved at {index_dir}")


if __name__ == "__main__":
    build_index()
