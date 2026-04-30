from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

DATA_DIR = Path("data")
INDEX_DIR = DATA_DIR / "faiss_index"

DOCS = {
    "policy_on_results_EN.pdf": "en",
    "policy_on_results_FR.pdf": "fr",
}

all_docs = []
# Smaller chunks for more precise retrieval
splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)

for fname, lang in DOCS.items():
    fpath = str(DATA_DIR / fname)
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
vectorstore.save_local(str(INDEX_DIR))
print(f"Done! Index saved at {INDEX_DIR}")
