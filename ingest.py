"""Steps 2-5: load -> split -> embed -> persist to Chroma -> sanity-test retriever.
Usage:  python ingest.py            (build index)
        python ingest.py --test "what was revenue in Q2"
"""
import sys, glob, os
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
import config


def get_vectorstore():
    return Chroma(
        collection_name=config.COLLECTION,
        embedding_function=OllamaEmbeddings(model=config.EMBED_MODEL),
        persist_directory=config.CHROMA_DIR,
    )


def get_retriever():
    return get_vectorstore().as_retriever(search_kwargs={"k": config.TOP_K})


def load_documents():
    docs = []
    for path in glob.glob(os.path.join(config.DATA_DIR, "*")):
        if path.lower().endswith(".pdf"):
            docs += PyPDFLoader(path).load()          # one Document per page, has page metadata
        elif path.lower().endswith((".txt", ".md")):
            docs += TextLoader(path, encoding="utf-8").load()
    return docs


def build_index():
    docs = load_documents()
    if not docs:
        sys.exit(f"No documents found in ./{config.DATA_DIR}. Add a PDF/TXT first.")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],  # prefer paragraph/row boundaries over mid-line cuts
    )
    chunks = splitter.split_documents(docs)
    print(f"{len(docs)} pages -> {len(chunks)} chunks")
    Chroma.from_documents(
        chunks,
        OllamaEmbeddings(model=config.EMBED_MODEL),
        collection_name=config.COLLECTION,
        persist_directory=config.CHROMA_DIR,
    )
    print("Index built.")


def test(query):
    for i, d in enumerate(get_retriever().invoke(query), 1):
        print(f"\n--- [{i}] {os.path.basename(d.metadata.get('source',''))} p.{d.metadata.get('page','?')}")
        print(d.page_content[:500])


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "--test":
        test(sys.argv[2])
    else:
        build_index()
