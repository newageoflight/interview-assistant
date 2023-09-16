from langchain import FAISS
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import OpenAIEmbeddings
from pathlib import Path


def pdf_to_kbase(pdf):
    """
    Convert a PDF into a FAISS knowledge base usable by the AI
    """
    # Since the PDF is a byte file we need to write it to a temp directory first
    tmp_dir = Path("./tmp")
    tmp_dir.mkdir(parents=True, exist_ok=True)
    with open(tmp_dir / "tmp.pdf", "wb") as ofp:
        ofp.write(pdf.getvalue())

    # Convert the chunks of text into embeddings to form a knowledge base
    loader = PyPDFLoader(str(tmp_dir / "tmp.pdf"))
    chunks = loader.load_and_split()
    embeddings = OpenAIEmbeddings()
    knowledgeBase = FAISS.from_documents(chunks, embeddings)

    (tmp_dir / "tmp.pdf").unlink()
    return knowledgeBase
