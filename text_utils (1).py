import re
from langchain_text_splitters import RecursiveCharacterTextSplitter
from config import CHUNK_SIZE, CHUNK_OVERLAP


def clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\n+", "\n", text)
    return text.strip()


def chunk_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )

    chunks = []
    for doc in documents:
        cleaned = clean_text(doc.page_content)
        splits = splitter.split_text(cleaned)

        for i, chunk in enumerate(splits):
            chunks.append({
                "content": chunk,
                "metadata": {
                    **doc.metadata,
                    "chunk_index": i,
                    "title": doc.metadata.get("source_id", "Unknown")
                }
            })

    return chunks
