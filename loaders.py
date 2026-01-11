import pdfplumber
from langchain_core.documents import Document

def load_uploaded_documents(uploaded_files):
    documents = []

    for uploaded_file in uploaded_files:
        if uploaded_file.name.endswith(".pdf"):
            with pdfplumber.open(uploaded_file) as pdf:
                text = "".join(page.extract_text() or "" for page in pdf.pages)

            documents.append(
                Document(
                    page_content=text,
                    metadata={
                        "source_type": "upload",
                        "source_id": uploaded_file.name
                    }
                )
            )

        elif uploaded_file.name.endswith(".txt"):
            text = uploaded_file.read().decode("utf-8")
            documents.append(
                Document(
                    page_content=text,
                    metadata={
                        "source_type": "upload",
                        "source_id": uploaded_file.name
                    }
                )
            )

    return documents

