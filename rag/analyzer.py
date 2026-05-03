"""Document Analyzer — generates document overview after PDF processing."""
from __future__ import annotations
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

ANALYSIS_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a document analyst. Given the text below, return a structured analysis "
     "with exactly these fields on separate lines:\n"
     "SUMMARY: one paragraph overview\n"
     "TOPICS: comma-separated list of 5-8 main topics\n"
     "KEY_FACTS: top 3 facts separated by semicolons\n"
     "DOCUMENT_TYPE: one of [Technical Report, Research Paper, Legal Document, "
     "Educational Material, Business Document, Other]\n"
     "Keep each field concise. Do not use markdown."),
    ("human", "Document text:\n{text}"),
])


def analyze_document(splits: list, llm) -> dict:
    total_pages = max((int(d.metadata.get("page", 0)) for d in splits), default=0) + 1
    total_words = sum(len(d.page_content.split()) for d in splits)
    total_chunks = len(splits)
    sample_indices = list({0, len(splits) // 2, len(splits) - 1})
    sample_text = "\n\n---\n\n".join(splits[i].page_content for i in sorted(sample_indices))
    sample_text = sample_text[:2000]
    try:
        result = llm.invoke(ANALYSIS_PROMPT.format_messages(text=sample_text))
        raw = result.content if hasattr(result, "content") else str(result)
    except Exception as e:
        raw = "SUMMARY: Analysis unavailable (" + str(e) + ")\nTOPICS: —\nKEY_FACTS: —\nDOCUMENT_TYPE: Other"
    parsed = {"summary": "", "topics": [], "key_facts": [], "doc_type": "Other"}
    for line in raw.splitlines():
        line = line.strip()
        if line.startswith("SUMMARY:"):
            parsed["summary"] = line[len("SUMMARY:"):].strip()
        elif line.startswith("TOPICS:"):
            raw_topics = line[len("TOPICS:"):].strip()
            parsed["topics"] = [t.strip() for t in raw_topics.split(",") if t.strip()]
        elif line.startswith("KEY_FACTS:"):
            raw_facts = line[len("KEY_FACTS:"):].strip()
            parsed["key_facts"] = [f.strip() for f in raw_facts.split(";") if f.strip()]
            if not parsed["key_facts"]:
                parsed["key_facts"] = [raw_facts]
        elif line.startswith("DOCUMENT_TYPE:"):
            parsed["doc_type"] = line[len("DOCUMENT_TYPE:"):].strip()
    parsed["stats"] = {"pages": total_pages, "words": total_words, "chunks": total_chunks}
    return parsed
