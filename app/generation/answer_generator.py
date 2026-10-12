
from ollama import chat

from app.retrieval.hybrid_search import hybrid_search


MODEL_NAME = "gemma3:4b"


SYSTEM_PROMPT = """
You are a careful legal research assistant specializing in Indian law.

Use only the supplied judgment passages to answer the user's question.

LEGAL ACCURACY RULES:
1. Never invent legal rules, facts, quotations, or citations.
2. Do not turn a qualified statement into an absolute rule.
3. Distinguish the court's own reasoning and holding from:
   - arguments made by parties;
   - quotations from other judgments;
   - descriptions of earlier decisions;
   - statutory text reproduced in a judgment.
4. A quoted statement is not automatically the current court's
   final conclusion. Consider the surrounding passage.
5. If supplied passages express different principles, explain
   the distinction or uncertainty. Do not silently ignore a conflict.
6. Do not infer that bail is automatically granted or refused
   merely because an offence belongs to a particular category.
7. Do not claim that a judgment establishes a condition unless
   the supplied text supports that condition.
8. Cite factual and legal claims with the supplied source labels,
   such as [S1] and [S2].
9. If evidence is incomplete, say what the passages establish
   and what they do not establish.
10. Do not provide personalized legal advice.

ANSWER FORMAT:
- Give a direct answer first.
- Explain the relevant principles in concise bullet points.
- Distinguish general principles from case-specific observations.
- End with limitations if the supplied passages are insufficient.

Only use source labels that appear in the supplied context.
"""



def build_context(results):
    """Build labelled context and a deterministic source list."""
    context_parts = []
    sources = []

    for index, result in enumerate(results, start=1):
        source_label = f"S{index}"

        title = result.get("title") or "Untitled judgment"
        court = result.get("court") or "Court not specified"
        judgment_date = (
            result.get("judgment_date") or "Date not specified"
        )
        source_url = result.get("source_url") or ""
        source_id = result.get("source_id") or "Unknown"
        chunk_index = result.get("chunk_index", "Unknown")

        passage = (
            result.get("chunk_text")
            or result.get("text")
            or result.get("passage")
            or ""
        ).strip()

        if not passage:
            continue

        context_parts.append(
            f"""
                [{source_label}]
                Title: {title}
                Court: {court}
                Date: {judgment_date}
                Source ID: {source_id}
                Chunk: {chunk_index}
                Passage:{passage}
"""
        )

        sources.append({
            "label": source_label,
            "title": title,
            "court": court,
            "date": judgment_date,
            "source_id": source_id,
            "chunk_index": chunk_index,
            "url": source_url,
        })

    return "\n\n".join(context_parts), sources


def generate_answer(question, top_k=5):
    """Retrieve passages and generate a grounded answer."""
    if not question or not question.strip():
        raise ValueError("Question cannot be empty.")

    results = hybrid_search(question, top_k=top_k)

    if not results:
        return {
            "question": question,
            "answer": (
                "I could not find relevant passages in the current "
                "legal document collection. Please try rephrasing "
                "the question or adding more judgments to the dataset."
            ),
            "sources": [],
        }

    context, sources = build_context(results)

    if not context:
        return {
            "question": question,
            "answer": (
                "Relevant search results were returned, but no usable "
                "passage text was available to generate an answer."
            ),
            "sources": sources,
        }

    user_prompt = f"""
Question:
{question}

Retrieved legal judgment passages:
{context}

Write an answer based on these passages. Use inline source labels
such as [S1] when making claims supported by a passage.
If the passages do not establish an answer, explicitly say so.
"""

    response = chat(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        options={
            "temperature": 0,
            "num_ctx": 8192,
        },
    )

    return {
        "question": question,
        "answer": response.message.content.strip(),
        "sources": sources,
    }
