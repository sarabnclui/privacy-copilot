# src/privacy_copilot/rag/answer.py
import os
from typing import List, Dict
from dotenv import load_dotenv
from openai import OpenAI
from privacy_copilot.rag.retrieve import Retriever

load_dotenv()  # reads OPENAI_API_KEY from .env

SYSTEM = """You are a GDPR assistant (FR). Answer in ≤120 words.
Always provide 2–4 EXACT quotes in double quotes with (source_file, page/tag).
If passages are insufficient => say 'Je ne suis pas sûr' and ask a clarifying question.
State risks + options; no categorical legal advice."""

TEMPLATE = """Question: {q}

Passages:
{ctx}

Answer (≤120 words) in French, then list 'Citations:' :
"""

def _format_ctx(passages: List[Dict]) -> str:
    lines = []
    for p in passages:
        snippet = (p.get("text","").replace("\n"," "))[:500]
        src = p.get("source_file","(doc)")
        tag = f"p.{p.get('page','?')}" if p.get("page") else p.get("tag","?")
        lines.append(f'- "{snippet}" ({src}, {tag})')
    return "\n".join(lines)

def answer(question: str, k: int = 4) -> str:
    r = Retriever()
    hits = r.search(question, k=max(k, 4))
    if not hits:
        return ("Je ne suis pas sûr. Peux-tu préciser le contexte (ex: B2B/B2C, "
                "transferts hors UE, finalité) ?")
    ctx = _format_ctx(hits[:k])
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.2,
        messages=[
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": TEMPLATE.format(q=question, ctx=ctx)}
        ],
    )
    return resp.choices[0].message.content
