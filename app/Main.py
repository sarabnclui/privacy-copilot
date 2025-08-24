import streamlit as st
from privacy_copilot.rag.answer import answer
from privacy_copilot.rag.retrieve import Retriever

st.set_page_config(page_title="Privacy Copilot GDPR", layout="wide")
st.title("Privacy Copilot (GDPR, FR)")

q = st.text_input("Ask a question (e.g., « Prospection B2B sans consentement ? »)")

col1, col2 = st.columns([1,1])
with col1:
    go = st.button("Answer")

if go and q:
    with st.spinner("Thinking…"):
        resp = answer(q)
    st.subheader("Answer")
    st.write(resp)

    st.divider()
    st.subheader("🔎 Retrieved sources")
    r = Retriever()
    for h in r.search(q, k=4):
        with st.expander(f"{h.get('source_file','(doc)')} — score {h.get('similarity', h.get('score'))}"):
            st.write(h["text"])

