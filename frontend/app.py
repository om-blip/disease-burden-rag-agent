import streamlit as st
import requests

st.set_page_config(page_title="DBE Assistant", layout="wide")

st.title("Disease Burden Estimation Assistant")

if "messages" not in st.session_state:
    st.session_state.messages = []

# -------------------------
# DISPLAY CHAT HISTORY
# -------------------------

for msg in st.session_state.messages:

    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

        if "sources" in msg:

            st.markdown("**Sources**")

            for s in msg["sources"]:

                st.markdown(f"**Source {s['id']}**")

                st.markdown(f"> {s['text']}")

        if "confidence" in msg:

            st.caption(f"Retrieval confidence: {msg['confidence']:.3f}")


# -------------------------
# USER INPUT
# -------------------------

user_input = st.chat_input("Ask a question about disease burden...")

if user_input:

    st.session_state.messages.append(
        {"role": "user", "content": user_input}
    )

    with st.chat_message("user"):
        st.markdown(user_input)

    response = requests.post(
        "http://127.0.0.1:8000/chat",
        json={"messages": st.session_state.messages}
    )

    data = response.json()


    answer = data["answer"]
    sources = data["sources"]
    confidence = data["confidence"]

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources,
        "confidence": confidence
    })

    with st.chat_message("assistant"):

        st.markdown(answer)

        st.markdown("### Sources")

        for s in sources:

            st.markdown(f"**Source {s['id']}**")

            st.markdown(f"> {s['text']}")

        st.caption(f"Retrieval confidence: {confidence:.3f}")