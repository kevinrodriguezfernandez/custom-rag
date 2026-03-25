# app/main.py
"""Streamlit UI entry point for the Custom RAG frontend.

Run with:
    streamlit run app/main.py
"""

import streamlit as st


def main() -> None:
    """Render the RAG chat interface."""
    st.set_page_config(page_title="Custom RAG", layout="wide")
    st.title("Custom RAG")
    st.caption("Ask questions about your ingested documents.")

    # --- Chat input ---
    query = st.chat_input("Ask a question...")
    if query:
        st.chat_message("user").write(query)
        # TODO: call the API /chat endpoint and display the response
        st.chat_message("assistant").write("Chat endpoint integration coming soon.")


if __name__ == "__main__":
    main()
