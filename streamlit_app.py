
import os

import requests
import streamlit as st


# --------------------------------------------------
# 1. Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Hybrid RAG Assistant",
    page_icon="📚",
    layout="wide",
)

API_BASE_URL = os.getenv(
    "RAG_API_URL",
    "http://127.0.0.1:8000",
).rstrip("/")


# --------------------------------------------------
# 2. Helper functions
# --------------------------------------------------

def check_api_health():
    """Check whether the FastAPI backend is responding."""
    try:
        response = requests.get(
            f"{API_BASE_URL}/health",
            timeout=5,
        )

        response.raise_for_status()
        return True, response.json()

    except requests.RequestException as error:
        return False, str(error)


def ask_rag(question):
    """Send a question to the RAG API."""
    response = requests.post(
        f"{API_BASE_URL}/v1/ask",
        json={"question": question},
        timeout=180,
    )

    response.raise_for_status()
    return response.json()


def display_metric(label, value):
    """Safely display a numeric metric."""
    if value is None:
        st.metric(label, "N/A")
    else:
        try:
            st.metric(label, f"{float(value):.3f}")
        except (ValueError, TypeError):
            st.metric(label, str(value))



def upload_document(uploaded_file):
    """Upload one document to the FastAPI ingestion endpoint."""

    files = {
        "files": (
            uploaded_file.name,
            uploaded_file.getvalue(),
            uploaded_file.type or "application/octet-stream",
        )
    }

    response = requests.post(
        f"{API_BASE_URL}/v1/ingest",
        files=files,
        timeout=180,
    )

    if not response.ok:
        st.error(f"Upload failed: HTTP {response.status_code}")

        try:
            st.json(response.json())
        except ValueError:
            st.code(response.text)

        response.raise_for_status()

    return response.json()


# --------------------------------------------------
# 3. Sidebar
# --------------------------------------------------

with st.sidebar:
    st.title("⚙️ Settings")

    st.caption("Backend configuration")
    st.code(API_BASE_URL)

    if st.button("Check backend health", use_container_width=True):
        with st.spinner("Checking backend..."):
            healthy, details = check_api_health()

        if healthy:
            st.success("FastAPI is responding.")
            st.json(details)
        else:
            st.error("Could not connect to FastAPI.")
            st.caption(str(details))

    
    st.divider()
    st.subheader("📄 Upload documents")

    uploaded_file = st.file_uploader(
        "Choose a document",
        type=["pdf", "txt", "md"],
        help="Supported formats: PDF, TXT, and Markdown.",
    )

    if uploaded_file is not None:
        st.caption(f"Selected: {uploaded_file.name}")

        if st.button(
            "Upload and ingest",
            use_container_width=True,
        ):
            try:
                with st.spinner(
                    "Uploading, chunking, and indexing document..."
                ):
                    result = upload_document(uploaded_file)

                st.success("Document ingestion completed.")
                st.json(result)

            except requests.RequestException as error:
                st.error("Document ingestion failed.")
                st.caption(str(error))

    st.divider()

    if st.button("Clear chat history", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.caption(
        "Answers are generated from retrieved documents. "
        "Always inspect the cited evidence for important decisions."
    )


# --------------------------------------------------
# 4. Main heading
# --------------------------------------------------

st.title("📚 Hybrid RAG Assistant")

st.markdown(
    """
    Ask questions about your internal documents.

    The system retrieves relevant passages, reranks the evidence,
    generates an answer, and reports citation and confidence metrics.
    """
)


# --------------------------------------------------
# 5. Initialize chat history
# --------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []


# --------------------------------------------------
# 6. Display previous messages
# --------------------------------------------------

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])

        if message["role"] == "assistant":
            result = message.get("result")

            if result:
                with st.expander("Answer diagnostics"):
                    col1, col2, col3 = st.columns(3)

                    with col1:
                        display_metric(
                            "Confidence",
                            result.get("confidence"),
                        )

                    with col2:
                        display_metric(
                            "Retrieval relevance",
                            result.get("retrieval_relevance"),
                        )

                    with col3:
                        display_metric(
                            "Completeness",
                            result.get("completeness"),
                        )

                    col4, col5 = st.columns(2)

                    with col4:
                        display_metric(
                            "Citation coverage",
                            result.get("citation_coverage"),
                        )

                    with col5:
                        display_metric(
                            "Citation accuracy",
                            result.get("citation_accuracy"),
                        )

                    if result.get("abstained"):
                        st.warning(
                            "The system abstained from answering."
                        )

                        reason = result.get("abstention_reason")
                        if reason:
                            st.write(f"Reason: {reason}")

                    invalid = result.get("invalid_citations", [])
                    if invalid:
                        st.error(
                            f"Invalid citations detected: {invalid}"
                        )

                sources = result.get("sources", [])

                if sources:
                    st.markdown("**Sources**")

                    for source in sources:
                        citation = source.get("citation", "?")
                        filename = source.get("source", "Unknown source")
                        chunk_index = source.get("chunk_index")
                        score = source.get("score")
                        preview = source.get("text_preview", "")

                        heading = f"[{citation}] {filename}"

                        if chunk_index is not None:
                            heading += f" — chunk {chunk_index}"

                        with st.expander(heading):
                            if score is not None:
                                st.caption(
                                    f"Reranker score: {score:.4f}"
                                )

                            st.write(preview or "No preview available.")


# --------------------------------------------------
# 7. Question input and RAG request
# --------------------------------------------------

question = st.chat_input(
    "Ask a question about your documents..."
)

if question:

    st.session_state.messages.append({
        "role": "user",
        "content": question,
    })

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):

        try:
            with st.spinner(
                "Retrieving documents and generating an answer..."
            ):
                result = ask_rag(question)

            answer = result.get(
                "answer",
                "The API returned no answer.",
            )

            st.markdown(answer)

            if result.get("abstained"):
                st.warning(
                    "The system could not establish a sufficiently "
                    "supported answer from the available evidence."
                )

                reason = result.get("abstention_reason")
                if reason:
                    st.caption(f"Reason: {reason}")

            # Display metrics
            st.markdown("### Answer diagnostics")
            st.caption(
                "Confidence and completeness are pipeline estimates, not "
                "probabilities of correctness. Reranker scores are model-specific. "
                "Citation accuracy depends on the configured verification process."
            )

            col1, col2, col3 = st.columns(3)

            with col1:
                display_metric(
                    "Confidence",
                    result.get("confidence"),
                )

            with col2:
                display_metric(
                    "Retrieval relevance",
                    result.get("retrieval_relevance"),
                )

            with col3:
                display_metric(
                    "Completeness",
                    result.get("completeness"),
                )

            col4, col5 = st.columns(2)

            with col4:
                display_metric(
                    "Citation coverage",
                    result.get("citation_coverage"),
                )

            with col5:
                display_metric(
                    "Citation accuracy",
                    result.get("citation_accuracy"),
                )

            # Display sources
            sources = result.get("sources", [])

            if sources:
                st.markdown("### Sources")

                for source in sources:
                    citation = source.get("citation", "?")
                    filename = source.get("source", "Unknown source")
                    chunk_index = source.get("chunk_index")
                    score = source.get("score")
                    preview = source.get("text_preview", "")

                    heading = f"[{citation}] {filename}"

                    if chunk_index is not None:
                        heading += f" — chunk {chunk_index}"

                    with st.expander(heading):
                        if score is not None:
                            st.caption(
                                f"Reranker score: {score:.4f}"
                            )

                        st.write(preview or "No preview available.")

            else:
                st.caption("No source documents were returned.")

            invalid = result.get("invalid_citations", [])
            if invalid:
                st.error(f"Invalid citations: {invalid}")

            # Save the complete response for chat history
            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "result": result,
            })

        except requests.Timeout:
            error_message = (
                "The request timed out. The RAG pipeline may need "
                "more time to retrieve, rerank, and generate an answer."
            )

            st.error(error_message)

            st.session_state.messages.append({
                "role": "assistant",
                "content": error_message,
            })

        except requests.RequestException as error:
            error_message = (
                "Could not obtain a successful response from the "
                "FastAPI backend. Check that the server is running "
                "and that the endpoint is available."
            )

            st.error(error_message)
            st.caption(str(error))

            st.session_state.messages.append({
                "role": "assistant",
                "content": error_message,
            })

        except (ValueError, TypeError, KeyError) as error:
            error_message = (
                "The API returned a response the dashboard could "
                "not process."
            )

            st.error(error_message)
            st.caption(str(error))

            st.session_state.messages.append({
                "role": "assistant",
                "content": error_message,
            })
