import streamlit as st
import time
import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Set page config
st.set_page_config(
    page_title="Project & Research Navigator",
    page_icon="🔍",
    layout="centered"
)

# Title
st.title("🔍 Project & Research Navigator")
st.caption("AI-powered Knowledge Retrieval Engine for Academic Research")
st.markdown("---")

# Initialize session state
if "pipeline" not in st.session_state:
    st.session_state.pipeline = None
    st.session_state.initialized = False
    st.session_state.questions_asked = []

# Sidebar for initialization
with st.sidebar:
    st.header("⚙️ System Setup")

    if st.button("🚀 Initialize AI System", type="primary", use_container_width=True):
        with st.spinner("Setting up AI pipeline (takes about 1 minute)..."):
            try:
                from ml_pipeline import get_pipeline

                st.session_state.pipeline = get_pipeline()
                st.session_state.initialized = True
                st.success("✅ AI System Ready!")
                st.balloons()
            except Exception as e:
                st.error(f"Error: {str(e)}")
                st.info("Make sure all dependencies are installed")

    st.markdown("---")

    # System status
    st.header("📊 Status")
    if st.session_state.initialized:
        st.success("✅ System Ready")
        if st.session_state.pipeline:
            status = st.session_state.pipeline.get_status()
            st.info(f"**Documents loaded:** {status['documents']}")
    else:
        st.warning("⚠️ System Not Ready")
        st.info("Click 'Initialize AI System' button")

    st.markdown("---")

    # Instructions
    st.header("ℹ️ How to Use")
    st.info("""
    1. Click **Initialize AI System**
    2. Ask a research or academic question
    3. Click **Get Answer**
    4. Review the answer with retrieved sources
    """)


# MAIN INTERFACE - SIMPLE AND CLEAR
st.header("💬 Ask a Research Question")

# Check if system is ready
if not st.session_state.initialized:
    st.warning("⚠️ Please initialize the AI system first from the sidebar")
    st.info("Click the '🚀 Initialize AI System' button in the sidebar to begin")
    st.stop()

# Question input
question = st.text_area(
    "Enter your academic or research question:",
    height=100,
    placeholder="Example: What is transformer architecture? Explain CRISPR technology.",
    help="Ask questions related to research papers, technical concepts, or academic topics",
)

# Get Answer button
if st.button("🔍 Get Answer", type="primary", use_container_width=True):
    if question:
        with st.spinner("🔍 Searching through academic documents..."):
            try:
                # Get answer from pipeline
                answer, retrieved_docs = st.session_state.pipeline.query(question)

                # Store in session
                st.session_state.last_answer = answer
                st.session_state.last_docs = retrieved_docs
                st.session_state.last_question = question

                # Add to history
                st.session_state.questions_asked.append(
                    {
                        "question": question,
                        "time": time.strftime("%H:%M:%S"),
                        "docs_found": len(retrieved_docs),
                    }
                )

            except Exception as e:
                st.error(f"Error getting answer: {str(e)}")
    else:
        st.warning("Please enter a question first")

# Display results if available
if "last_answer" in st.session_state:
    st.markdown("---")

    # Display the question
    st.subheader(f"❓ **Question:** {st.session_state.last_question}")

    # Display the answer
    st.subheader("🤖 **Answer:**")
    st.markdown(
        f'<div style="background-color: #f0f2f6; padding: 20px; border-radius: 10px; margin: 10px 0; color: #000000;">{st.session_state.last_answer}</div>',
        unsafe_allow_html=True,
    )

    # Display retrieved documents
    if st.session_state.last_docs:
        st.subheader(f"📄 **Retrieved Documents ({len(st.session_state.last_docs)})**")

        for doc in st.session_state.last_docs:
            with st.expander(
                f"Document {doc['rank']}: {doc['metadata'].get('source_file', 'Unknown')} (Similarity: {doc['similarity_score']:.3f})"
            ):
                st.write(f"**Source:** {doc['metadata'].get('source_file', 'Unknown')}")
                st.write(f"**Similarity Score:** {doc['similarity_score']:.3f}")
                st.write(f"**Content:**")
                st.text(
                    doc["content"][:500] + "..."
                    if len(doc["content"]) > 500
                    else doc["content"]
                )
    else:
        st.info("No documents retrieved for this question")

# Query history
if st.session_state.questions_asked:
    st.markdown("---")
    st.subheader("📜 Recent Questions")

    for q in reversed(st.session_state.questions_asked[-5:]):
        st.write(f"**{q['time']}:** {q['question'][:50]}... ({q['docs_found']} docs)")

# Footer
st.markdown("---")
st.caption("Project & Research Navigator | AI-powered Academic Knowledge Retrieval")

# Instructions at bottom
if len(st.session_state.questions_asked) == 0:
    st.info(
        "💡 **Tip:** After initializing, type your question above and click 'Get Answer'"
    )
