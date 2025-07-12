import streamlit as st
from modules.rag import setup_rag_pipeline
from modules.transcriber import transcribe_audio
from modules.summarizer import summarize_text
import tempfile

st.set_page_config(page_title = "🧠 Zoned-Out", layout = "centered")
st.title("🧠 Meeting Summarizer for the Zoned-Out")

uploaded_file = st.file_uploader("Upload Meeting Audio", type = ["mp3", "wav", "m4a"])


if uploaded_file:
    st.audio(uploaded_file, format = "audio/wav")
    file_path = ""
    with tempfile.NamedTemporaryFile(delete = False, suffix = ".wax") as tmp_file:
        tmp_file.write(uploaded_file.read())
        file_path = tmp_file.name

    
    st.info("Transcribing audio with Whisper...")
    transcript = transcribe_audio(file_path=file_path)
    st.success("✅ Transcription complete!")


    st.subheader("📝 Transcipt")
    st.text_area("Transcript", transcript, height = 250)


    if st.button("Summarize Transcript"):
        st.info("Summarizing...")
        summary = summarize_text(transcript)
        st.subheader("📌 Summary")
        st.write(summary)


    st.subheader("💬 Ask Questions About the Meeting")
    query = st.text_input("Type your question...")

    if 'qa_chain' not in st.session_state and transcript:
        with st.spinner("Setting up RAG pipeline..."):
            st.session_state.qa_chain = setup_rag_pipeline(transcript)
            st.success("RAG is ready!")

        
    if query and 'qa_chain' in st.session_state:
        with st.spinner("Thinking..."):
            answer = st.session_state.qa_chain.invoke(query)
            st.write("**Answer:**", answer)



