# 🧠 Zoned-Out: Meeting Summarizer

Zoned-Out is a Streamlit web application that helps you transcribe meeting audio, generate concise summaries, and ask questions about the meeting using Retrieval-Augmented Generation (RAG).

## Features

- **Audio Transcription:** Upload your meeting audio (mp3, wav, m4a) and get an accurate transcript using OpenAI Whisper.
- **Summarization:** Instantly generate a concise summary of the meeting transcript.
- **Question Answering (QA):** Ask questions about the meeting content using a RAG pipeline powered by Hugging Face models.

## How It Works

1. **Upload Audio:** Drag and drop your meeting audio file.
2. **Transcription:** The app transcribes the audio to text.
3. **Summary:** Generate a summary of the transcript with one click.
4. **Ask Questions:** Type your questions about the meeting and get answers based on the transcript.


## Project Structure

The project is organized as follows:

```
zone_out/
├── src/
│   ├── app.py
│   └── modules/
│       ├── rag.py
│       ├── transcriber.py
│       └── summarizer.py
├── requirements.txt
└── README.md
```

Enjoy effortless meeting transcription, summarization, and Q&A with Zoned-Out!
