from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import (
    HuggingFaceEmbeddings,
    HuggingFaceEndpoint,
    ChatHuggingFace,
)
from langchain.vectorstores import FAISS
# from langchain.chains import RetrievalQA
from langchain_core.documents import Document
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from huggingface_hub import login
import os


hf_token = os.getenv("HF_TOKEN")
login(token=hf_token)


# Step 1: Split trancript into chunks
def spilt_text(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return splitter.split_text(text)


# Step 2: Create vector store
def build_vector_store(chunks):
    documents = [Document(page_content=chunk) for chunk in chunks]

    embeddings = HuggingFaceEmbeddings(model_name="mixedbread-ai/mxbai-embed-large-v1")
    db = FAISS.from_documents(documents, embedding=embeddings)
    return db


# Step 3: Create RAG chain
def create_rag_chain(db):

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 3})

    llm = HuggingFaceEndpoint(
        repo_id="HuggingFaceTB/SmolLM3-3B",
        task="text-generation",
        max_new_tokens=512,
        do_sample=False,
    )

    chat = ChatHuggingFace(llm=llm, verbose=True)

    prompt_template = "You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise. Question: {question} Context: {context} Answer:"

    prompt = PromptTemplate(
        input_variables=["context", "question"], template=prompt_template,
    )
    qa_chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough(),
        }
        | prompt
        | chat
        | StrOutputParser()
    )

    # qa_chain = RetrievalQA.from_llm(llm=chat, retriever=retriever)
    return qa_chain


# Master function
def setup_rag_pipeline(transcript):
    chunks = spilt_text(transcript)

    vectorstore = build_vector_store(chunks)

    qa_chain = create_rag_chain(vectorstore)

    return qa_chain
