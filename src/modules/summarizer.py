from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.output_parsers import StrOutputParser

from langchain.prompts import PromptTemplate

llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen3-235B-A22B-Instruct-2507",
    max_new_tokens=128,
    repetition_penalty=1.03,
)
chat = ChatHuggingFace(llm=llm)

prompt = PromptTemplate(
    input_variables=["text"],
    template="""
        summarize the passage below into short and consice bullet points, output nothing other than that:
        {text}
    """,
)


def summarize_text(text, max_chunk=500):
    text = text.replace("\n", " ")

    # prepare the model input
    chunks = [text[i : i + max_chunk] for i in range(0, len(text), max_chunk)]

    summary = ""

    chain = prompt | chat | StrOutputParser()
    for chunk in chunks:
        out = chain.invoke(chunk)
        summary += out + " "

    return summary.strip()


if __name__ == "__main__":
    sample_article = "Recurrent neural networks, long short-term memory [13] and gated recurrent [7] neural networks in particular, have been firmly established as state of the art approaches in sequence modeling and transduction problems such as language modeling and machine translation [35, 2, 5]. Numerous efforts have since continued to push the boundaries of recurrent language models and encoder-decoder architectures [38, 24, 15]. Recurrent models typically factor computation along the symbol positions of the input and output sequences. Aligning the positions to steps in computation time, they generate a sequence of hidden states ht, as a function of the previous hidden state ht−1 and the input for position t. This inherently sequential nature precludes parallelization within training examples, which becomes critical at longer sequence lengths, as memory constraints limit batching across examples. Recent work has achieved significant improvements in computational efficiency through factorization tricks [21] and conditional computation [32], while also improving model performance in case of the latter. The fundamental constraint of sequential computation, however, remains. Attention mechanisms have become an integral part of compelling sequence modeling and transduction models in various tasks, allowing modeling of dependencies without regard to their distance in the input or output sequences [2, 19]. In all but a few cases [27], however, such attention mechanisms are used in conjunction with a recurrent network. In this work we propose the Transformer, a model architecture eschewing recurrence and instead relying entirely on an attention mechanism to draw global dependencies between input and output. The Transformer allows for significantly more parallelization and can reach a new state of the art in translation quality after being trained for as little as twelve hours on eight P100 GPUs"

    # print(summarize_text(text=sample_article, max_chunk=750))
