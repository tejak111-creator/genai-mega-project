from langchain_core.prompts import PromptTemplate


RAG_PROMPT_TEMPLATE = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are a helpful assistant.

Use ONLY the context below to answer.

Context:
{context}

Question:
{question}

Answer:
""",
)