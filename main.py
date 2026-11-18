from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
# Ollama should be running

model = OllamaLLM(model="llama3.2")

template = """
You are an expert in answering questions about Agriculture.

Here are some relevant reviews: {reviews} 
# reviews = are articles, papers, and other relevant information about Agriculture.

Here is the question to answer: {question}
"""
prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model # chain to combine together for our Local LLM.

# TEST RUN
result = chain.invoke({"reviews": [], "question": "What are the best practices for sustainable agriculture?"})
print(result)
