from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from theRag import retriever # Import the retriever from theRag.py
# Ollama should be runningc

model = OllamaLLM(model="llama3.2")

template = """
You are an expert in answering questions about Agriculture.

Here are some relevant reviews: {reviews} 
# reviews = are articles, papers, and other relevant information about Agriculture.

Here is the question to answer: {question}
"""
prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model # chain to combine together for our Local LLM.

while True:
    question = input("Ask your question about Agriculture (or type 'exit' to quit/exit): ")
    print("\n\n")
    if question.lower() == "exit":
        break

# TEST RUN
    reviews = retriever.invoke(question) # retrieve the relevant documents from the ChromaDB using the retriever
    result = chain.invoke({"reviews": reviews, "question": question}) # pass in the retrieved documents and the question to the chain to generate the answer
    print(result)


 
