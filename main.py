from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from theRag import retriever # Import the retriever from theRag.py
# Ollama should be running

model = OllamaLLM(model="llama3.2")

template = """
You are an Agriculture assistant. Answer the user's question using ONLY the provided reference documents. 
If the answer cannot be found in the references, say "I don't have enough information in my database to answer that." Do not guess or use outside knowledge.

Reference Documents: {reviews} 

Here is the question to answer: {question}
"""
prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model # chain to combine together for our Local LLM.

while True:
    question = input("Ask your question about Agriculture (or type 'exit' to quit/exit): ")
    print("\n\n")

    if not question:
        continue  # Skip empty questions

    if question.lower() in ["exit", "quit"]:
        print("Exiting the program. Goodbye!")
        break

    print("\nSearching database and generating answer...\n")


    try:
        # 3. Retrieve documents and handle empty database matches
        reviews = retriever.invoke(question)
        if not reviews:
            print("No relevant documents found in the database.\n")
            continue

        formatted_reviews = "\n\n".join(doc.page_content for doc in reviews)

        # 4. Stream the answer cleanly
        print("AI: ", end="", flush=True)
        for chunk in chain.stream({"reviews": formatted_reviews, "question": question}):
            print(chunk, end="", flush=True)
        
        # 5. Print unique sources used so you can verif
        sources = {doc.metadata.get("source", "Unknown") for doc in reviews}
        print(f"\n\nSources: {', '.join(sources)}")
        print("-" * 50 + "\n")

    except Exception as e:
        # 6. Prevent crash if Ollama isn't running or disconnects
        print(f"\n[Error]: Could not generate a response. Make sure Ollama is running. ({e})\n")

        
# TEST RUN
    # reviews = retriever.invoke(question) # retrieve the relevant documents from the ChromaDB using the retriever
    # #result = chain.invoke({"reviews": reviews, "question": question}) # pass in the retrieved documents and the question to the chain to generate the answer
    # #print(result)

    # # Format the retrieved documents cleanly so raw metadata doesn't clutter the prompt
    # formatted_reviews = "\n\n".join(doc.page_content for doc in reviews)

    # # Pass "reviews" (matching {reviews} in your template) into chain.stream
    # for chunk in chain.stream({"context": formatted_reviews, "question": question}):
    #     print(chunk, end="", flush=True)
    # print("\n")


 
