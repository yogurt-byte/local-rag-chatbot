from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings,OllamaLLM
from langchain_core.messages import HumanMessage,AIMessage

#load the environment variables
load_dotenv()

#connecting to our document database
persistent_directory="db/chroma_db"
embedding_model = OllamaEmbeddings(model="nomic-embed-text")
db=Chroma(persist_directory=persistent_directory, embedding_function=embedding_model)

#setting up AI model
model=OllamaLLM(model="llama3.2:3b")

#storing our convos as messages
chat_history=[]

def start_chat():
    print("ask me questions! Type 'quit' to exit.")

    while True:
        question=input("\nYour question:")

        if question.lower()=='quit':
            print("Goodbye!")
            break
        ask_question(question)

def ask_question(user_question):
    print(f"\n---u asked:{user_question}---")

    #1)check if chat history is empty and if not rewrite the current quest using it
    if chat_history:
        history_text="\n".join([
            f"{'human' if isinstance(msg,HumanMessage) else 'Assistant'}: {msg.content}"
            for msg in chat_history
        ])

        rewrite_prompt = f"""Given the chat history, rewrite the new question to be standalone and searchable. Just return the rewritten question, nothing else.

        Chat History:
        {history_text}

        New Question:
        {user_question}

        Standalone Question:"""

        search_question=model.invoke(rewrite_prompt)
        print(f"searching for: {search_question}")

    else:
        search_question=user_question
    
    """SYNTAX EXPLANATION
    [
    f"{'Human' if isinstance(msg, HumanMessage) else 'Assistant'}: {msg.content}"
    for msg in chat_history
] 
This is a list comprehension — shorthand for a for loop:

# This:
[something for msg in chat_history]

# Is the same as:
result = []
for msg in chat_history:
    result.append(something)
    

'Human' if isinstance(msg, HumanMessage) else 'Assistant'

# isinstance() just checks what type the object is:
# if msg is a HumanMessage object → returns 'Human'
# if msg is an AIMessage object   → returns 'Assistant'

msg.content  # just the text stored inside the message object

FULL EXAMPLE WITH REAL DATA

chat_history = [
    HumanMessage(content="what is tesla?"),
    AIMessage(content="Tesla is an electric car company."),
]

# Loop produces:
# "Human: what is tesla?"
# "Assistant: Tesla is an electric car company."

# join combines them:
history_text = "Human: what is tesla?\nAssistant: Tesla is an electric car company."

WHICH PRINTS AS:

Human: what is tesla?
Assistant: Tesla is an electric car company.
"""

    #step2:Find relavent documents

    retriever = db.as_retriever(
        search_type="mmr",
        search_kwargs={"k":3, "fetch":6, "lambda_mult":0.7}
    )

    """EXPLANATION:
    MMR = Maximum Marginal Relevance
Controls how documents are selected. Two options:
pythonsearch_type="similarity"   # picks the 3 most similar docs
                           # problem: all 3 might be near identical

search_type="mmr"          # picks 3 diverse AND relevant docs
                           # solves your duplicate documents problem

search_kwargs
Just a dictionary of settings passed to the search:
"k": 3
How many documents to return to you finally.
python"k": 3   # give me 3 documents back
"fetch_k": 15
How many documents to fetch first before MMR picks the best ones.
python"fetch_k": 15  # fetch 15 candidates first
               # then MMR picks the best diverse 3 from those 15
Think of it like: fetch 15 applicants, interview them, pick best 3.
"lambda_mult": 0.7
Controls the balance between relevance vs diversity:
pythonlambda_mult = 0.0   # maximum diversity (might miss relevant docs)
lambda_mult = 1.0   # maximum relevance (back to duplicate problem)
lambda_mult = 0.7   # 70% relevance, 30% diversity — sweet spot
"""

    docs=retriever.invoke(search_question)

    print(f"found {len(docs)} relevant documents:")
    for i, doc in enumerate(docs,1): #1 is here because then it starts the index from 1 and not 0 like while printing
        #showing 1st 2 lines of each document
        lines=doc.page_content.split('\n')[:2]
        preview='\n'.join(lines)
        print(f" Doc{i}: {preview}...")

    #step3:Build final prompt
    doc_text="\n".join([f"-{doc.page_content}" for doc in docs])
    final_prompt = f"""You are a helpful assistant. Answer the question using only the documents provided.
If the answer isn't in the documents, say "I don't have enough information to answer that."

    Documents:
    {doc_text}

    Question:
    {search_question}
    
    Answer:"""

    #step4: get answer
    answer=model.invoke(final_prompt)

    # Step 5: Save to chat history as message objects
    chat_history.append(HumanMessage(content=user_question))
    chat_history.append(AIMessage(content=answer))

    print(f"\n--- Answer ---")
    print(answer)

    return answer

"""# chat_history now looks like:
[
    HumanMessage(content="what is tesla?"),
    AIMessage(content="Tesla is an electric car company.")
]

# After second question:
chat_history.append(HumanMessage(content="who founded it?"))
chat_history.append(AIMessage(content="Elon Musk co-founded it."))

# chat_history now looks like:
[
    HumanMessage(content="what is tesla?"),
    AIMessage(content="Tesla is an electric car company."),
    HumanMessage(content="who founded it?"),
    AIMessage(content="Elon Musk co-founded it.")
]

Why this format?
HumanMessage and AIMessage are LangChain objects that tag who said what:
pythonHumanMessage(content="...")  # marks this as user's message
AIMessage(content="...")     # marks this as AI's response
This is why later when we loop through history we can check isinstance(msg, HumanMessage) — because each message knows its own type.
"""

if __name__ == "__main__":
    start_chat()
