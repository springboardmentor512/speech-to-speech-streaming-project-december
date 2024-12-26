from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import ConversationChain

# It creates a memory for the conversation
memory = ConversationBufferMemory()

# Intialize the chat model
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")

# It creates a conversation chain with the chat model and the memory
conversation = ConversationChain(
    llm=llm, 
    memory=memory,
)

while True:
    # Here we are taking user input
    user_input = input("\nYou: ")
    
    # Check for exit command
    if user_input.lower() in ['bye', 'exit']:
        print("Goodbye!")

        # Print the conversation history
        print(conversation.memory.buffer)
        break
    
    # Here we are getting the response from the AI
    response = conversation.predict(input=user_input)
    print("\nAI:", response)
