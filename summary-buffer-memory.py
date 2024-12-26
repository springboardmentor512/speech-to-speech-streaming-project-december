from langchain.chains.conversation.memory import ConversationSummaryBufferMemory
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import ConversationChain

# Initialize the chat model
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")

# It creates a summary buffer memory that keeps a summary of old messages and recent messages in buffer
memory = ConversationSummaryBufferMemory(
    llm=llm,
    max_token_limit=200, # Maximum number of tokens to keep in memory
)

# It creates a conversation chain with the chat model and summary buffer memory
conversation = ConversationChain(
    llm=llm,
    memory=memory,
    verbose=True # To see the conversation flow
)

while True:
    # Here we are taking user input
    user_input = input("\nYou: ")
    
    # Check for exit command
    if user_input.lower() in ['bye', 'exit']:
        print("Goodbye!")

        # Print the conversation history with summaries
        print("\nConversation Summary:")
        print(conversation.memory.buffer)
        break
    
    # Here we are getting the response from the AI
    response = conversation.predict(input=user_input)
    print("\nAI:", response)
