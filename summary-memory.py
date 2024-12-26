from langchain.chains.conversation.memory import ConversationSummaryMemory
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import ConversationChain

# Initialize the chat model
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")

# It creates a summary memory for the conversation that will store summaries of past interactions
memory = ConversationSummaryMemory(
    llm=llm,
    return_messages=True,
    max_token_limit=200, # Maximum number of tokens to keep in memory
)

# It creates a conversation chain with the chat model and the summary memory
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
        
        # Print the summarized conversation history
        print(conversation.memory.buffer)
        break
    
    # Here we are getting the response from the AI
    response = conversation.predict(input=user_input)
    print("\nAI:", response)
