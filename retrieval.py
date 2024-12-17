from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_pinecone import PineconeVectorStore
from langchain.prompts import PromptTemplate
from langchain import hub
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

if __name__ == "__main__":
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro")
    query = "Who constructed the Taj Mahal?"

    prompt_template = PromptTemplate(input_variables=[], template=query)

    vectorstore = PineconeVectorStore(index_name="basic-rag-project", embedding=embeddings)

    # prompt for retrieving
    prompt = hub.pull('langchain-ai/retrieval-qa-chat')

    # create_stuff_documents_chain- This chain takes list of documents and formats them into a single document and then passes it to
    # the LLM for further processing.
    combined_docs_chain = create_stuff_documents_chain(llm, prompt)

    # create_retriever_chain- This chain takes a query, passes it to the retriever, retrieves a list of documents, and then passes
    retriever_chain = create_retrieval_chain(retriever=vectorstore.as_retriever(), combine_docs_chain=combined_docs_chain)

    result = retriever_chain.invoke({ "input": query })

    print(result)