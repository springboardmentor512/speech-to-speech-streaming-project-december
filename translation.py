from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts.prompt import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os   
from dotenv import load_dotenv

load_dotenv()

if __name__ == "__main__":

    summary_prompt = """
    You are a translater where you will be given input text
    {input}
    you need to translate that input text into desired language i.e {convertLanguage}    
    """

    prompt_tempalate = PromptTemplate(input_variables=["input", "convertLanguage"], template=summary_prompt)

    llm = ChatGoogleGenerativeAI(
        model='gemini-1.5-flash',
        api_key=os.getenv('GOOGLE_API_KEY')
    )

    chain = prompt_tempalate | llm | StrOutputParser()

    res = chain.invoke({ "input": "My name is Aayush and I live in Delhi.", "convertLanguage": "Spanish" });

    print(res)