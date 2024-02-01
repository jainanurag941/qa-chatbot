#importing the required modules and dependencies
import logging
from dotenv import load_dotenv
from langchain.document_loaders import PyPDFLoader, Docx2txtLoader, UnstructuredFileLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
import openai
import numpy
from pymilvus import (connections, utility, FieldSchema, CollectionSchema, DataType, Collection)
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Milvus
import tiktoken
from langchain.chains.question_answering import load_qa_chain
from langchain import OpenAI, PromptTemplate, LLMChain
from langchain.chains.mapreduce import MapReduceChain
from langchain.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
import textwrap
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain import PromptTemplate
from langchain.callbacks import get_openai_callback

#function to load the api key from env file
def get_api_key():
    load_dotenv()
    my_api_key=os.getenv("API_KEY")
    return my_api_key

#function to configure the logger
def configure_logger():
    # Create and configure logger
    logging.basicConfig(filename="logs/log_file.log", format='%(asctime)s %(message)s', filemode='w')
    
    # Creating an object
    logger = logging.getLogger()
    
    # Setting the threshold of logger to DEBUG
    logger.setLevel(logging.DEBUG)

    return logger


#function to load various files according to their format
def read_file_content(myfilepath, file_extension):
    # loader = UnstructuredFileLoader(myfilepath)
    # myLoadedDocs = loader.load()
    # docs = myLoadedDocs[0].page_content.replace("\n", "")
    # return docs
    loadedFileContent = ""
    if file_extension == "docx":
        loader = Docx2txtLoader(myfilepath)
        loadedFileContent = loader.load()
    elif file_extension == "pdf":
        doc_loader = PyPDFLoader(myfilepath)
        loadedFileContent = doc_loader.load()
    elif file_extension == "txt":
        loader = TextLoader(myfilepath)
        loadedFileContent = loader.load()

    return loadedFileContent

#function to create chunks from loaded documents
def create_chunks(my_docs):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 50)
    chunks = text_splitter.split_documents(my_docs)
    return chunks

#This functions finds the summary of the loaded documents and prints the summary
def find_summary(final_chunks):
    os.environ['OPENAI_API_KEY'] = api
    llm = OpenAI(temperature=0, model_name='text-davinci-003')
    with get_openai_callback() as cb:
        chain = load_summarize_chain(llm, chain_type="map_reduce", verbose = True)
        output_summary = chain.run(final_chunks)
        print("\n")
        print(f"Total Cost (USD): ${cb.total_cost}")
        print(f'Spent a total of {cb.total_tokens} tokens')
        print(f"Prompt Tokens: {cb.prompt_tokens}")
        print(f"Completion Tokens: {cb.completion_tokens}")
        print("\n")
        wrapped_text = textwrap.fill(output_summary, width=100)
        print(wrapped_text)

#This function asks user to select a pdf file, or a txt file or a docx file, creates chunks and returns them
def file_to_interact():
    file_paths = ["/home/anuragjain97/Desktop/generative-ai-assignment-jainanurag95-1686736948770/my_files/electric-vehicle-methods.pdf", "/home/anuragjain97/Desktop/generative-ai-assignment-jainanurag95-1686736948770/my_files/nature.txt", "/home/anuragjain97/Desktop/generative-ai-assignment-jainanurag95-1686736948770/my_files/richieRich.docx", "/home/anuragjain97/Downloads/EmployeeHandbook.pdf"]
    file = ["1- Electric-Vehicle-Methods.pdf", "2- Nature.txt", "3- Richie-Rich.docx", "4-Employee Handbook"]
    for name in file:
        print(name)

    option = int(input("Enter the file you want to interact with: "))
    if option > len(file_paths):
        logger.error("File does not exists")
        raise ValueError('Chosen wrong value')
    # print(option)
    # print(file[option-1])

    loaded_file_path = file_paths[option-1]
    file_extension = loaded_file_path.split('.')[-1]
    # print(loaded_file_path)
    # print(file_extension)

    loaded_file_content = read_file_content(loaded_file_path, file_extension)

    final_chunks = create_chunks(loaded_file_content)

    # print(final_chunks)
    # print(loaded_file_content)
    return final_chunks

#This function calls the find_summary function to print the final summary
def get_summary():
    final_chunks = file_to_interact()

    summary = find_summary(final_chunks)

#This function allows user to ask questions to chatbot and get the appropriate answer
def get_qna(mychunkcontent):
    os.environ['OPENAI_API_KEY'] = api
    #creating embeddings from chunks
    embeddings = OpenAIEmbeddings()

    vector_db = Milvus.from_documents(
        mychunkcontent,
        embeddings,
        connection_args={"host": "127.0.0.1", "port": "19530"},
        drop_old = True
    )

    query_template = """
    Use the context provided below and the chat history to answer the question:
    {context}

    {history}

    {question}
    Answer:
    """
    prompt = PromptTemplate(
        input_variables=["history", "context", "question"],
        template=query_template,
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=ChatOpenAI(model_name='gpt-3.5-turbo', temperature=0.3),
        chain_type='stuff',
        retriever=vector_db.as_retriever(search_type="similarity", search_kwargs={"k":5}),
        verbose=True,
        chain_type_kwargs={
            "verbose": True,
            "prompt": prompt,
            "memory": ConversationBufferMemory(
                memory_key="history",
                input_key="question"),
        }
    )

    choice = int(input("Enter 1 to ask questions or enter 0 to exit: "))
    while choice != 0:
        your_query = input("Enter your question: ")
        with get_openai_callback() as cb:
            print(qa_chain.run({"query": your_query}))
            print("\n")
            print(f"Total Cost (USD): ${cb.total_cost}")
            print(f'Spent a total of {cb.total_tokens} tokens')
            print(f"Prompt Tokens: {cb.prompt_tokens}")
            print(f"Completion Tokens: {cb.completion_tokens}")
            print("\n")

        choice = int(input("Enter 1 to ask questions or enter 0 to exit: "))

    print(qa_chain.combine_documents_chain.memory)

if __name__ == '__main__':
    api = get_api_key()
    logger = configure_logger()

    choice = int(input("Enter 1 to get summary, Enter 2 to interact with chatbot: "))
    if choice == 1:
        summary = get_summary()
    elif choice == 2:
        chunks = file_to_interact()
        # print(chunks)
        QandA = get_qna(chunks)
    else:
        logger.error("Wrong choice selected")
        raise ValueError('Please enter a valid choice')
    
    