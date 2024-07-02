import shutil
import os

from langchain import hub
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

VECTORE_STORE_DIR = './chroma_db'

vectoreStore = None
qa_chain = None
documents = []

# define a function to format documents
def format_docs(docs):
    return '\n\n'.join(doc.page_content for doc in docs)


def reset_vector_store():
    """
    Delete the existing vectore store directory to re intialize the embedding 
    """

    if os.path.exists(VECTORE_STORE_DIR):
        shutil.rmtree(VECTORE_STORE_DIR)


def load_documents(directory: str):
    """
    Load documents from a specified directory and set up the vectore store and !A chain.
    """

    global vectoreStore, qa_chain, documents

    try:
        # Reset the vector store
        reset_vector_store()

        #Load documents from the specified directory
        loader = DirectoryLoader(directory,glob='**/*.pdf')
        documents = loader.load()

        # Create an embedding function and a vector store
        embedding_function = OllamaEmbeddings(model = 'llama2')

        vectoreStore = Chroma(
            persist_directory=VECTORE_STORE_DIR,
            embedding_function=embedding_function
        )

        # Add document to the vector store
        vectoreStore.add_documents(documents)

        # Intialize the language model
        llm = Ollama(model = 'llama2')

        # Create a retriver from the vector store
        retriver = vectoreStore.as_retriever()

       
        # Pull the RAG propmpt
        rag_prompt = hub.pull('rlm/rag-prompt')


        # Set up the QA chain components properly
        qa_chain = (
            {
                'context': retriver | format_docs,
                'question': RunnablePassthrough()
            }
            | rag_prompt
            | llm
            | StrOutputParser()
        )

        return f'Loaded {len(documents)} documents.'
    
    except Exception as e:
        raise RuntimeError(f'Error loading documents: {e}')


def ask_question(question: str):
    """
    Ask a question and get an answer based on the loaded focuments.
    """
    global qa_chain, documents
    if qa_chain is None:
        raise RuntimeError('Documents are not loaded. Please load documents first.')
    

    try:
        # Retriver and format relevant documents
        retriver = vectoreStore.as_retriever()
        context_docs = retriver.invoke({
            'query':question
        })
        formatted_context = format_docs(context_docs)
        
        #Invoke the QA chain
        answer = qa_chain.invoke({
            'context': formatted_context,
            'question': question
        })

        return answer

    except Exception as e:
        raise RuntimeError(f'Error asking question: {e}')
    