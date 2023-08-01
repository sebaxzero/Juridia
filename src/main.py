import os
import logging
import urllib.request
from typing import Any, Dict, List, Union, Literal

from langchain.document_loaders import DirectoryLoader, PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import Chroma
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler #for streaming
from langchain.llms import LlamaCpp, OpenAI, TextGen
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory

from tempfile import NamedTemporaryFile
logging.basicConfig(level=logging.CRITICAL)

allowedllm = Literal["OpenAI", "TextGen", "LlamaCpp"]

class Loader:
    @staticmethod
    def load_local(directory:str = './Documents'):
        PDFloader = DirectoryLoader(directory, glob='./*.pdf', loader_cls=PyPDFLoader)
        Textloader = DirectoryLoader(directory, glob='./*.txt', loader_cls=TextLoader)
        return PDFloader.load()+Textloader.load()

    @staticmethod
    def load_cloud(file_type: str, file_content):
        logging.debug(f'file type: {file_type}')
        if file_type == "text/plain":
            Loader = TextLoader
        elif file_type == "application/pdf":
            Loader = PyPDFLoader
        else:
            logging.error(f'file type not supported')
            raise ValueError("load error") 
        
        with NamedTemporaryFile(mode='wb', delete=False) as f:
            f.write(file_content)
            temp_file_path = os.path.abspath(f.name)
        logging.debug(f'temporary file saved in {temp_file_path}')
        pdf_reader = Loader(temp_file_path)
        return pdf_reader.load()
             
class Splitter:
    @staticmethod
    def split(documents: Union[str, list], chunk_size: int = 512, chunk_overlap: int = 0):
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        return text_splitter.split_documents(documents=documents)

class Embeddings:
    @staticmethod
    def get_dict():
        return [{'sentence-transformers/all-MiniLM-L6-v2': 'Sentence Transformers'}, {'hkunlp/instructor-base':'Instructor Embedding'}]

    @staticmethod
    def get(model_name: str, device: str = 'cpu'):
        model_dict = Embeddings.get_dict()
        for model in model_dict:
            if model_name in model:
                model_type = model[model_name]
                if model_type == 'Sentence Transformers':
                    logging.debug(f'using {model_name} as {model_type}')
                    return HuggingFaceEmbeddings(model_name=model_name, model_kwargs={'device': device})
                elif model_type == 'Instructor Embedding':
                    logging.debug(f'using {model_name} as {model_type}')
                    return HuggingFaceInstructEmbeddings(model_name=model_name, model_kwargs={'device': device})
                else:
                    logging.error(f'{model_type} in {model} not supported')
                    raise ValueError("Embeddings error")
        else:
            logging.error(f'{model_name} not found in model dictionary')
            raise ValueError("Embeddings error")

class VectorStore:    
    @staticmethod
    def create(persist_directory, splitted_docs, embeddings):
        return Chroma.from_documents(documents=splitted_docs, embedding=embeddings, persist_directory=persist_directory)
    
    @staticmethod
    def create_Cloud(file_type, file_content, chunk_size:int = 512, chunk_overlap:int = 0, k:int = 2, model_name: str = 'sentence-transformers/all-MiniLM-L6-v2', device: str = 'cpu'):
        documents = Loader.load_cloud(file_type=file_type, file_content=file_content)
        texts = Splitter.split(documents=documents, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        embeddings = Embeddings.get(model_name=model_name, device=device)
        return Chroma.from_documents(texts, embeddings)
    
    @staticmethod
    def load(name, embeddings):
        index_dir = f'./Sessions/{name}/Index'
        os.makedirs(index_dir, exist_ok=True)
        logging.info(f'loading an existing Chroma database from: "{index_dir}"...')
        return Chroma(persist_directory=index_dir, embedding_function=embeddings)
    
    @staticmethod
    def retriever(db, k:int = 2):
        retriever = db.as_retriever()
        retriever.search_kwargs["k"] = k
        retriever.search_kwargs["distance_metric"] = "cos"
        return retriever
    
    @staticmethod
    def add(splitted_docs, db):
        logging.info('Docs found in Documents folder, adding to existing db')
        texts = [doc.page_content for doc in splitted_docs]
        metadatas = [doc.metadata for doc in splitted_docs]
        db.add_texts(texts=texts, metadatas=metadatas)
        return db
        
    @staticmethod
    def get(name:str = "example", chunk_size:int = 512, chunk_overlap:int = 0, model_name: str = 'sentence-transformers/all-MiniLM-L6-v2', device: str = 'cpu'):
        doc_dir = './Documents'
        os.makedirs(doc_dir, exist_ok=True)
        index_dir = f'./Sessions/{name}/Index'
        os.makedirs(index_dir, exist_ok=True)
        
        embeddings = Embeddings.get(model_name=model_name, device=device)
        
        if len(os.listdir(index_dir)) >= 1:
            logging.debug(f'files found in {index_dir}')
            vectorstore = VectorStore.load(name=name, embeddings=embeddings)
            if len(os.listdir(doc_dir)) >= 1:
                logging.debug(f'files found in {doc_dir}')
                documents = Loader.load_local(directory=doc_dir)
                splitted_docs = Splitter.split(documents=documents, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
                vectorstore = VectorStore.add(splitted_docs=splitted_docs, db=vectorstore)
                
                logging.debug(f'deleting docs in {doc_dir}')
                file_list = os.listdir(doc_dir)
                for file_name in file_list:
                    file_path = os.path.join(doc_dir, file_name)
                    if os.path.isfile(file_path):
                        os.remove(file_path)
        else:
            logging.debug(f'no files found in {index_dir}')
            
            if not os.listdir(doc_dir):
                raise ValueError("no doc files") 
                
            documents = Loader.load_local(directory=doc_dir)    
            splitted_docs = Splitter.split(documents=documents, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            vectorstore = VectorStore.create(persist_directory=index_dir, splitted_docs=splitted_docs, embeddings=embeddings)
            
            logging.debug(f'deleting docs in {doc_dir}')
            file_list = os.listdir(doc_dir)
            for file_name in file_list:
                file_path = os.path.join(doc_dir, file_name)
                if os.path.isfile(file_path):
                    os.remove(file_path)
                    
        vectorstore.persist()
        
        return vectorstore
  
class Weights:
    @staticmethod
    def get_dict(): #list of ggml {repo_id:filename} for model weights to use with llama.cpp
        return [{'TheBloke/vicuna-13b-v1.3.0-GGML':'vicuna-13b-v1.3.0.ggmlv3.q2_K.bin'},
                {'TheBloke/WizardLM-13B-V1.1-GGML':'wizardlm-13b-v1.1.ggmlv3.q2_K.bin'},
                {'TheBloke/vicuna-7B-v1.3-GGML':'vicuna-7b-v1.3.ggmlv3.q2_K.bin'}]
        
    @staticmethod
    def get(model: str): #download models weights to ./Models/
        model_dict = None
        for dictionary in Weights.get_dict():
            if model in dictionary.values():
                model_dict = dictionary
                break

        if model_dict is not None:
            repo_id, name = list(model_dict.keys())[0], list(model_dict.values())[0]
            os.makedirs('./Models', exist_ok=True)
            filename = f'./Models/{name}'
            if not os.path.isfile(filename):
                url = f'https://huggingface.co/{repo_id}/resolve/main/{name}'
                urllib.request.urlretrieve(url=url, filename=filename)
                print("File downloaded successfully.")
            else:
                print("File already exists.")
        else:
            print("Model path not found in the dictionary.")
            
class LLM:
    @staticmethod
    def get(llm:allowedllm='TextGen',
            OpenAi_Key: str ='sk-111111111111111111111111111111111111111111111111',
            OpenAi_Model: str = 'text-davinci-003',
            OpenAi_Host: str = 'https://api.openai.com/v1',
            local_Host: str = 'http://127.0.0.1:5000',
            temperature: float = 0.1,
            top_p: float = 0.1,
            max_tokens: int = 2048,
            top_k: int = 40,
            stopping_strings = ['### System:', '### User:', '\n\n'],
            model_path: str = './Models/model.bin',
            ):
        
        if llm=='OpenAI':
            return OpenAI(
                model_name=OpenAi_Model,
                openai_api_base=OpenAi_Host,
                openai_api_key=OpenAi_Key,
                streaming=False,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                )
        if llm=='TextGen':
            return TextGen(
                model_url=local_Host,
                temperature=temperature,
                max_new_tokens=max_tokens,
                top_p=top_p,
                top_k=top_k,
                stopping_strings=stopping_strings,
                
                )
        if llm=='LlamaCpp':
            return LlamaCpp(
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                top_k=top_k,
                stop=stopping_strings,
                model_path=model_path
                )

class Template:    
    @staticmethod
    def get(sys_name: str = '### System:', system:str = 'Esta es una conversación entre un usuario y un asistente de inteligencia virtual llamado juridIA, especializado en derecho de Chile.', 
            user_name:str='### User:',user:str ='Dado el siguiente diálogo y una pregunta de seguimiento, reformula la pregunta de seguimiento para que sea una pregunta independiente, no respondas la pregunta, solo reformula. si no hace referencia al dialogo, no alteres la pregunta, solo responde con la pregunta sin texto adicional.', 
            input_name:str='### Input:',input:str ='Chat History:\n{chat_history}\nPregunta de seguimiento: {question}', 
            res_name:str='### Response:',response:str =''):
        return f'{sys_name}\n{system}\n\n{user_name}\n{user}\n\n{input_name}\n{input}\n\n{res_name}\n{response}'
    
    @staticmethod
    def getQA(sys_name: str = '### System:',system:str = 'Esta es una conversación entre un usuario y un asistente de inteligencia virtual llamado juridIA, especializado en derecho de Chile.', 
            user_name:str='### User:',user:str ='Utiliza los siguientes fragmentos de contexto para responder la pregunta. Si no sabes la respuesta, simplemente di que no lo sabes, no intentes inventar una respuesta.', 
            input_name:str='### Input:',input:str ='Pregunta: {question}\nFragmentos de contexto:\n{context}\n', 
            res_name:str='### Response:',response:str =''):
        return f'{sys_name}\n{system}\n\n{user_name}\n{user}\n\n{input_name}\n{input}\n\n{res_name}\n{response}'

class AnswerConversationBufferMemory(ConversationBufferMemory):
    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, str]) -> None:
        return super(AnswerConversationBufferMemory, self).save_context(inputs,{'response': outputs['answer']})

class Memory:
    @staticmethod 
    def get(memory_key:str ="chat_history"):
        return AnswerConversationBufferMemory(memory_key=memory_key, return_messages=True)

class Chain:
    @staticmethod 
    def get_no_mem(llm, retriever, condense_question_prompt = PromptTemplate.from_template(template=Template.get()), qa_prompt = PromptTemplate.from_template(template=Template.getQA())):
        return ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever,
            condense_question_prompt=condense_question_prompt,
            return_source_documents=True,
            combine_docs_chain_kwargs={"prompt": qa_prompt},
            rephrase_question = True,
            verbose=False
        )
        
    @staticmethod 
    def Get(llm, retriever, memory, condense_question_prompt = PromptTemplate.from_template(template=Template.get()), qa_prompt = PromptTemplate.from_template(template=Template.getQA())):
        return ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever,
            condense_question_prompt=condense_question_prompt,
            return_source_documents=True,
            combine_docs_chain_kwargs={"prompt": qa_prompt},
            rephrase_question = True,
            memory=memory,
            verbose=False
        )
    
    @staticmethod 
    def process_response(res):
        answer = res["answer"]
        source_documents = {}
    
        for document in res['source_documents']:
            page_content = document.page_content
            source = document.metadata['source']
            page = document.metadata['page']
            document_string = f'contenido: "{page_content}"'
            if source not in source_documents:
                source_documents[source] = {}
            source_documents[source][page] = document_string
    
        return answer, source_documents

    @staticmethod 
    def query(prompt: str, chain):
        res = chain({"question" : prompt})
        answer, source_documents = Chain.process_response(res=res)
        return answer, source_documents

    @staticmethod 
    def query_no_mem(prompt: str, chain, chat_history):
        res = chain({"question" : prompt, "chat_history": chat_history})
        answer, source_documents = Chain.process_response(res=res)
        return answer, source_documents

if __name__ == "__main__":
    memory = Memory.get()
    llm = LLM.get(llm='TextGen')
    retriever = VectorStore.retriever(k=3, db=VectorStore.get())
    while True:
        prompt :str = input('ingrese consulta: ')
        answer, source_documents = Chain.query(prompt=prompt, chain=Chain.Get(llm=llm, retriever=retriever, memory=memory)) 
        print("Respuesta:","\n",answer,"\n\n")
        print("Documentos de fuente:","\n\n")
        for source, pages in source_documents.items():
            print(f"Fuente: {source}")
            for page, content in pages.items():
                print(f"Página {page}:\n\n{content}\n\n")

