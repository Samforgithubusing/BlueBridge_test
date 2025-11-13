
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_text_splitters import CharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document
import os
import argparse

class DocProcessing():
  def __init__(self,
               path_to_docs:str,                     # path to the folder containing .md files
               save_folder:str,                      # path to the folder to save 
               embed_model_name="nomic-embed-text",  # embedding model 
               doc_type:str ='.md',
               chunk_size=512,
               chunk_overlap=50,
               max_tokens = 1024,                    # context constraint
               splitter_type='recursive',            # recursive or token like splitter 
               save_vector_store=True,
               ):

    self.path_to_docs = path_to_docs
    self.chunk_size = chunk_size
    self.chunk_overlap = chunk_overlap
    self.k = max_tokens//chunk_size                  # The number of retrieved chunks calculated to respect the limit of 1024 tokens
    self.embedding_name = embed_model_name
    self.extension = doc_type
    self.splitter_type =splitter_type
    self.save = save_vector_store
    self.save_folder = save_folder
    self.embeddings = OllamaEmbeddings(model=self.embedding_name)


  def text_splitter(self, docs):
    if self.splitter_type == 'recursive':
      text_splitter = RecursiveCharacterTextSplitter(chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)

    if self.splitter_type == 'token':
      text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
      encoding_name="cl100k_base", chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap )

    texts = text_splitter.split_text(docs)
    return texts

  def load_and_split_file(self, file_path):
    content = [ ]
    loader = UnstructuredMarkdownLoader(file_path, encoding="utf-8")
    document = loader.load()
    for page in document:
      txts=self.text_splitter(page.page_content)
      Docs =[Document(page_content=t) for t in txts]
      content += Docs
    return content 

  def load_and_split_folder(self):
    content=[]
    files = [file for file in os.listdir(self.path_to_docs) if file.endswith(self.extension)]
    for file in files:
      file_path=os.path.join(self.path_to_docs, file)
      content+=self.load_and_split_file(file_path)
    return  content
  
  def save_vectorstore(self, texts):
    vectorstore = FAISS.from_documents(texts, embedding=self.embeddings)
    if self.save :
      vectorstore.save_local(self.save_folder)
    else :
      return vectorstore

  def load_vectorstore(self):
    vectorstore = FAISS.load_local(self.save_folder, self.embeddings, allow_dangerous_deserialization=True)
    retriever = vectorstore.as_retriever(search_kwargs={"k": self.k})
    return retriever


if __name__ =='__main__':

  parser = argparse.ArgumentParser(description="Process documents and save vector store.")
  parser.add_argument('--path_to_docs', type=str, default='/content/sample_data/data',  help="Path to your folder containing .md files.")
  parser.add_argument('--save_folder', type=str, default='/content/sample_data/data/faiss_index', help="Path to save the vecor store.")
  args = parser.parse_args()

  path_to_docs = args.path_to_docs
  save_folder = args.save_folder
  Processing  = DocProcessing(path_to_docs, save_folder)
  texts_chunks = Processing.load_and_split_folder()
  Processing.save_vectorstore(texts_chunks)

