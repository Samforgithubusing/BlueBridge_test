# prepare the questions
import json
from langchain_ollama.llms import OllamaLLM
from preprocess import DocProcessing 
import argparse 
import tiktoken 

llm  = OllamaLLM(model="llama3.2:3b") # The open source LLM used

# get questions within a python dictionary 
def get_questions(file):
  out  ={}
  with open(file ,'r') as questions:
     q =json.load(questions)
  qs =q['questions']

  for question in qs :
    out[question['id']]=question["question"]
  return out

# count tokens to raise an exeption 

def verify_tokens_count(doc_string):
  tokenizer = tiktoken.get_encoding(encoding_name="cl100k_base")
  tokens = tokenizer.encode(doc_string)
  num_tokens =len(tokens)

  if num_tokens > 1024:
    raise ValueError(f"Context too long: {num_tokens} tokens (limit is {1024})")

  return num_tokens

# Chat with the LLama3.2:3b using the save retriever/vector store
def rag_bot(question: str ,retriever) -> dict:

    docs = retriever.invoke(question)
    docs_string = "".join(doc.page_content for doc in docs)

    instructions = f"""You are a helpful assistant who is good at analyzing source information
    and answering questions. Use the following source documents to answer the user's questions.
    Be as concise as possible in your answers.
    Documents:
    {docs_string}"""

    try:
      verify_tokens_count(docs_string)
      ai_msg = llm.invoke([
              {"role": "system", "content": instructions},
              {"role": "user", "content": question},
          ],
      )
      return {"answer": ai_msg, "documents": [doc.page_content for doc in  docs]}
    
    except ValueError as e:
       print('Error:', e)


if __name__=='__main__':
    parser = argparse.ArgumentParser(description="Respond to question using documents.")
    parser.add_argument('--path_to_docs', type=str, default='/content/sample_data/data',  help="Path to your folder containing .md files.")
    parser.add_argument('--save_folder', type=str, default='/content/sample_data/data/faiss_index', help="Path to save the vecor store.")
    parser.add_argument('--q_file', type=str, default='./questions.json',  help="Path to your json file questions")
    parser.add_argument('--a_file', type=str, default='./answers.json',  help="Path to save your answers")
     
    args = parser.parse_args()
    proc = DocProcessing(args.path_to_docs, args.save_folder)
    retriever = proc.load_vectorstore()

    questions = get_questions(args.q_file)
    answers ={}

    for id , q in questions.items():
      answers[id]=rag_bot(q, retriever)
    with open(args.a_file , 'w') as ans:
      json.dump(answers , ans ,indent=2)

