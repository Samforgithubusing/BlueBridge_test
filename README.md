##  BlueBridge_test
 
In this repository we  design a question-answering (QA) system that intelligently selects and presents the most relevant context to a local language model (LLM) under a strict token budget(1024 tokens).

The repository contains 3 mains files:
 - preprocess.py: to process .md files (parsing, chunking and embedding) and output FAISS vectore_store/retriever
 - evaluation.py: to evaluate the results of the retrievel process using precision and recall  
 - llm_chat.py: for answering questions and output a .json file answers.json 

## ⚙️ Dependencies: create your conda environment (or python virtual environment) and install requirements
If you have conda installed in your machine run the bash command:
``` bash 
conda create --name bluebridge python==3.11 
conda activate bluebridge
pip install -r requirements.txt
```
If you don't have conda, create a python virtual environment and install the requirements
``` bash 
python -m venv bluebridge
bleubridge/Scripts/activate
pip install -r requirements.txt
```
I have used the Ollama open source LLMs for this project.To install Ollama follow the link https://ollama.com/download 

Pull the embedding model and LLama3.2:3b LLM using the bash command
``` bash 
ollama pull nomic-embed-text
ollama pull llama3.2:3b 
```
## 💫 Run 
For the FAISS  vector store creation run:

``` bash
python preprocess.py --path_to_docs /path/to/your/.md --save_folder /path/to/save/FAISS  
```

For llm answering:
``` bash 
python llm_chat.py --path_to_docs /path/to/your/.md --save_folder /path/to/save/FAISS --q_file /path/questions --a_file /path/to/save/answers 
```

For the evaluation of the retriever:
``` bash 
python evaluation.py --path_to_docs /path/to/your/.md --save_folder /path/to/save/FAISS --json_file /path/to/evaluation.json
```



