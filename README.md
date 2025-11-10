# BlueBridge_test
This repository is created for a job interview test with the company blue bridge ai.

It contains 3 mains files:
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
I have used the Ollama open source LLMs for this project.To install Ollama follow the link 

Pull the embedding model and LLama3.2:3b LLM using the command
``` bash 
ollama pull nomic-embed-text
ollama pull llama3.2:3b 
```