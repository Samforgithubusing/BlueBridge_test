# Evaluate the retriever
import json
import matplotlib.pyplot as plt
from preprocess import DocProcessing 
import os 
import argparse 

def evaluate(eval_json, md_folder, retriever, proc:DocProcessing):

  with open(eval_json, 'r') as eval:
    eval = json.load(eval)

  # get relevant chunks and retrieved chunks  
  md_files = [file for file in  os.listdir(md_folder)  if file.endswith('.md')]
  keys = [file[:-3] for file in md_files ]

  relevant_dict = {}
  retrieved_dict = {}

  for i, k  in enumerate(keys) :
    retrieved_dict[k] = [page.page_content for page in retriever.invoke(eval[k])]
    md_path = os.path.join(md_folder, md_files[i])
    relevant_dict[k] = [page.page_content for page in proc.load_and_split_file(md_path)]

  # Calculate the recall and precision
  recalls = []
  precisions = []

  for k in keys:
    rec = 0
    prec= 0
    for chunk  in relevant_dict[k]:
      if  chunk in retrieved_dict[k]:
        rec+=1
    rec /= len(relevant_dict[k])

    for chunk in retrieved_dict[k]:
      if chunk in relevant_dict[k]:
        prec+=1
    prec /= len(retrieved_dict[k])

    recalls.append(rec)
    precisions.append(prec)

  return recalls, precisions, keys

def plot(recs, precs, keys):

 # Plot Recall per document
  plt.figure(figsize=(10, 6))
  plt.bar(keys, recs, color='skyblue')
  plt.xticks(rotation=45, ha='right', fontsize=10)
  for i, v in enumerate(recs):
    plt.text(i, v + 0.01, f"{int(v*100):.0f}%", ha='center', fontweight='bold')

  plt.ylabel('Recall per document')
  plt.ylim(0, 1)

  # Plot Precision per document
  plt.figure(figsize=(10, 6))
  plt.bar(keys, precs, color='skyblue')
  plt.xticks(rotation=45, ha='right', fontsize=10)
  for i, v in enumerate(precs):
    plt.text(i, v + 0.01, f"{int(v*100):.0f}%", ha='center', fontweight='bold')

  plt.ylabel('Precision per document')
  plt.ylim(0, 1)

  plt.show()


if __name__=='__main__' :
  parser = argparse.ArgumentParser(description="Process documents and save vector store.")
  parser.add_argument('--path_to_docs', type=str, default='/content/sample_data/data',  help="Path to your folder containing .md files.")
  parser.add_argument('--save_folder', type=str, default='/content/sample_data/data/faiss_index', help="Path to save the vecor store.")
  parser.add_argument('--json_file', type=str, default='./evaluation.json', help="Path to the evaluation.json.")
  args = parser.parse_args()

  proc= DocProcessing(args.path_to_docs, args.save_folder)
  retr = proc.load_vectorstore()
  recalls, precs, names = evaluate(args.json_file, args.path_to_docs, retr, proc)
  plot(recalls, precs, names)
