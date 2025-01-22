from time import time
from beir import util, LoggingHandler
from beir.retrieval import models
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES

import logging
import pathlib, os
import random
import json

#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
#### /print debug information to stdout

import sys
model_name = sys.argv[1]
dataset = sys.argv[2]

#### Download nfcorpus.zip dataset and unzip the dataset
url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(dataset)
out_dir = os.path.join(pathlib.Path(__file__).parent.absolute(), "datasets")
data_path = util.download_and_unzip(url, out_dir)

#### Provide the data path where nfcorpus has been downloaded and unzipped to the data loader
# data folder would contain these files: 
# (1) nfcorpus/corpus.jsonl  (format: jsonlines)
# (2) nfcorpus/queries.jsonl (format: jsonlines)
# (3) nfcorpus/qrels/test.tsv (format: tsv ("\t"))

corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split="train")

#### Dense Retrieval using SBERT (Sentence-BERT) ####
#### Provide any pretrained sentence-transformers model
#### The model was fine-tuned using cosine-similarity.
#### Complete list - https://www.sbert.net/docs/pretrained_models.html

# Finetuned
if model_name == "contriever":
    model = DRES(models.SentenceBERT(f"output/facebook/contriever"))
elif model_name == "contriever-sft":
    model = DRES(models.SentenceBERT(f"output/contriever-sft-{dataset}"))
elif model_name == "e5-sft":
    model = DRES(models.SentenceBERT(f"output/e5-sft-{dataset}"))

name = model_name.split("/")[-1]
retriever = EvaluateRetrieval(model)

#### Retrieve dense results (format of results is identical to qrels)
start_time = time()
results = retriever.retrieve(corpus, queries)
end_time = time()
print("Time taken to retrieve: {:.2f} seconds".format(end_time - start_time))
#### Evaluate your retrieval using NDCG@k, MAP@K ...

top_k = 100
with open(f"stage2/{name}_{dataset}_stage2_res_top100.json", "w") as wf:
    for query_id, ranking_scores in results.items():
        scores_sorted = sorted(ranking_scores.items(), key=lambda item: item[1], reverse=True)
        docs = []
        query = queries[query_id]
        for rank in range(top_k):
            doc_id = scores_sorted[rank][0]
            docs.append(corpus[doc_id].get("text"))
        wf.write(json.dumps({"query": query, "retrieved_passages": docs})+"\n")
