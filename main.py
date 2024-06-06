import sys
import os
import shutil
from collections import namedtuple

import pandas as pd
import pyterrier as pt
from pyterrier.measures import *

pt.init()

import onir_pt

from utils import get_dataset_index, load_config, create_index

"""
Resources:
- https://pyterrier.readthedocs.io/en/latest/datasets.html
- https://github.com/Georgetown-IR-Lab/OpenNIR
- https://github.com/terrier-org/cikm2021tutorial/blob/main/notebooks/notebook4_2.ipynb
- https://github.com/terrier-org/cikm2021tutorial/blob/main/notebooks/notebook3.ipynb
"""

# Get config params
config = load_config("./config.yaml")

# Init dataset
dataset = pt.datasets.get_dataset('irds:cord19/trec-covid')
topics = dataset.get_topics(variant='description')
qrels = dataset.get_qrels()
get_title_abstract = pt.text.get_text(dataset, 'title') >> pt.text.get_text(dataset, 'abstract') >> pt.apply.title_abstract(lambda r: r['title'] + ' ' + r['abstract'])

# Get indices
index_path = config['index_base_path'] + config['experiment_name']
index_ref = create_index(
    dataset=dataset, 
    pt_index_path=config['index_base_path'] + config['experiment_name'],
    stemmer=config['stemmer'],
    stopwords=config['stopwords']
)
index = get_dataset_index(index_ref)

# Setup the pipeline
term_pipe = []
if config['stopwords'] == "terrier":
    term_pipe.append("Stopwords")
if config['stemmer'] == "porter":
    term_pipe.append("PorterStemmer")

properties = {"termpipelines" : ",".join(term_pipe)}

# Init retrieval objects
if config['qe_model'] == "none":
    bm25 = pt.BatchRetrieve(index, wmodel="BM25", properties=properties)
    tfidf = pt.BatchRetrieve(index, wmodel="TF_IDF", properties=properties)
else:
    controls={"qe":"on", "qemodel" : config['qe_model']}
    bm25 = pt.BatchRetrieve(index, wmodel="BM25", properties=properties, controls=controls)
    tfidf = pt.BatchRetrieve(index, wmodel="TF_IDF", properties=properties, controls=controls)

# Setup language models for reranking
vbert = onir_pt.reranker('vanilla_transformer', 'bert', text_field='title_abstract', vocab_config={'train': True})
sledge = onir_pt.reranker.from_checkpoint('https://macavaney.us/scibert-medmarco.tar.gz', text_field='title_abstract', expected_md5="854966d0b61543ffffa44cea627ab63b")

bert_pipeline = bm25 >> get_title_abstract >> vbert
sledge_pipeline = bm25 >> get_title_abstract >> sledge

# Start Experiment
cutoffs = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
results = pt.Experiment(
    [tfidf, bm25, bert_pipeline, sledge_pipeline] + [bm25 % cutoff >> get_title_abstract >> vbert for cutoff in cutoffs] + [bm25 % cutoff >> get_title_abstract >> sledge for cutoff in cutoffs],
    topics,
    qrels,
    eval_metrics=["map", AP(rel=2), P(rel=2)@10, nDCG, nDCG@10, nDCG@100, 'mrt'],
    names=["TF_IDF", "BM25", "BM25 >> BERT", "BM25 >> SLEDGE"] + [f'BM25 >> BERT c={cutoff}' for cutoff in cutoffs] + [f'BM25 >> SLEDGE c={cutoff}' for cutoff in cutoffs]
)

# Save the results
results.to_csv(f"./results/{config['experiment_name']}.csv", index=False)
