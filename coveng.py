# Import libraries
import streamlit as st
import pandas as pd

import sys
import os
import shutil
from collections import namedtuple

import pandas as pd
import pyterrier as pt
from pyterrier.measures import *

if not pt.started():
    pt.init()

import onir_pt

from utils import get_dataset_index, load_config, create_index

st.session_state['data_loaded'] = False

if not st.session_state['data_loaded']:
    st.session_state['data_loaded'] = True

    # Get config params
    config = load_config("./config.yaml")

    # Init dataset
    dataset = pt.datasets.get_dataset('irds:cord19/trec-covid')
    topics = dataset.get_topics(variant='description')
    qrels = dataset.get_qrels()
    get_title_abstract = pt.text.get_text(dataset, 'title') >> pt.text.get_text(dataset, 'abstract') >> pt.apply.title_abstract(lambda r: r['title'] + '/' + r['abstract'])

    # Get indices
    index_path = config['index_base_path'] + config['experiment_name']
    index_ref = create_index(
        dataset=dataset, 
        pt_index_path=config['index_base_path'] + config['experiment_name'],
        stemmer=config['stemmer'],
        stopwords=config['stopwords']
    )
    index = get_dataset_index(index_ref)

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

    sledge = onir_pt.reranker.from_checkpoint('https://macavaney.us/scibert-medmarco.tar.gz', text_field='title_abstract', expected_md5="854966d0b61543ffffa44cea627ab63b")
    
    tfidf_pipeline = tfidf >> get_title_abstract
    bm25_pipeline = bm25 >> get_title_abstract
    sledge_pipeline = bm25 % 10 >> get_title_abstract >> sledge

# Page setup
st.set_page_config(page_title="COVENG Covid-19 Article Search Engine", page_icon="🔎", layout="wide")
st.header('COVENG Covid-19 Article Search Engine 🔎')

# Use a text_input to get the keywords
text_search = st.text_input("Enter the search query", value="")

input_col1, input_col2 = st.columns([2, 1])

with input_col1:
    # Get the result limit
    output_limit = st.slider("Limit the number of results", 1, 30, 15)

with input_col2:
    # Get the pipeline type
    pipeline_type = st.radio(
        "Select the pipeline type",
        ["TF-IDF", "BM25", "Sledge"]
    )

# Get the results based on the given query
if pipeline_type == "TF-IDF":
    pipeline = tfidf_pipeline
elif pipeline_type == "BM25":
    pipeline = bm25_pipeline
else:
    pipeline = sledge_pipeline

df_search = pipeline.search(text_search)

# Sort the results by score column in descending order
if pipeline_type == "Sledge":
    df_search = df_search.sort_values(by=['score'], ascending=False)

df_search = df_search.reset_index()[:output_limit]

# Show the cards
N_cards_per_row = 3
key = 0
if text_search:
    for n_row, row in df_search.iterrows():
        i = n_row%N_cards_per_row

        if i == 0:
            st.write("---")
            cols = st.columns(N_cards_per_row)

        # Draw the card
        with cols[n_row%N_cards_per_row]:
            key += 1
            title = row['title_abstract'].split("/")[0]

            try:
                abstract_list = row['title_abstract'].split("/")[1].split(" ")
                first_term = text_search.split(" ")[0]
                snippet_idx = abstract_list.index(first_term)
                snippet = abstract_list[snippet_idx - 100: snippet_idx + 100]
                snippet = " ".join(snippet)
                for term in text_search.split(" "):
                    snippet = snippet.replace(term, f":green[{term}]")
                st.subheader(title)
                st.markdown(f"{snippet}...")
                abstract = row['title_abstract'].split("/")[1]
                for term in text_search.split(" "):
                    abstract = abstract.replace(term, f":green[{term}]")
            except:
                abstract = row['title_abstract'].split("/")[1]
                for term in text_search.split(" "):
                    abstract = abstract.replace(term, f":green[{term}]")
                st.subheader(title)
                st.markdown(f"{abstract[:300]}...")

            expander = st.expander("Read More...")
            expander.write(abstract)
