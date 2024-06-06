import os
import yaml

import pyterrier as pt


def load_config(config_path):
    with open(config_path) as stream:
        config = yaml.safe_load(stream)

    return config


def create_index(dataset, pt_index_path, stemmer, stopwords):
    if not os.path.exists(pt_index_path + "/data.properties"):
        # create the index, using the IterDictIndexer indexer
        indexer = pt.index.IterDictIndexer(pt_index_path,
                                           stemmer=stemmer,
                                           stopwords=stopwords)
        
        # we give the dataset get_corpus_iter() directly to the indexer
        # while specifying the fields to index and the metadata to record
        index_ref = indexer.index(dataset.get_corpus_iter(), fields=('title', 'abstract',), meta=('docno',))

    else:
        # if you already have the index, use it.
        index_ref = pt.IndexRef.of(pt_index_path + "/data.properties")

    return index_ref


def get_dataset_index(index_ref):
    index = pt.IndexFactory.of(index_ref)
    return index


def get_dataset_statistics(index_ref):
    print(index_ref.getCollectionStatistics())
