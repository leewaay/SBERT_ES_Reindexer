import argparse
import logging
from functools import reduce
from typing import List, Dict, Any, Optional

import numpy as np
from elasticsearch import Elasticsearch, helpers
from sentence_transformers import SentenceTransformer


logger = logging.getLogger(__name__)


class ElasticsearchVectorIndexer:
    """
    Class for indexing sentence embeddings in Elasticsearch using Sentence Transformers
    """

    def __init__(
        self, 
        host: str, 
        port: int, 
        sbert_model_name: str, 
        username: Optional[str] = None, 
        password: Optional[str] = None
    ):
        """
        Initializes the ElasticsearchVectorIndexer class

        Args:
            host (str): Elasticsearch host name
            port (int): Elasticsearch port number
            sbert_model_name (str): Name of the SentenceTransformer model
            username (Optional[str]): Elasticsearch username for authentication. Default is None
            password (Optional[str]): Elasticsearch password for authentication. Default is None
        
        """
        if username and password:
            self.es = Elasticsearch([{"host": host, "port": port, "http_auth": (username, password)}])
        else:
            self.es = Elasticsearch([{"host": host, "port": port}])
        
        self.model = SentenceTransformer(sbert_model_name)
        logger.info(f"Initialized with SBERT model: {sbert_model_name}")

    def get_data_from_es(
        self, 
        index: str, 
        query_size: int = 1000, 
        embedding_text_field: List[str] = ['text']
    ) -> List[Dict[str, Any]]:
        """
        Fetches data from Elasticsearch.

        Args:
            index (str): Name of the Elasticsearch index to fetch data from.
            query_size (int, optional): Number of documents to fetch. Defaults to 1000.
            embedding_text_field (List[str], optional): Path to the text field in the data for embedding. Defaults to ['text'].

        Returns:
            List[Dict[str, Any]]: List of documents from Elasticsearch.
        
        """
        logger.info(f"Fetching {query_size} documents from index: {index}")
        query = {
            "size": query_size,
            "_source": ".".join(embedding_text_field),
            "query": {
                "match_all": {}
            }
        }
        res = self.es.search(index=index, body=query)
        logger.info(f"Retrieved {len(res['hits']['hits'])} documents from index: {index}")
        return res['hits']['hits']

    def embed_data(
        self, 
        data: List[Dict[str, Any]], 
        embedding_text_field: List[str] = ['text']
    ) -> List[List[float]]:
        """
        Generates embeddings for the given data

        Args:
            data (List[Dict[str, Any]]): Data for which embeddings are to be generated
            embedding_text_field (List[str], optional): Path to the text field in the data for embedding. Defaults to ['text']

        Returns:
            List[List[float]]: List of embeddings
        
        """
        sentences = [reduce(lambda d, key: d[key], embedding_text_field, item['_source']) for item in data]
        logger.info(f"Generating embeddings for {len(sentences)} sentences")
        embeddings = self.model.encode(sentences)
        logger.info("Embeddings generated successfully")
        return embeddings

    def create_index_with_mapping(self, index: str) -> None:
        """
        Creates a new index with the appropriate mapping for embeddings.

        Args:
            index (str): Name of the Elasticsearch index to be created.
        
        """
        logger.info(f"Creating index with mapping for index: {index}")
        if not self.es.indices.exists(index=index):
            mapping = {
                "mappings": {
                    "properties": {
                        "embedding": {
                            "type": "dense_vector",
                            "dims": self.model.get_sentence_embedding_dimension()
                        }
                    }
                }
            }
            self.es.indices.create(index=index, body=mapping)
            logger.info(f"Created index: {index} successfully")
    
    def index_data(
        self, 
        target_index: str, 
        data: List[Dict[str, Any]], 
        embeddings: List[List[float]], 
        embedding_text_field: List[str] = ['text'], 
        decimal_precision: int = 5, 
        update: bool = True
    ) -> None:
        """
        Indexes the embeddings into Elasticsearch

        Args:
            target_index (str): Name of the target Elasticsearch index
            data (List[Dict[str, Any]]): Original data documents
            embeddings (List[List[float]]): Embeddings to be indexed
            embedding_text_field (List[str], optional): Path to the text field in the data. Defaults to ['text']
            decimal_precision (int, optional): Number of decimal places for the embedding values. Defaults to 5
            update (bool, optional): Whether to update existing documents or index new ones. Defaults to True
        
        """
        logger.info(f"Indexing data into target index: {target_index}")
        if not self.es.indices.exists(index=target_index):
            self.create_index_with_mapping(target_index)
        
        actions = []
        for i, item in enumerate(data):
            if update:
                action = {
                    "_op_type": "update",
                    "_index": target_index,
                    "_id": item["_id"],
                    "doc": {
                        "embedding": np.round(embeddings[i].tolist(), decimal_precision).tolist(),
                    }
                }
            else:
                action = {
                    "_op_type": "index",
                    "_index": target_index,
                    "_id": item["_id"],
                    "_source": {
                        "text": reduce(lambda d, key: d[key], embedding_text_field, item['_source']),
                        "embedding": np.round(embeddings[i].tolist(), decimal_precision).tolist(),
                    }
                }
            actions.append(action)
        helpers.bulk(self.es, actions)
        logger.info(f"Indexed data into target index: {target_index} successfully")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="localhost", help="Elasticsearch host")
    parser.add_argument("--port", default=9200, type=int, help="Elasticsearch port")
    parser.add_argument("--username", default=None, help="Elasticsearch username for authentication")
    parser.add_argument("--password", default=None, help="Elasticsearch password for authentication")
    parser.add_argument("--source_index", required=True, help="Name of the source index to fetch data from")
    parser.add_argument("--target_index", required=True, help="Name of the target index to save embeddings")
    parser.add_argument("--sbert_model_name", default="snunlp/KR-SBERT-V40K-klueNLI-augSTS", help="Name of the SBERT model to use for embeddings")
    parser.add_argument("--embedding_text_field", nargs='+', default=['text'], help="Path to the text field in the data")
    parser.add_argument("--decimal_precision", default=5, type=int, help="Number of decimal places for the embedding values")
    parser.add_argument("--query_size", default=1000, type=int, help="Number of documents to fetch from Elasticsearch")
    args = parser.parse_args()

    indexer = ElasticsearchVectorIndexer(
        host=args.host, 
        port=args.port, 
        sbert_model_name=args.sbert_model_name,
        username=args.username,
        password=args.password
    )
    
    data = indexer.get_data_from_es(args.source_index, query_size=args.query_size, embedding_text_field=args.embedding_text_field)
    
    embeddings = indexer.embed_data(data, embedding_text_field=args.embedding_text_field)
    
    update = args.source_index == args.target_index
    indexer.index_data(
        args.target_index, 
        data, 
        embeddings, 
        embedding_text_field=args.embedding_text_field, 
        decimal_precision=args.decimal_precision,
        update = update
    )
    
    logger.info("Process completed successfully")