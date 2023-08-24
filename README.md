# SBERT_ES_Reindexer
Elasticsearch에 문장 임베딩을 색인하는 데 사용되는 도구입니다. 이 도구는 Sentence Transformers를 사용하여 문장의 임베딩을 생성하고, 해당 임베딩을 Elasticsearch에 색인합니다.

## Installation

1. Python 환경 준비하기:

```bash
python -m venv env
source env/bin/activate  # Linux/macOS
env\Scripts\activate  # Windows
```

2. 필요한 패키지 설치하기:

```bash
pip install -r requirements.txt
```

## Usage

```bash
python vector_indexer.py --host [Elasticsearch 호스트] --port [포트 번호] --source_index [원본 인덱스 이름] --target_index [대상 인덱스 이름] [--기타 옵션...]
```

기타 옵션에 대한 설명은 python vector_indexer.py --help를 참조하세요.

## Citing & Authors

이 도구는 Sentence Transformers를 기반으로 하며, 해당 라이브러리에 대한 자세한 정보나 출처는 [Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks](https://arxiv.org/abs/1908.10084)와 [Making Monolingual Sentence Embeddings Multilingual using Knowledge Distillation](https://arxiv.org/abs/2004.09813)을 참조하세요.

또한 Sentence Transformers와 관련된 기타 연구자료는 [Publications](https://www.sbert.net/docs/publications.html)에서 확인하실 수 있습니다.