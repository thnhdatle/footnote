# Retrieval evaluation: scifact

Run `20261019T234931Z-71a7f52` at commit `71a7f52`. Rows marked **shipped** use the shipped defaults for every setting the row varies. Document-level metrics: chunks ranked to depth 100 and pooled to documents by first occurrence.

| Configuration | Split | Settings | R@1 | R@5 | R@10 | R@100 | nDCG@10 | MRR@10 | p50 ms | p95 ms | Queries |
|---|---|---|---|---|---|---|---|---|---|---|---|
| vector [heading] **shipped** | test | chunking=heading | 0.5854 | 0.7661 | 0.8469 | 0.9567 | 0.7221 | 0.6870 | 32.1 | 37.2 | 300 |
| lexical any+ts_rank [heading] **shipped** | test | chunking=heading lexical_match=any lexical_rank=ts_rank | 0.4500 | 0.6514 | 0.7205 | 0.8556 | 0.5900 | 0.5587 | 31.0 | 40.5 | 300 |
| hybrid lexical weight 0.3 [heading] **shipped** | test | chunking=heading lexical_match=any lexical_rank=ts_rank rrf_k=60 rrf_weights=lexical:0.3,vector:1 | 0.5794 | 0.7788 | 0.8496 | 0.9567 | 0.7205 | 0.6850 | 60.5 | 75.3 | 300 |
| hybrid+rerank int8 rerank_n 30 [heading] **shipped** | test | chunking=heading lexical_match=any lexical_rank=ts_rank rrf_k=60 rrf_weights=lexical:0.3,vector:1 reranker=int8 rerank_n=30 | 0.5646 | 0.7660 | 0.8402 | 0.9567 | 0.7047 | 0.6678 | 299.4 | 340.3 | 300 |
| bm25s (reference, in-process) | test | bm25s lucene k1=1.5 b=0.75, English stopwords, Snowball stemmer | 0.5342 | 0.7362 | 0.8187 | 0.9127 | 0.6863 | 0.6492 | 0.2 | 0.2 | 300 |
| bm25s + vector RRF (reference, not shippable) | test | rrf_k=60 rrf_weights=lexical:0.3,vector:1; document-level fusion | 0.5961 | 0.7918 | 0.8536 | 0.9700 | 0.7343 | 0.7010 | 32.3 | 37.5 | 300 |

## BEIR published references (nDCG@10, test)

| Reference | nDCG@10 |
|---|---|
| BEIR BM25 | 0.665 |
| BEIR BM25+CE | 0.688 |

## Run

- dataset `scifact` (archive sha256 536e14446a0b): 5183 documents
- embedding space `bge-small-en-v1.5-onnx`: onnx_local `bge-small-en-v1.5` (BAAI/bge-small-en-v1.5@5c38ec7c405e, model sha256 828e1496d7fa)
- reranker `ms-marco-MiniLM-L6-v2` (cross-encoder/ms-marco-MiniLM-L6-v2@233902d25c44; files fp32 5d3e70fd0c9f, int8-amd64 c80a8b34256e, int8-arm64 3573b6b9593c); configured variant int8
- FTS config `english`; binary relevance from qrels; depth 100, k_lexical = k_vector = max(configured, 100); documents pooled by first occurrence
- p50/p95: in-process `SearchService.search` latency per query (query embedding included, sequential queries)
- heading chunking: 6362 chunks, 6362 texts embedded (cache misses and hits), indexed in 378s
- embedding cache: on, marked warm: scifact
- temporary tenant 01a15692-00e0-ee69-05a7-7a0fb4dd3901 deleted; total 509s
