# Retrieval evaluation: scifact

Run `20261020T091940Z` on a Footnote v0.1.0 pre-release build. Rows marked **shipped** use the shipped defaults for every setting the row varies. Document-level metrics: chunks ranked to depth 100 and pooled to documents by first occurrence.

| Configuration | Split | Settings | R@1 | R@5 | R@10 | R@100 | nDCG@10 | MRR@10 | p50 ms | p95 ms | Queries |
|---|---|---|---|---|---|---|---|---|---|---|---|
| vector [heading] **shipped** | test | chunking=heading | 0.5854 | 0.7661 | 0.8469 | 0.9567 | 0.7221 | 0.6870 | 20.1 | 24.8 | 300 |
| lexical any+ts_rank [heading] **shipped** | test | chunking=heading lexical_match=any lexical_rank=ts_rank | 0.4500 | 0.6514 | 0.7205 | 0.8556 | 0.5900 | 0.5587 | 33.9 | 44.8 | 300 |
| lexical any+ts_rank_cd [heading] | test | chunking=heading lexical_match=any lexical_rank=ts_rank_cd | 0.2228 | 0.4356 | 0.5250 | 0.7690 | 0.3717 | 0.3290 | 44.5 | 74.5 | 300 |
| lexical all+ts_rank [heading] | test | chunking=heading lexical_match=all lexical_rank=ts_rank | 0.0661 | 0.0661 | 0.0661 | 0.0661 | 0.0669 | 0.0700 | 13.0 | 15.0 | 300 |
| lexical websearch+ts_rank [heading] | test | chunking=heading lexical_match=websearch lexical_rank=ts_rank | 0.0694 | 0.0694 | 0.0694 | 0.0694 | 0.0703 | 0.0733 | 13.3 | 15.9 | 300 |
| hybrid lexical weight 1 [heading] | train | chunking=heading lexical_match=any lexical_rank=ts_rank rrf_k=60 rrf_weights=lexical:1,vector:1 | 0.5849 | 0.7726 | 0.8273 | 0.9582 | 0.7157 | 0.6860 | 44.9 | 55.6 | 809 |
| hybrid lexical weight 0.5 [heading] | train | chunking=heading lexical_match=any lexical_rank=ts_rank rrf_k=60 rrf_weights=lexical:0.5,vector:1 | 0.5923 | 0.7913 | 0.8548 | 0.9674 | 0.7339 | 0.7006 | 44.7 | 56.0 | 809 |
| hybrid lexical weight 0.3 [heading] | train | chunking=heading lexical_match=any lexical_rank=ts_rank rrf_k=60 rrf_weights=lexical:0.3,vector:1 | 0.5997 | 0.8111 | 0.8700 | 0.9627 | 0.7473 | 0.7148 | 45.1 | 55.8 | 809 |
| hybrid lexical weight 0.2 [heading] **shipped** | train | chunking=heading lexical_match=any lexical_rank=ts_rank rrf_k=60 rrf_weights=lexical:0.2,vector:1 | 0.5987 | 0.8165 | 0.8704 | 0.9627 | 0.7503 | 0.7182 | 44.6 | 55.9 | 809 |
| hybrid lexical weight 1 [heading] | test | chunking=heading lexical_match=any lexical_rank=ts_rank rrf_k=60 rrf_weights=lexical:1,vector:1 | 0.5619 | 0.7432 | 0.8132 | 0.9467 | 0.6954 | 0.6651 | 44.7 | 57.3 | 300 |
| hybrid lexical weight 0.5 [heading] | test | chunking=heading lexical_match=any lexical_rank=ts_rank rrf_k=60 rrf_weights=lexical:0.5,vector:1 | 0.5886 | 0.7638 | 0.8362 | 0.9683 | 0.7172 | 0.6858 | 44.9 | 56.9 | 300 |
| hybrid lexical weight 0.3 [heading] | test | chunking=heading lexical_match=any lexical_rank=ts_rank rrf_k=60 rrf_weights=lexical:0.3,vector:1 | 0.5794 | 0.7788 | 0.8496 | 0.9567 | 0.7205 | 0.6850 | 44.2 | 57.3 | 300 |
| hybrid lexical weight 0.2 [heading] **shipped** | test | chunking=heading lexical_match=any lexical_rank=ts_rank rrf_k=60 rrf_weights=lexical:0.2,vector:1 | 0.5827 | 0.7721 | 0.8496 | 0.9567 | 0.7224 | 0.6875 | 44.4 | 57.6 | 300 |
| hybrid+rerank int8 rerank_n 30 [heading] **shipped** | test | chunking=heading lexical_match=any lexical_rank=ts_rank rrf_k=60 rrf_weights=lexical:0.2,vector:1 reranker=int8 rerank_n=30 | 0.5679 | 0.7693 | 0.8429 | 0.9567 | 0.7093 | 0.6725 | 281.1 | 304.0 | 300 |
| hybrid+rerank int8 rerank_n 50 [heading] | test | chunking=heading lexical_match=any lexical_rank=ts_rank rrf_k=60 rrf_weights=lexical:0.2,vector:1 reranker=int8 rerank_n=50 | 0.5596 | 0.7610 | 0.8177 | 0.9567 | 0.6959 | 0.6641 | 422.4 | 450.5 | 300 |
| hybrid+rerank fp32 rerank_n 30 [heading] | test | chunking=heading lexical_match=any lexical_rank=ts_rank rrf_k=60 rrf_weights=lexical:0.2,vector:1 reranker=fp32 rerank_n=30 | 0.5646 | 0.7649 | 0.8329 | 0.9567 | 0.7045 | 0.6692 | 509.8 | 547.4 | 300 |
| hybrid+rerank fp32 rerank_n 50 [heading] | test | chunking=heading lexical_match=any lexical_rank=ts_rank rrf_k=60 rrf_weights=lexical:0.2,vector:1 reranker=fp32 rerank_n=50 | 0.5629 | 0.7588 | 0.8152 | 0.9567 | 0.6968 | 0.6655 | 856.4 | 1099.4 | 300 |
| hybrid+rerank int8 rerank_n 30 [recursive] | test | chunking=recursive lexical_match=any lexical_rank=ts_rank rrf_k=60 rrf_weights=lexical:0.2,vector:1 reranker=int8 rerank_n=30 | 0.5679 | 0.7693 | 0.8429 | 0.9567 | 0.7093 | 0.6725 | 288.5 | 331.9 | 300 |
| bm25s (reference, in-process) | test | bm25s lucene k1=1.5 b=0.75, English stopwords, Snowball stemmer | 0.5342 | 0.7362 | 0.8187 | 0.9127 | 0.6863 | 0.6492 | 0.2 | 0.2 | 300 |
| bm25s + vector RRF (reference, not shippable) | test | rrf_k=60 rrf_weights=lexical:0.2,vector:1; document-level fusion | 0.5894 | 0.7734 | 0.8469 | 0.9700 | 0.7288 | 0.6957 | 20.3 | 25.0 | 300 |
| bm25s (reference, in-process) | train | bm25s lucene k1=1.5 b=0.75, English stopwords, Snowball stemmer | 0.5700 | 0.7484 | 0.8137 | 0.9283 | 0.6964 | 0.6662 | 0.2 | 0.2 | 809 |

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
- heading chunking: 6362 chunks, 6362 texts embedded (cache misses and hits), indexed in 126s
- recursive chunking: 6362 chunks, 6362 texts embedded (cache misses and hits), indexed in 127s
- embedding cache: on, marked warm: scifact
- temporary tenant 01a1589b-ffab-d404-00f5-3cba6073b25d deleted; total 1217s
