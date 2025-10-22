# app/rag/fusion.py
from collections import defaultdict

def rrf(rank_lists, k=60):
    # rank_lists: list of lists of doc_ids in ranked order
    scores = defaultdict(float)
    for lst in rank_lists:
        for rank, doc_id in enumerate(lst, start=1):
            scores[doc_id] += 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)

def weighted_scores(candidates, weights):
    # candidates: dict like {"bm25": [(id, score), ...], "vector": [(id, score), ...], "ocr": [(id, score), ...]}
    # weights:   dict like {"bm25": 0.4, "vector": 0.5, "ocr": 0.1}
    agg = defaultdict(float)
    for name, pairs in candidates.items():
        w = weights.get(name, 0.0)
        for doc_id, score in pairs:
            agg[doc_id] += w * score
    return sorted(agg.items(), key=lambda x: x[1], reverse=True)
