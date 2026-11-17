"""Bounded traces of real vectors from a built inference index."""
from __future__ import annotations

from collections import Counter

import numpy as np
from sklearn.preprocessing import normalize

from thai_legal_rag.data import search_text
from thai_legal_rag.retrieval import _tokenize, _word2vec_average

_LIMIT = 10


def query_vector(index, question: str, method: str):
    """Create the same normalized query vector used by ``InferenceIndex.search``."""
    if method not in index.methods:
        raise ValueError(f"unknown or unavailable method {method!r}; choose from {list(index.methods)}")
    text = search_text(question)
    if method in {"r1", "r2"}:
        return normalize(index.vectorizers[method].transform([_tokenize(text)]))
    if method == "r3":
        return _word2vec_average([_tokenize(text)], index.w2v)
    vectors = index.encoder.encode([text], show_progress_bar=False).astype(np.float32)
    return normalize(vectors)


def _values(matrix, row: int) -> np.ndarray:
    values = matrix[row]
    return np.asarray(values.toarray() if hasattr(values, "toarray") else values,
                      dtype=np.float32).reshape(-1)


def _coordinates(vector: np.ndarray) -> list[int]:
    active = np.flatnonzero(vector)
    return sorted(active, key=lambda i: abs(float(vector[i])), reverse=True)[:_LIMIT]


def _statute_rows(index, method: str, statute_ids: list[str], dimensions: list[int], terms=None):
    rows = []
    for statute_id in statute_ids:
        if statute_id not in index.corpus:
            raise ValueError(f"unknown statute ID {statute_id!r}")
        row = _values(index.matrices[method], index.ids.index(statute_id))
        statute = index.corpus[statute_id]
        coords = []
        for dimension in dimensions:
            coord = {"index": int(dimension), "value": float(row[dimension])}
            if terms is not None:
                coord["term"] = str(terms[dimension])
            coords.append(coord)
        rows.append({
            "statute_id": statute_id,
            "law_name": str(statute.get("law_name", "")),
            "section_num": str(statute.get("section_num", "")),
            "normalized_l2_norm": float(np.linalg.norm(row)),
            "coordinates": coords,
        })
    return rows


def _sparse_trace(index, question: str, method: str, query_vector, statute_ids: list[str]):
    tokens = _tokenize(search_text(question))
    counts = Counter(tokens)
    vectorizer = index.vectorizers[method]
    terms = vectorizer.get_feature_names_out()

    # Reconstruct unnormalized values with the same vocabulary and TF-IDF
    # weights. The indexed matrix and query vector contain their L2-normalized form.
    raw = np.zeros(len(terms), dtype=np.float32)
    for token, count in counts.items():
        dimension = vectorizer.vocabulary_.get(token)
        if dimension is not None:
            raw[dimension] = 1.0 if method == "r1" else count * vectorizer.idf_[dimension]
    dimensions = list(np.flatnonzero(raw))
    for statute_id in statute_ids:
        if statute_id not in index.corpus:
            raise ValueError(f"unknown statute ID {statute_id!r}")
        doc = _values(index.matrices[method], index.ids.index(statute_id))
        dimensions.extend(i for i in np.flatnonzero(doc) if i not in dimensions)
        if len(dimensions) >= _LIMIT:
            break
    dimensions = dimensions[:_LIMIT]
    docs = []
    for row in _statute_rows(index, method, statute_ids, dimensions, terms):
        tokens_in_doc = _tokenize(search_text(index.corpus[row["statute_id"]]["section_content"]))
        doc_counts = Counter(tokens_in_doc)
        doc_raw = np.zeros(len(terms), dtype=np.float32)
        for token, count in doc_counts.items():
            dimension = vectorizer.vocabulary_.get(token)
            if dimension is not None:
                doc_raw[dimension] = 1.0 if method == "r1" else count * vectorizer.idf_[dimension]
        row["raw_norm"] = float(np.linalg.norm(doc_raw))
        for coord in row["coordinates"]:
            coord["raw_value"] = float(doc_raw[coord["index"]])
        docs.append(row)

    details = {
        "method": index.methods[method],
        "dimension": len(terms),
        "tokenizer": "PyThaiNLP newmm",
        "tokens": tokens,
        "raw_query_norm": float(np.linalg.norm(raw)),
        "normalized_query_norm": float(np.linalg.norm(query_vector.toarray())),
        "coordinates": [
            {
                "index": int(i), "term": str(terms[i]),
                "query_token_count": int(counts[str(terms[i])]),
                "raw_value": float(raw[i]),
                "normalized_value": float(query_vector[0, i]),
                **({"idf": float(vectorizer.idf_[i])} if method == "r2" else {}),
            }
            for i in dimensions
        ],
        "statute_vectors": docs,
    }
    if method == "r1":
        details["representation"] = "binary term presence, followed by L2 normalization"
    else:
        details["representation"] = "term frequency × smoothed IDF, followed by L2 normalization"
    return details


def _word2vec_trace(index, question: str, vector: np.ndarray, statute_ids: list[str]):
    tokens = _tokenize(search_text(question))
    known = [token for token in tokens if token in index.w2v.wv]
    unknown = [token for token in tokens if token not in index.w2v.wv]
    mean = (np.mean([index.w2v.wv[token] for token in known], axis=0)
            if known else np.zeros(index.w2v.vector_size, dtype=np.float32))
    dimensions = _coordinates(vector[0])
    return {
        "method": index.methods["r3"],
        "dimension": int(index.w2v.vector_size),
        "tokenizer": "PyThaiNLP newmm",
        "tokens": tokens,
        "known_tokens": [
            {"token": token, "frequency": int(index.w2v.wv.get_vecattr(token, "count"))}
            for token in known
        ],
        "out_of_vocabulary_tokens": unknown,
        "raw_mean_norm": float(np.linalg.norm(mean)),
        "normalized_query_norm": float(np.linalg.norm(vector[0])),
        "coordinates": [
            {"index": int(i), "raw_mean": float(mean[i]), "normalized_value": float(vector[0, i])}
            for i in dimensions
        ],
        "statute_vectors": _statute_rows(index, "r3", statute_ids, dimensions),
    }


def _bge_trace(index, question: str, vector: np.ndarray, statute_ids: list[str]):
    tokenizer = index.encoder.tokenizer
    text = search_text(question)
    limit = int(index.manifest["bge_max_tokens"])
    full = tokenizer(text, add_special_tokens=True, truncation=False)["input_ids"]
    used = tokenizer(text, add_special_tokens=True, truncation=True, max_length=limit)["input_ids"]
    dimensions = _coordinates(vector[0])
    docs = _statute_rows(index, "r4", statute_ids, dimensions)
    for row in docs:
        doc = search_text(index.corpus[row["statute_id"]]["section_content"])
        full_doc = tokenizer(doc, add_special_tokens=True, truncation=False)["input_ids"]
        used_doc = tokenizer(doc, add_special_tokens=True, truncation=True, max_length=limit)["input_ids"]
        row["token_count"] = len(full_doc)
        row["truncated"] = len(full_doc) > limit
        row["token_preview"] = tokenizer.convert_ids_to_tokens(used_doc[:_LIMIT])
    return {
        "method": index.methods["r4"],
        "model": index.manifest["bge_model"],
        "model_revision": index.manifest["bge_revision"],
        "dimension": int(vector.shape[1]),
        "tokenizer": "BGE-M3 model tokenizer",
        "token_count": len(full),
        "token_budget": limit,
        "truncated": len(full) > limit,
        "effective_token_count": len(used),
        "token_preview": tokenizer.convert_ids_to_tokens(used[:_LIMIT]),
        "normalized_query_norm": float(np.linalg.norm(vector[0])),
        "coordinates": [{"index": int(i), "value": float(vector[0, i])} for i in dimensions],
        "statute_vectors": docs,
        "coordinate_note": "Dense dimensions do not map to individual words.",
    }


def describe_embeddings(index, question: str, method: str, query_vectors,
                        sample_statute_ids: list[str]) -> dict:
    """Return a small JSON-safe trace of a real embedding and chosen statute vectors."""
    if method not in index.methods:
        raise ValueError(f"unknown or unavailable method {method!r}; choose from {list(index.methods)}")
    if query_vectors.shape[0] != 1 or query_vectors.shape[1] != index.matrices[method].shape[1]:
        raise ValueError("expected one query vector with the same dimension as the index")
    if method in {"r1", "r2"}:
        return _sparse_trace(index, question, method, query_vectors, sample_statute_ids)
    if method == "r3":
        return _word2vec_trace(index, question, query_vectors, sample_statute_ids)
    return _bge_trace(index, question, query_vectors, sample_statute_ids)
