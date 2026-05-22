from __future__ import annotations

from dataclasses import dataclass


SEED = 42
OUTER_FOLDS = 5
VAL_SIZE = 0.2
METRIC_FOR_SELECTION = "f1_macro"
LOGISTIC_C_VALUES = [0.01, 0.1, 1, 10]


@dataclass(frozen=True)
class RepresentationSpec:
    name: str
    family: str
    vectorizer_type: str | None = None
    analyzer: str | None = None
    ngram_range: tuple[int, int] | None = None
    max_features: int | None = None
    model_name: str | None = None
    normalize_embeddings: bool = False

    @property
    def slug(self) -> str:
        raw = self.model_name or self.name
        return (
            raw.lower()
            .replace("/", "__")
            .replace(" ", "_")
            .replace("(", "")
            .replace(")", "")
            .replace(",", "_")
        )


LEXICAL_SPECS = {
    "bow_unigram": RepresentationSpec(
        name="bow_unigram",
        family="lexical",
        vectorizer_type="count",
        analyzer="word",
        ngram_range=(1, 1),
        max_features=1000,
    ),
    "bow_unigram_bigram": RepresentationSpec(
        name="bow_unigram_bigram",
        family="lexical",
        vectorizer_type="count",
        analyzer="word",
        ngram_range=(1, 2),
        max_features=1000,
    ),
    "tfidf_unigram": RepresentationSpec(
        name="tfidf_unigram",
        family="lexical",
        vectorizer_type="tfidf",
        analyzer="word",
        ngram_range=(1, 1),
        max_features=1000,
    ),
    "tfidf_unigram_bigram": RepresentationSpec(
        name="tfidf_unigram_bigram",
        family="lexical",
        vectorizer_type="tfidf",
        analyzer="word",
        ngram_range=(1, 2),
        max_features=1000,
    ),
    "tfidf_char_wb_3_5": RepresentationSpec(
        name="tfidf_char_wb_3_5",
        family="lexical",
        vectorizer_type="tfidf",
        analyzer="char_wb",
        ngram_range=(3, 5),
        max_features=1000,
    ),
}


DENSE_SPECS = {
    "all_minilm_l6_v2": RepresentationSpec(
        name="all_minilm_l6_v2",
        family="dense",
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        normalize_embeddings=True,
    ),
    "all_mpnet_base_v2": RepresentationSpec(
        name="all_mpnet_base_v2",
        family="dense",
        model_name="sentence-transformers/all-mpnet-base-v2",
        normalize_embeddings=True,
    ),
    "multi_qa_minilm_l6_cos_v1": RepresentationSpec(
        name="multi_qa_minilm_l6_cos_v1",
        family="dense",
        model_name="sentence-transformers/multi-qa-MiniLM-L6-cos-v1",
        normalize_embeddings=True,
    ),
    "e5_large_v2": RepresentationSpec(
        name="e5_large_v2",
        family="dense",
        model_name="intfloat/e5-large-v2",
        normalize_embeddings=True,
    ),
    "bge_m3": RepresentationSpec(
        name="bge_m3",
        family="dense",
        model_name="BAAI/bge-m3",
        normalize_embeddings=True,
    ),
    "gte_modernbert_base": RepresentationSpec(
        name="gte_modernbert_base",
        family="dense",
        model_name="Alibaba-NLP/gte-modernbert-base",
        normalize_embeddings=True,
    ),
    "nomic_embed_text_v1_5": RepresentationSpec(
        name="nomic_embed_text_v1_5",
        family="dense",
        model_name="nomic-ai/nomic-embed-text-v1.5",
        normalize_embeddings=True,
    ),
    "embeddinggemma_300m": RepresentationSpec(
        name="embeddinggemma_300m",
        family="dense",
        model_name="google/embeddinggemma-300m",
        normalize_embeddings=True,
    ),
    "granite_embedding_125m_english": RepresentationSpec(
        name="granite_embedding_125m_english",
        family="dense",
        model_name="ibm-granite/granite-embedding-125m-english",
        normalize_embeddings=True,
    ),
    "qwen3_embedding_0_6b": RepresentationSpec(
        name="qwen3_embedding_0_6b",
        family="dense",
        model_name="Qwen/Qwen3-Embedding-0.6B",
        normalize_embeddings=True,
    ),
    "qwen3_embedding_8b": RepresentationSpec(
        name="qwen3_embedding_8b",
        family="dense",
        model_name="Qwen/Qwen3-Embedding-8B",
        normalize_embeddings=True,
    ),
    "gemini_embedding_2_preview": RepresentationSpec(
        name="gemini_embedding_2_preview",
        family="remote",
        model_name="google/gemini-embedding-2-preview",
        normalize_embeddings=False,
    ),
    "text_embedding_3_large": RepresentationSpec(
        name="text_embedding_3_large",
        family="remote",
        model_name="openai/text-embedding-3-large",
        normalize_embeddings=False,
    ),
}


ALL_SPECS = {**LEXICAL_SPECS, **DENSE_SPECS}
