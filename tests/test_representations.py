from embeddings_pipeline.representations import apply_text_prefix, get_text_prefix


def test_e5_prefix_is_applied() -> None:
    texts = ["hello world"]
    assert get_text_prefix("intfloat/e5-large-v2") == "query: "
    assert apply_text_prefix(texts, "intfloat/e5-large-v2") == ["query: hello world"]


def test_nomic_prefix_is_applied() -> None:
    texts = ["hello world"]
    assert get_text_prefix("nomic-ai/nomic-embed-text-v1.5") == "classification: "
    assert apply_text_prefix(texts, "nomic-ai/nomic-embed-text-v1.5") == [
        "classification: hello world"
    ]
