def retrieve_chunks(
    question,
    collection,
    query_model,
    top_k=5,
    alpha=0.5
):
    """
    Retrieve the most relevant medical chunks
    using Weaviate Hybrid Search.
    """

    # Convert question to embedding
    query_embedding = query_model.encode(
        question,
        normalize_embeddings=True
    )

    # Hybrid search
    response = collection.query.hybrid(
        query=question,
        vector=query_embedding.tolist(),
        alpha=alpha,
        limit=top_k
    )

    # Extract results
    results = []

    for obj in response.objects:
        results.append({
            "text": obj.properties.get("text", ""),
            "section": obj.properties.get("section", ""),
            "source_id": obj.properties.get("source_id", ""),
            "brand_name": obj.properties.get("brand_name", ""),
            "generic_name": obj.properties.get("generic_name", ""),
            "chunk_index": obj.properties.get("chunk_index", 0)
        })

    return results