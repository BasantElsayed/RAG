# def retrieve_chunks(
#     question,
#     collection,
#     query_model,
#     top_k=5,
#     alpha=0.5
# ):
#     """
#     Retrieve the most relevant medical chunks
#     using Weaviate Hybrid Search.
#     """

#     # Convert question to embedding
#     query_embedding = query_model.encode(
#         question,
#         normalize_embeddings=True
#     )

#     # Hybrid search
#     response = collection.query.hybrid(
#         query=question,
#         vector=query_embedding.tolist(),
#         alpha=alpha,
#         limit=top_k
#     )

#     # Extract results
#     results = []

#     for obj in response.objects:
#         results.append({
#             "text": obj.properties.get("text", ""),
#             "section": obj.properties.get("section", ""),
#             "source_id": obj.properties.get("source_id", ""),
#             "brand_name": obj.properties.get("brand_name", ""),
#             "generic_name": obj.properties.get("generic_name", ""),
#             "chunk_index": obj.properties.get("chunk_index", 0)
#         })
#     return results


from weaviate.classes.query import Filter

def detect_target_sections(question):
    """Simple rule-based router to detect intended FDA label section."""
    q_lower = question.lower()
    sections = []

    if "warning" in q_lower or "precaution" in q_lower:
        sections.extend(["warnings", "boxed_warning"])
    if "side effect" in q_lower or "adverse reaction" in q_lower:
        sections.append("adverse_reactions")
    if "contraindication" in q_lower or "not take" in q_lower or "avoid" in q_lower:
        sections.append("contraindications")
    if "interact" in q_lower or "other drug" in q_lower:
        sections.append("drug_interactions")
    if "dose" in q_lower or "dosage" in q_lower or "how to take" in q_lower:
        sections.append("dosage_and_administration")
    if "indication" in q_lower or "used for" in q_lower:
        sections.append("indications_and_usage")
    if "ingredient" in q_lower:
        sections.append("active_ingredient")

    return sections if sections else None

def retrieve_chunks(
    question,
    collection,
    query_model,
    top_k=5,
    alpha=0.5
):
    """
    Retrieve the most relevant medical chunks
    using Weaviate Hybrid Search with exact section filtering.
    """

    # Convert question to embedding
    query_embedding = query_model.encode(
        question,
        normalize_embeddings=True
    )

    # 1. Scope Detection (Rule-based Metadata Filter)
    target_sections = detect_target_sections(question)

    weaviate_filter = None
    if target_sections:
        if len(target_sections) == 1:
            weaviate_filter = Filter.by_property("section").equal(target_sections[0])
        else:
            weaviate_filter = Filter.any_of([
                Filter.by_property("section").equal(sec) for sec in target_sections
            ])

    # 2. Hybrid search with Filter
    response = collection.query.hybrid(
        query=question,
        vector=query_embedding.tolist(),
        alpha=alpha,
        limit=top_k,
        filters=weaviate_filter
    )

    # 3. Fallback: If filtered search returns empty, try without filter
    if len(response.objects) == 0 and weaviate_filter is not None:
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
