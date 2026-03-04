def build_context_with_sources(passages):

    context = ""
    sources = []

    for i, p in enumerate(passages):

        source_tag = f"[Source {i+1}]"

        context += f"{source_tag}\n{p}\n\n"

        sources.append({
            "id": i + 1,
            "text": p[:400]
        })

    return context, sources