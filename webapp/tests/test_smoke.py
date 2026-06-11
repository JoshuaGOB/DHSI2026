def test_dependencies_importable():
    import chromadb  # noqa: F401
    import fastapi  # noqa: F401
    from langchain_anthropic import ChatAnthropic  # noqa: F401
    from langchain_community.document_loaders import PyPDFLoader  # noqa: F401
    from langchain_core.embeddings import DeterministicFakeEmbedding  # noqa: F401
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings  # noqa: F401
    from langchain_text_splitters import RecursiveCharacterTextSplitter  # noqa: F401
