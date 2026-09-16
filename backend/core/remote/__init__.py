"""
Runtime-side clients for the cloud: the LLM gateway, the index and the session.

They implement the same ports as the local build (`ModelGateway`,
`SemanticIndex`), so only `core/providers.py` ever chooses them.
"""
