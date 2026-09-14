"""
Data contracts shared by the local runtime and the (future) cloud services.

Plain pydantic models only. Nothing in this package may import from `core/`,
`agents/` or `api/` — it is the one thing both sides of the boundary depend on,
so it cannot depend on either of them.
"""
