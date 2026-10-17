"""
Cloud implementations of the index ports in `cloud/indexing/ports.py`.

Each one passes its port's behaviour suite in `tests/port_contracts/`, as does
any in-memory fake the tests stand in for it (`tests/fakes/`). Repositories for
the cloud's own tables — users, projects, jobs, usage — implement no index port,
so they live in `cloud/db/` instead.
"""
