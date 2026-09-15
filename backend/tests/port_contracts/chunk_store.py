"""
What every `ChunkStore` must do identically — Chroma locally, pgvector in the cloud.

Subclass `ChunkStoreContract` in an adapter's test module and provide three
fixtures:

  * `store`        a store for this test;
  * `new_project`  an async factory returning a fresh project id the store
                   accepts (a path for Chroma, a real `projects` row for Postgres);
  * `dimensions`   the vector size the store is set up for.
"""
from __future__ import annotations

import pytest

from core.index.ports import IndexedChunk, chunk_id


def unit_vector(dimensions: int, axis: int) -> list[float]:
    vector = [0.0] * dimensions
    vector[axis % dimensions] = 1.0
    return vector


class ChunkStoreContract:
    @pytest.fixture
    def chunk(self, dimensions):
        def make(path="a.py", start=1, file_hash="h1", axis=0) -> IndexedChunk:
            return IndexedChunk(
                file_path=path,
                start_line=start,
                end_line=start + 9,
                file_hash=file_hash,
                vector=unit_vector(dimensions, axis),
            )

        return make

    # ── The manifest ─────────────────────────────────────────────────────────

    async def test_an_unknown_project_has_an_empty_manifest(self, store, new_project):
        assert await store.manifest(await new_project()) == {}

    async def test_upserted_chunks_are_listed_by_file_with_their_hash(
        self, store, new_project, chunk
    ):
        project = await new_project()
        await store.apply(
            project,
            upserts=[chunk("a.py", 1), chunk("a.py", 20), chunk("b.py", 1, file_hash="h2")],
            delete=[],
        )

        manifest = await store.manifest(project)
        assert set(manifest) == {"a.py", "b.py"}
        assert sorted(manifest["a.py"].ids) == sorted([chunk_id("a.py", 1), chunk_id("a.py", 20)])
        assert manifest["b.py"].file_hash == "h2"

    async def test_a_path_containing_a_colon_survives_the_round_trip(
        self, store, new_project, chunk
    ):
        project = await new_project()
        await store.apply(project, upserts=[chunk("dir/a:b.py", 7)], delete=[])
        assert (await store.manifest(project))["dir/a:b.py"].ids == (chunk_id("dir/a:b.py", 7),)

        await store.apply(project, upserts=[], delete=[chunk_id("dir/a:b.py", 7)])
        assert await store.manifest(project) == {}

    # ── Applying a job ───────────────────────────────────────────────────────

    async def test_upserting_an_existing_id_replaces_it(self, store, new_project, chunk):
        project = await new_project()
        await store.apply(project, upserts=[chunk(file_hash="old")], delete=[])
        await store.apply(project, upserts=[chunk(file_hash="new")], delete=[])

        manifest = await store.manifest(project)
        assert manifest["a.py"].ids == (chunk_id("a.py", 1),)
        assert manifest["a.py"].file_hash == "new"

    async def test_delete_removes_exactly_the_given_ids(self, store, new_project, chunk):
        project = await new_project()
        await store.apply(
            project, upserts=[chunk("a.py", 1), chunk("a.py", 20), chunk("b.py", 1)], delete=[]
        )

        deleted = await store.apply(project, upserts=[], delete=[chunk_id("a.py", 20)])

        manifest = await store.manifest(project)
        assert deleted == 1
        assert manifest["a.py"].ids == (chunk_id("a.py", 1),)
        assert "b.py" in manifest

    async def test_deleting_ids_that_do_not_exist_is_harmless(self, store, new_project, chunk):
        project = await new_project()
        await store.apply(project, upserts=[chunk()], delete=[])
        await store.apply(project, upserts=[], delete=[chunk_id("ghost.py", 1)])
        assert set(await store.manifest(project)) == {"a.py"}

    async def test_one_apply_upserts_and_deletes_together(self, store, new_project, chunk):
        project = await new_project()
        await store.apply(project, upserts=[chunk("a.py", 1), chunk("a.py", 20)], delete=[])

        await store.apply(
            project, upserts=[chunk("a.py", 1, file_hash="h2")], delete=[chunk_id("a.py", 20)]
        )

        manifest = await store.manifest(project)
        assert manifest["a.py"].ids == (chunk_id("a.py", 1),)
        assert manifest["a.py"].file_hash == "h2"

    async def test_reset_discards_the_project_before_upserting(self, store, new_project, chunk):
        project = await new_project()
        await store.apply(project, upserts=[chunk("old.py")], delete=[])

        await store.apply(project, upserts=[chunk("new.py")], delete=[], reset=True)

        assert set(await store.manifest(project)) == {"new.py"}

    async def test_projects_never_see_each_other(self, store, new_project, chunk, dimensions):
        one, two = await new_project(), await new_project()
        await store.apply(one, upserts=[chunk("one.py")], delete=[])
        await store.apply(two, upserts=[chunk("two.py")], delete=[])

        assert set(await store.manifest(one)) == {"one.py"}
        hits = await store.search(two, unit_vector(dimensions, 0), 10)
        assert [hit.file_path for hit in hits] == ["two.py"]

    # ── Search ───────────────────────────────────────────────────────────────

    async def test_search_ranks_the_nearest_chunk_first(
        self, store, new_project, chunk, dimensions
    ):
        project = await new_project()
        await store.apply(
            project,
            upserts=[
                chunk("far.py", 1, file_hash="h-far", axis=1),
                chunk("near.py", 10, file_hash="h-near", axis=0),
            ],
            delete=[],
        )

        hits = await store.search(project, unit_vector(dimensions, 0), 2)

        assert [hit.file_path for hit in hits] == ["near.py", "far.py"]
        assert (hits[0].start_line, hits[0].end_line, hits[0].file_hash) == (10, 19, "h-near")
        assert hits[0].score > hits[1].score

    async def test_search_returns_at_most_n(self, store, new_project, chunk, dimensions):
        project = await new_project()
        await store.apply(
            project, upserts=[chunk("a.py", start) for start in (1, 11, 21, 31, 41)], delete=[]
        )
        assert len(await store.search(project, unit_vector(dimensions, 0), 2)) == 2

    async def test_searching_an_empty_project_finds_nothing(self, store, new_project, dimensions):
        assert await store.search(await new_project(), unit_vector(dimensions, 0), 5) == []
