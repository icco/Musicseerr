"""Tests for LibraryDB paginated query methods."""

import asyncio
import threading
from pathlib import Path

import pytest

from infrastructure.persistence.library_db import LibraryDB


@pytest.fixture
def db(tmp_path: Path) -> LibraryDB:
    return LibraryDB(db_path=tmp_path / "test.db", write_lock=threading.Lock())


def _make_albums(count: int, *, start: int = 1) -> list[dict]:
    """Generate album dicts with predictable, sortable data."""
    albums = []
    for i in range(start, start + count):
        albums.append(
            {
                "mbid": f"album-{i:04d}",
                "artist_mbid": f"artist-{(i % 5) + 1:04d}",
                "artist_name": f"Artist {chr(65 + (i % 26))}",
                "title": f"Album {chr(65 + ((i + 13) % 26))} {i:04d}",
                "year": 2000 + (i % 24),
                "cover_url": None,
                "monitored": True,
                "date_added": 1700000000 + i * 100,
            }
        )
    return albums


def _make_artists(count: int) -> list[dict]:
    """Generate artist dicts with predictable data."""
    artists = []
    for i in range(1, count + 1):
        artists.append(
            {
                "mbid": f"artist-{i:04d}",
                "name": f"Artist {chr(65 + (i % 26))}",
                "album_count": i,
                "date_added": 1700000000 + i * 100,
            }
        )
    return artists


async def _seed(db: LibraryDB, n_albums: int = 100, n_artists: int = 20) -> None:
    await db.save_library(_make_artists(n_artists), _make_albums(n_albums))


# --- Album pagination ---


def test_albums_basic_pagination(db: LibraryDB):
    asyncio.run(_seed(db))
    items, total = asyncio.run(
        db.get_albums_paginated(limit=10, offset=0)
    )
    assert total == 100
    assert len(items) == 10


def test_albums_offset_beyond_total(db: LibraryDB):
    asyncio.run(_seed(db))
    items, total = asyncio.run(
        db.get_albums_paginated(limit=10, offset=200)
    )
    assert total == 100
    assert len(items) == 0


def test_albums_last_partial_page(db: LibraryDB):
    asyncio.run(_seed(db))
    items, total = asyncio.run(
        db.get_albums_paginated(limit=30, offset=90)
    )
    assert total == 100
    assert len(items) == 10


def test_albums_sort_by_title_asc(db: LibraryDB):
    asyncio.run(_seed(db))
    items, _ = asyncio.run(
        db.get_albums_paginated(limit=100, offset=0, sort_by="title", sort_order="asc")
    )
    titles = [i.get("title", "") for i in items]
    assert titles == sorted(titles, key=str.casefold)


def test_albums_sort_by_title_desc(db: LibraryDB):
    asyncio.run(_seed(db))
    items, _ = asyncio.run(
        db.get_albums_paginated(limit=100, offset=0, sort_by="title", sort_order="desc")
    )
    titles = [i.get("title", "") for i in items]
    assert titles == sorted(titles, key=str.casefold, reverse=True)


def test_albums_sort_by_year(db: LibraryDB):
    asyncio.run(_seed(db))
    items, _ = asyncio.run(
        db.get_albums_paginated(limit=100, offset=0, sort_by="year", sort_order="desc")
    )
    years = [i.get("year", 0) or 0 for i in items]
    assert years == sorted(years, reverse=True)


def test_albums_sort_by_date_added(db: LibraryDB):
    asyncio.run(_seed(db))
    items, _ = asyncio.run(
        db.get_albums_paginated(limit=100, offset=0, sort_by="date_added", sort_order="desc")
    )
    dates = [i.get("date_added", 0) or 0 for i in items]
    assert dates == sorted(dates, reverse=True)


def test_albums_search_by_title(db: LibraryDB):
    asyncio.run(_seed(db))
    items, total = asyncio.run(
        db.get_albums_paginated(limit=100, offset=0, search="Album A")
    )
    assert total > 0
    assert all("Album A" in i.get("title", "") for i in items)


def test_albums_search_by_artist(db: LibraryDB):
    asyncio.run(_seed(db))
    items, total = asyncio.run(
        db.get_albums_paginated(limit=100, offset=0, search="Artist A")
    )
    assert total > 0
    assert all(
        "Artist A" in i.get("artist_name", "") or "Artist A" in i.get("title", "")
        for i in items
    )


def test_albums_search_no_results(db: LibraryDB):
    asyncio.run(_seed(db))
    items, total = asyncio.run(
        db.get_albums_paginated(limit=10, offset=0, search="zzz_no_match_zzz")
    )
    assert total == 0
    assert len(items) == 0


def test_albums_search_case_insensitive(db: LibraryDB):
    asyncio.run(_seed(db))
    items_upper, total_upper = asyncio.run(
        db.get_albums_paginated(limit=100, offset=0, search="ALBUM A")
    )
    items_lower, total_lower = asyncio.run(
        db.get_albums_paginated(limit=100, offset=0, search="album a")
    )
    assert total_upper == total_lower
    assert len(items_upper) == len(items_lower)


def test_albums_search_escapes_like_metacharacters(db: LibraryDB):
    asyncio.run(_seed(db))
    items_pct, total_pct = asyncio.run(
        db.get_albums_paginated(limit=100, offset=0, search="100%")
    )
    assert total_pct == 0
    assert len(items_pct) == 0
    items_under, total_under = asyncio.run(
        db.get_albums_paginated(limit=100, offset=0, search="Album_A")
    )
    assert total_under == 0


def test_artists_search_escapes_like_metacharacters(db: LibraryDB):
    asyncio.run(_seed(db))
    items, total = asyncio.run(
        db.get_artists_paginated(limit=100, offset=0, search="Artist%B")
    )
    assert total == 0


def test_albums_invalid_sort_falls_back(db: LibraryDB):
    asyncio.run(_seed(db))
    items, total = asyncio.run(
        db.get_albums_paginated(limit=10, offset=0, sort_by="nonexistent")
    )
    assert total == 100
    assert len(items) == 10


def test_albums_empty_library(db: LibraryDB):
    items, total = asyncio.run(
        db.get_albums_paginated(limit=10, offset=0)
    )
    assert total == 0
    assert len(items) == 0


# --- Artist pagination ---


def test_artists_basic_pagination(db: LibraryDB):
    asyncio.run(_seed(db))
    items, total = asyncio.run(
        db.get_artists_paginated(limit=5, offset=0)
    )
    assert total == 20
    assert len(items) == 5


def test_artists_offset_beyond_total(db: LibraryDB):
    asyncio.run(_seed(db))
    items, total = asyncio.run(
        db.get_artists_paginated(limit=10, offset=50)
    )
    assert total == 20
    assert len(items) == 0


def test_artists_sort_by_name_asc(db: LibraryDB):
    asyncio.run(_seed(db))
    items, _ = asyncio.run(
        db.get_artists_paginated(limit=20, offset=0, sort_by="name", sort_order="asc")
    )
    names = [i.get("name", "") for i in items]
    assert names == sorted(names, key=str.casefold)


def test_artists_sort_by_album_count_desc(db: LibraryDB):
    asyncio.run(_seed(db))
    items, _ = asyncio.run(
        db.get_artists_paginated(limit=20, offset=0, sort_by="album_count", sort_order="desc")
    )
    counts = [i.get("album_count", 0) for i in items]
    assert counts == sorted(counts, reverse=True)


def test_artists_search(db: LibraryDB):
    asyncio.run(_seed(db))
    items, total = asyncio.run(
        db.get_artists_paginated(limit=20, offset=0, search="Artist B")
    )
    assert total > 0
    assert all("Artist B" in i.get("name", "") for i in items)


def test_artists_search_no_results(db: LibraryDB):
    asyncio.run(_seed(db))
    items, total = asyncio.run(
        db.get_artists_paginated(limit=10, offset=0, search="zzz_no_match_zzz")
    )
    assert total == 0
    assert len(items) == 0


def test_artists_empty_library(db: LibraryDB):
    items, total = asyncio.run(
        db.get_artists_paginated(limit=10, offset=0)
    )
    assert total == 0
    assert len(items) == 0


# --- Pagination consistency (no duplicates/missing across pages) ---


def test_albums_pagination_no_duplicates(db: LibraryDB):
    asyncio.run(_seed(db, n_albums=50))
    all_mbids: list[str] = []
    offset = 0
    page_size = 10
    while True:
        items, total = asyncio.run(
            db.get_albums_paginated(
                limit=page_size, offset=offset, sort_by="title", sort_order="asc"
            )
        )
        if not items:
            break
        all_mbids.extend(i.get("mbid", "") for i in items)
        offset += page_size
    assert len(all_mbids) == 50
    assert len(set(all_mbids)) == 50


def test_artists_pagination_no_duplicates(db: LibraryDB):
    asyncio.run(_seed(db, n_albums=10, n_artists=30))
    all_mbids: list[str] = []
    offset = 0
    page_size = 7
    while True:
        items, total = asyncio.run(
            db.get_artists_paginated(
                limit=page_size, offset=offset, sort_by="name", sort_order="asc"
            )
        )
        if not items:
            break
        all_mbids.extend(i.get("mbid", "") for i in items)
        offset += page_size
    assert len(all_mbids) == 30
    assert len(set(all_mbids)) == 30
