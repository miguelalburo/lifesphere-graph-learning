"""Neo4j connection handling — the only place the driver is constructed.

Credentials come from the gitignored repo-root `.env`. Nothing here interprets
the graph; it hands back rows exactly as the driver produced them, which is what
lets `raw.py` claim its output is verbatim.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from neo4j import Session

# The repo root, resolved from this file rather than the working directory, so a
# pull launched from a notebook subdirectory still finds the same `.env`.
REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Neo4jSettings:
    """Connection settings for one Neo4j instance."""

    uri: str
    user: str
    password: str
    database: str

    def redacted(self) -> str:
        """Describe the target without exposing the password."""
        return f"{self.user}@{self.uri}/{self.database}"


def settings_from_env(env_path: Path | None = None) -> Neo4jSettings:
    """Read connection settings from `.env`, falling back to the environment.

    Raises rather than defaulting: a silently wrong database is the expensive
    failure here, since `lifesphere` (clinical only) and `old` (legacy, carries
    the omics layer) have overlapping schemas and different contents.
    """
    _load_env_file(env_path if env_path is not None else REPO_ROOT / ".env")

    missing = [
        key
        for key in ("NEO4J_URI", "NEO4J_USER", "NEO4J_PASSWORD", "NEO4J_DATABASE")
        if not os.environ.get(key)
    ]
    if missing:
        raise RuntimeError(
            f"Missing Neo4j settings: {', '.join(missing)}. "
            f"Expected them in {REPO_ROOT / '.env'} or the environment."
        )

    return Neo4jSettings(
        uri=os.environ["NEO4J_URI"],
        user=os.environ["NEO4J_USER"],
        password=os.environ["NEO4J_PASSWORD"],
        database=os.environ["NEO4J_DATABASE"],
    )


def _load_env_file(path: Path) -> None:
    """Populate `os.environ` from a `.env` file, without overriding what is set.

    Deliberately not `dotenv.load_dotenv()`: that resolves the file by walking
    the *caller's* stack, which fails outright under `python -c` and silently
    picks up the wrong file when a pull is launched from elsewhere. The path is
    an argument here.
    """
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip("'\""))


@contextmanager
def graph_session(settings: Neo4jSettings | None = None) -> Iterator[Session]:
    """Yield a session against the configured database.

    The database is passed explicitly on every session — omitting it silently
    targets the instance's default, which is how a clinical-only query ends up
    reading the legacy `old` database.
    """
    from neo4j import GraphDatabase

    resolved = settings if settings is not None else settings_from_env()
    driver = GraphDatabase.driver(resolved.uri, auth=(resolved.user, resolved.password))
    try:
        driver.verify_connectivity()
        with driver.session(database=resolved.database) as session:
            yield session
    finally:
        driver.close()


def run(session: Session, cypher: str, **params: Any) -> list[dict[str, Any]]:
    """Run `cypher` and materialise every row as a plain dict.

    Materialising is intentional: the caller writes the result to `data/raw/`,
    and a lazily-consumed result that fails halfway leaves a truncated file that
    looks complete.
    """
    return [record.data() for record in session.run(cypher, **params)]
