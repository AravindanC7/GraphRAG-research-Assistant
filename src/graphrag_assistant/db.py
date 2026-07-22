"""Neo4j connection helpers and schema/index setup."""

from neo4j import Driver, GraphDatabase

from .config import settings

CHUNK_VECTOR_INDEX = "chunk_embedding"


def get_driver() -> Driver:
    return GraphDatabase.driver(
        settings.neo4j_uri, auth=(settings.neo4j_user, settings.neo4j_password)
    )


def ensure_schema(driver: Driver) -> None:
    """Create uniqueness constraints and the chunk vector index. Idempotent."""
    dims = int(settings.embedding_dimensions)
    with driver.session(database=settings.neo4j_database) as session:
        session.run(
            "CREATE CONSTRAINT paper_id IF NOT EXISTS "
            "FOR (p:Paper) REQUIRE p.id IS UNIQUE"
        )
        session.run(
            "CREATE CONSTRAINT chunk_id IF NOT EXISTS "
            "FOR (c:Chunk) REQUIRE c.id IS UNIQUE"
        )
        # dims is an int from our own config, so inlining is safe (index
        # OPTIONS historically did not accept query parameters).
        session.run(
            f"""
            CREATE VECTOR INDEX {CHUNK_VECTOR_INDEX} IF NOT EXISTS
            FOR (c:Chunk) ON (c.embedding)
            OPTIONS {{ indexConfig: {{
                `vector.dimensions`: {dims},
                `vector.similarity_function`: 'cosine'
            }} }}
            """
        )
