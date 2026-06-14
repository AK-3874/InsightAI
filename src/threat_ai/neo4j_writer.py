from typing import Any, Dict
from pathlib import Path

try:
    from neo4j import GraphDatabase
except Exception:
    GraphDatabase = None


def export_graph_cypher(facts: list, out_path: str = "neo4j_export.cypher"):
    lines = []
    # facts: list of dicts with subject, object, predicate, metadata
    for fact in facts:
        subj = fact.get("subject")
        obj = fact.get("object")
        pred = fact.get("predicate") or "related_to"
        # create nodes and relationship
        lines.append(f"MERGE (a:Node {{name: \"{subj}\"}});")
        lines.append(f"MERGE (b:Node {{name: \"{obj}\"}});")
        lines.append(f"MERGE (a)-[:{pred.upper()}]->(b);")

    Path(out_path).write_text("\n".join(lines), encoding="utf-8")


def write_to_neo4j(facts: list, uri: str, user: str, password: str, dry_run: bool = False):
    """Writes facts into Neo4j. If the `neo4j` driver isn't installed, falls back to writing Cypher file."""
    if GraphDatabase is None or dry_run:
        export_graph_cypher(facts, out_path="neo4j_export.cypher")
        return {"status": "fallback_written", "path": "neo4j_export.cypher"}

    driver = GraphDatabase.driver(uri, auth=(user, password))
    with driver.session() as session:
        for fact in facts:
            subj = fact.get("subject")
            obj = fact.get("object")
            pred = (fact.get("predicate") or "related_to").upper()
            cypher = (
                "MERGE (a:Node {name: $subj}) MERGE (b:Node {name: $obj}) "
                f"MERGE (a)-[:{pred}]->(b)"
            )
            session.run(cypher, subj=subj, obj=obj)
    driver.close()
    return {"status": "written", "uri": uri}
