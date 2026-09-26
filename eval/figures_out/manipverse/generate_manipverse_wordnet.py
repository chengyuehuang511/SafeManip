"""WordNet-style faceted view of ManipVerse.

Reorganizes manipverse.json's flat entity->properties lists into several
independent hypernym trees ("dimensions"), WordNet-fashion: each entity
attaches to one or more nodes per dimension, and every node knows its
hypernym path back to the dimension root. Like WordNet, an entity may have
multiple senses within a dimension (wine is food/drink/alcohol AND, as a
bottle, vessel/receptacle).

Purely a re-projection of manipverse.json -- no new annotation. Run
generate_manipverse.py first, then:  python3 generate_manipverse_wordnet.py

Outputs (next to this script):
  - manipverse_wordnet.json   dimension trees + per-entity sense attachments
  - manipverse_wordnet.txt    human-readable tree with member counts
"""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE / "manipverse.json"

# ---------------------------------------------------------------------------
# Dimension trees. Each node: name -> parent (None = dimension root).
# `trigger` maps a node to the ontology property that attaches an entity to
# it; nodes without a trigger are pure hypernyms (attachment comes only via
# descendants).
# ---------------------------------------------------------------------------
DIMENSIONS: dict[str, dict] = {
    "kind": {
        "description": "is-a taxonomy: what the entity fundamentally is",
        "tree": {
            "kind": None,
            "food": "kind",
            "meat": "food", "dairy": "food", "fruit": "food", "vegetable": "food",
            "bread_food": "food", "cooked_food": "food", "pastry": "food",
            "sweet": "food", "condiment": "food", "spice": "food",
            "drink": "food", "alcohol": "drink",
            "vessel": "kind",
            "receptacle": "vessel", "cookware": "receptacle",
            "implement": "kind",
            "tool": "implement", "utensil": "tool",
        },
        "triggers": {
            "food": "food", "meat": "meat", "dairy": "dairy", "fruit": "fruit",
            "vegetable": "vegetable", "bread_food": "bread_food",
            "cooked_food": "cooked_food", "pastry": "pastry", "sweet": "sweet",
            "condiment": "condiment", "spice": "spice", "drink": "drink",
            "alcohol": "alcohol", "receptacle": "receptacle",
            "cookware": "cookware", "tool": "tool", "utensil": "utensil",
        },
    },
    "hazard": {
        "description": "safety hazard class the entity introduces",
        "tree": {
            "hazard": None,
            "breakage_hazard": "hazard",
            "contamination_hazard": "hazard",
            "spill_hazard": "hazard",
            "burn_hazard": "hazard",
        },
        "triggers": {
            "breakage_hazard": "fragile",
            "contamination_hazard": "raw",
            "spill_hazard": "liquid",
            "burn_hazard": "heated",  # fixtures only in practice
        },
    },
    "affordance": {
        "description": "what can safely be done to/with the entity",
        "tree": {
            "affordance": None,
            "manipulation": "affordance",
            "graspable": "manipulation", "pourable": "manipulation",
            "twistable": "manipulation", "openable": "manipulation",
            "closeable": "manipulation",
            "thermal_treatment": "affordance",
            "cookable": "thermal_treatment", "microwavable": "thermal_treatment",
            "toastable": "thermal_treatment", "fridgable": "thermal_treatment",
            "freezable": "thermal_treatment",
            "cleaning": "affordance",
            "washable": "cleaning", "dishwashable": "cleaning",
        },
        "triggers": {
            name: name
            for name in (
                "graspable", "pourable", "twistable", "openable", "closeable",
                "cookable", "microwavable", "toastable", "fridgable",
                "freezable", "washable", "dishwashable",
            )
        },
    },
    "role": {
        "description": "entity role in the scene, with fixture subtypes",
        "tree": {
            "role": None,
            "movable_object": "role",
            "fixture": "role",
            "heating_appliance": "fixture",
            "storage_fixture": "fixture",
            "cleaning_fixture": "fixture",
            "support_surface": "fixture",
        },
        # role attachment is computed specially (see entity_senses), not via
        # a single property trigger.
        "triggers": {},
    },
}

_ROLE_FIXTURE_SUBTYPES = (
    ("heating_appliance", {"heated"}),
    ("storage_fixture", {"storage", "fridge", "cabinet", "drawer"}),
    ("cleaning_fixture", {"washing", "sink"}),
)


def hypernym_path(tree: dict[str, str | None], node: str) -> list[str]:
    path = [node]
    while tree[path[-1]] is not None:
        path.append(tree[path[-1]])
    return list(reversed(path))


def entity_senses(role: str, props: set[str]) -> dict[str, list[str]]:
    senses: dict[str, set[str]] = {dim: set() for dim in DIMENSIONS}
    for dim, spec in DIMENSIONS.items():
        for node, trigger in spec["triggers"].items():
            if trigger in props:
                senses[dim].add(node)
    # role dimension
    if role == "object":
        senses["role"].add("movable_object")
    else:
        matched = False
        for node, keys in _ROLE_FIXTURE_SUBTYPES:
            if props & keys:
                senses[dim := "role"].add(node)
                matched = True
        if any(p.startswith("support:") for p in props):
            senses["role"].add("support_surface")
            matched = True
        if not matched:
            senses["role"].add("fixture")
    # WordNet-style: keep only the most specific nodes (drop a node if one of
    # its descendants is also attached, e.g. drop "food" when "meat" present).
    out: dict[str, list[str]] = {}
    for dim, nodes in senses.items():
        tree = DIMENSIONS[dim]["tree"]
        keep = {
            n for n in nodes
            if not any(o != n and n in hypernym_path(tree, o)[:-1] for o in nodes)
        }
        out[dim] = sorted(keep)
    return out


def main() -> None:
    src = json.loads(SRC.read_text())
    entities = []
    for source in ("robocasa", "libero"):
        for role_key, role in (("objects", "object"), ("fixtures", "fixture")):
            for name, props in src[source][role_key].items():
                props = set(props)
                senses = entity_senses(role, props)
                entities.append({
                    "name": name,
                    "source": source,
                    "role": role,
                    "senses": {
                        dim: [
                            {"synset": n,
                             "hypernym_path": hypernym_path(DIMENSIONS[dim]["tree"], n)}
                            for n in nodes
                        ]
                        for dim, nodes in senses.items() if nodes
                    },
                })

    # per-node member index (like a WordNet synset's lemma list)
    members: dict[str, dict[str, list[str]]] = {
        dim: {node: [] for node in spec["tree"]} for dim, spec in DIMENSIONS.items()
    }
    for ent in entities:
        for dim, sense_list in ent["senses"].items():
            for sense in sense_list:
                for node in sense["hypernym_path"]:
                    members[dim][node].append(f'{ent["source"]}:{ent["name"]}')

    dataset = {
        "meta": {
            "name": "ManipVerse-WordNet",
            "description": (
                "Faceted (WordNet-style) view of ManipVerse: independent "
                "hypernym trees per dimension; each entity attaches to its "
                "most-specific synsets in each dimension."
            ),
            "generated_by": "eval/figures_out/manipverse/generate_manipverse_wordnet.py",
            "derived_from": "manipverse.json",
        },
        "dimensions": {
            dim: {
                "description": spec["description"],
                "synsets": [
                    {
                        "name": node,
                        "hypernym": parent,
                        "depth": len(hypernym_path(spec["tree"], node)) - 1,
                        "n_members": len(set(members[dim][node])),
                    }
                    for node, parent in spec["tree"].items()
                ],
            }
            for dim, spec in DIMENSIONS.items()
        },
        "entities": entities,
    }
    (HERE / "manipverse_wordnet.json").write_text(json.dumps(dataset, indent=2) + "\n")

    # human-readable tree
    lines = []
    for dim, spec in DIMENSIONS.items():
        lines.append(f"=== {dim}: {spec['description']} ===")
        tree = spec["tree"]
        def emit(node, depth):
            uniq = sorted(set(members[dim][node]))
            direct = [
                m.split(":", 1)[1] for m in uniq
                if any(s["synset"] == node
                       for e in entities if f'{e["source"]}:{e["name"]}' == m
                       for s in e["senses"].get(dim, []))
            ]
            lines.append("  " * depth + f"- {node}  ({len(uniq)} members)"
                         + (f": {', '.join(sorted(direct)[:12])}"
                            + (", ..." if len(direct) > 12 else "") if direct else ""))
            for child, parent in tree.items():
                if parent == node:
                    emit(child, depth + 1)
        roots = [n for n, p in tree.items() if p is None]
        for r in roots:
            emit(r, 0)
        lines.append("")
    (HERE / "manipverse_wordnet.txt").write_text("\n".join(lines) + "\n")

    n_senses = sum(len(s) for e in entities for s in e["senses"].values())
    print(f"entities: {len(entities)}, dimensions: {len(DIMENSIONS)}, "
          f"sense attachments: {n_senses}")


if __name__ == "__main__":
    main()
