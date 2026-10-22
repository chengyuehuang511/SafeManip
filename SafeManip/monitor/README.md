# RoboCasa Symbolic Pipeline

This directory mirrors the structure of the main SceneFlow symbolic pipeline, but for task-agnostic RoboCasa properties.

Files:
- `primitives.py`: low-level predicate implementations backed by simulator state, inferred metadata, or explicit placeholder fallbacks where needed for coverage.
- `predicates.py`: generic role entities plus natural-language-friendly predicate wrappers.
- `specs.py`: source of truth for active propositions, descriptions, variant axes, and LTL rules.
- `symbolic_properties.py`: reusable `SymbolicProperty` definitions built from the active proposition vocabulary.
- `LTLfDFA.py`: RoboCasa-local LTLf-to-DFA wrapper that keeps the root driving implementation unchanged.
- `monitor.py`: product-state-based symbolic monitor for direct simulator-state checking.
- `export_properties.py`: helper to regenerate `DOCS_SYMBOLIC_PIPELINE.md`.
- `generate_verification.py`: helper to regenerate task-guide verification artifacts under `verification/`.

Current assumptions:
- The exported atomic propositions and LTL formulas are intended to be reusable across tasks; task relevance should be decided inside predicate implementations rather than encoded directly in the LTL names.
- Symbolic binding is now role-based first: `object`, `support`, `fixture`, `button`, and `tool` are the primary entities, while variants are expressed through attributes rather than hard-coded variant entities.
- Reusable predicate families are grouped around interaction state, readiness/requirements, object outcomes, risk signals, and role-specific attributes.
- Variant axes are organized by role attributes plus higher-level requirement, risk, and outcome axes.

Current status:
- The active LTL library is task-agnostic by design: the intended default is to run the same rule bank for every RoboCasa task and let predicate truth values decide whether a rule is relevant or vacuously true.
- Many predicates are now grounded from privileged simulator state or external RoboCasa metadata.
- Several semantic attributes such as `raw`, `ready_to_eat`, `hot`, `cold`, and `liquid` remain in the vocabulary even when they require inferred or placeholder-backed implementations.
- Some sequence / hygiene / recovery concepts from the task notes are still intentionally missing from the active rule bank when they are not yet grounded well enough.
- Per-task verification reports are generated under `verification/` to make those gaps explicit instead of hiding them.

Useful outputs:
- `DOCS_SYMBOLIC_PIPELINE.md`: generated inventory of active propositions and LTL rules.
- `verification/ATTRIBUTE_IMPLEMENTATION_STATUS.md`: attribute grounding / inference / placeholder status.
- `verification/*_VERIFICATION.md`: item-by-item task-note coverage reports for the current handwritten task docs.
- `verification/TARGET50_TASK_INVENTORY.md`: target-task inventory derived from the external RoboCasa dataset registry.

### RoboCasa fixture support

Fixture support requires both geometric interior containment (the object's body
position lies in any declared interior region) and an active MuJoCo contact
below the contacted object collision geom's live center. The normal, oriented
from support toward object, must point within 60 degrees of anti-gravity.
All nested fixture geoms are eligible, including shelves, racks and articulated
parts; contact is not restricted to the fixture's main body or bottom floor.
Containment alone, side-wall contact and overhead contact do not establish
support. Fixtures with no matching declared interior region do not qualify.

This rule applies to general fixture support, correct-target fixture support
and support-reference selection. Receptacle-object checks, simulator physics,
grasp debounce settings and LTL formulas are unchanged.
