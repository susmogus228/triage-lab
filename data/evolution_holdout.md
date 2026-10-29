# Evolution holdout, version 1

A separate AI author composed and labelled the 126 English messages without consulting model code, predictions, or training/development/test files. The author did receive the eleven topic names and the abbreviated category mapping before writing. These are AI-authored synthetic examples, not actual customer requests and not independent human annotation. The messages and labels reflect one author's judgments and may contain errors or accidental semantic similarity to other datasets.

- 66 known single-intent cases, 6 per known category.
- 30 multi-intent cases, expected REVIEW because one forced class would omit a distinct intent.
- 26 unrelated and 4 ambiguous cases, expected REVIEW.
- Known classes follow the supplied narrow mapping: ORDER includes tracking and cancel purchase; CANCEL is contract termination fees; DELIVERY is available methods and delivery periods; SHIPPING is address amendment; SUBSCRIPTION concerns newsletters.
- Includes known single-topic requests with conjunctions and multi-topic requests with different constructions, but language and sample construction remain limited and possibly patterned.
- Cases have one `text` field; production combines subject and body. Generalization to that path is untested.
- No participants or user submissions were collected; results cannot estimate performance on real support tickets or demographic groups.
- A single author is insufficient to establish label reliability. Ambiguous labels must be documented transparently with any sensitivity analysis.

Before inspecting model outputs, freeze model selection, training sets, review threshold, and code. Treat this holdout as consumed after its results are examined; do not modify it to improve metrics or reuse it as an untouched test for later tuning. A new independent test is required after changes based on these results.

SHA-256 (`evolution_holdout.jsonl`): `de368f082d8af5fd15c2187f052497196d6abbeb677bb0a016e026aceb29f457`
