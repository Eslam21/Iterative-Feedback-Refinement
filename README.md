# Iterative Feedback–Refinement for Faithful Structured Clinical-Note Representation

Code and preprocessed data for the paper *"Iterative Feedback–Refinement for
Faithful Structured Clinical Notes Representation."* The framework reframes
structured clinical documentation as **adaptive schema induction**: instead of
generating instances from a fixed template, it iteratively optimizes a shared
JSON schema so that improvements propagate to every downstream extraction.

A dual-LLM loop alternates between an **evaluator** (scores the current schema
against source notes and emits categorical feedback) and a **refiner**
(incorporates aggregated feedback into the next schema version). The final schema
is held fixed and applied to a held-out validation set. Evaluated on the public
MTSamples corpus across eight open-weight LLMs, iterative schema refinement
achieves the highest faithfulness on all three LLM-judge metrics while compressing
notes to ~34.5% of their original length — driven by higher recall at comparable
precision (reduced omission without added hallucination).
