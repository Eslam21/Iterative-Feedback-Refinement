"""
SelfCheckGPT-based evaluation (BERTScore + MQAG).

Per-unit scores: BERTScore between each summary unit (sentence or JSON field)
and the source note. Lower BERTScore in SelfCheckGPT means MORE consistent
(less hallucinated) — see selfcheckgpt docs.

Aggregate scores: MQAG produces four divergence measures (kl_div, counting,
hellinger, total_variation) between QA distributions over candidate vs source.
"""

from __future__ import annotations

from typing import List, Optional

import pandas as pd
import torch

from config import (
    MQAG_NUM_QUESTIONS,
    SEED,
    SELFCHECK_BERT_CONSISTENT_THRESHOLD,
)
from flatten import to_candidate_text, to_sentences


# ─────────────────────────────────────────────────────────────
# Model holder (lazy + cached so we init once per process)
# ─────────────────────────────────────────────────────────────
class SelfCheckModels:
    def __init__(self, device: Optional[str] = None):
        from selfcheckgpt.modeling_mqag import MQAG
        from selfcheckgpt.modeling_selfcheck import SelfCheckBERTScore

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        torch.manual_seed(SEED)
        self.bert = SelfCheckBERTScore(rescale_with_baseline=True)
        self.mqag = MQAG(g1_model_type="race", device=self.device)


# ─────────────────────────────────────────────────────────────
# Scoring primitives
# ─────────────────────────────────────────────────────────────
def _score_bertscore(models: SelfCheckModels, sentences: List[str], document: str):
    return models.bert.predict(sentences=sentences, sampled_passages=[document])


def _score_mqag(models: SelfCheckModels, candidate: str, document: str,
                num_questions: int):
    return models.mqag.score(
        candidate=candidate,
        reference=document,
        num_questions=num_questions,
        verbose=False,
    )


def _empty_aggregate(doc_index, note_id, approach, model, fmt, reason):
    return {
        "doc_index":      doc_index,
        "note_id":        note_id,
        "approach":       approach,
        "model":          model,
        "summary_format": fmt,
        "n_units":        0,
        "bertscore_avg":  None,
        "pct_consistent": None,
        "mqag_kl_div":    None,
        "mqag_counting":  None,
        "mqag_hellinger": None,
        "mqag_total_var": None,
        "skip_reason":    reason,
    }


# ─────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────
def score_run(df: pd.DataFrame, fmt: str, models: SelfCheckModels,
              num_questions: int = MQAG_NUM_QUESTIONS,
              consistent_threshold: float = SELFCHECK_BERT_CONSISTENT_THRESHOLD,
              ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Score a loaded run's dataframe.

    Expects standardized columns: source_text, model_output, note_id, approach, model.
    Returns (unit_df, aggregate_df).
    """
    unit_rows = []
    agg_rows = []

    approach = df["approach"].iloc[0] if len(df) else "?"
    model_name = df["model"].iloc[0] if len(df) else "?"

    for i, row in df.iterrows():
        document = str(row["source_text"])
        summary = str(row["model_output"])
        note_id = row["note_id"]

        sentences = to_sentences(summary, fmt)
        candidate = to_candidate_text(summary, fmt)

        if not sentences or not candidate.strip():
            print(f"  [Skip] row {i} note_id={note_id}: empty/invalid output")
            agg_rows.append(_empty_aggregate(
                i, note_id, approach, model_name, fmt,
                reason="empty_or_invalid_output",
            ))
            continue

        try:
            bert_scores = _score_bertscore(models, sentences, document)
            mqag_scores = _score_mqag(models, candidate, document, num_questions)
        except Exception as e:
            print(f"  [Error] row {i} note_id={note_id}: {e}")
            agg_rows.append(_empty_aggregate(
                i, note_id, approach, model_name, fmt,
                reason=f"error:{type(e).__name__}",
            ))
            continue

        for j, (unit, score) in enumerate(zip(sentences, bert_scores)):
            unit_rows.append({
                "doc_index":  i,
                "note_id":    note_id,
                "approach":   approach,
                "model":      model_name,
                "unit_index": j,
                "unit_type":  "field" if fmt == "json" else "sentence",
                "unit_text":  unit,
                "bertscore":  round(float(score), 4),
                "consistent": int(score < consistent_threshold),
            })

        avg_bert = round(float(sum(bert_scores) / len(bert_scores)), 4)
        pct_consistent = round(
            sum(1 for s in bert_scores if s < consistent_threshold)
            / len(bert_scores) * 100, 1,
        )
        agg_rows.append({
            "doc_index":      i,
            "note_id":        note_id,
            "approach":       approach,
            "model":          model_name,
            "summary_format": fmt,
            "n_units":        len(sentences),
            "bertscore_avg":  avg_bert,
            "pct_consistent": pct_consistent,
            "mqag_kl_div":    round(mqag_scores["kl_div"], 4),
            "mqag_counting":  round(mqag_scores["counting"], 4),
            "mqag_hellinger": round(mqag_scores["hellinger"], 4),
            "mqag_total_var": round(mqag_scores["total_variation"], 4),
            "skip_reason":    None,
        })
        print(
            f"  [{i+1}/{len(df)}] bert={avg_bert} kl={mqag_scores['kl_div']:.4f} "
            f"hell={mqag_scores['hellinger']:.4f} consistent={pct_consistent}%"
        )

    return pd.DataFrame(unit_rows), pd.DataFrame(agg_rows)
