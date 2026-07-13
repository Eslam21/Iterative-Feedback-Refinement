"""
Baseline: BERT-Centroid Extractive Summarization

Classical embedding-based extractive summarization using Bio_ClinicalBERT.
Selects sentences from the original note that are most similar to the
document centroid (average of all sentence embeddings).

This is the only baseline that does NOT use an LLM - it's a deterministic
embedding-based method. It serves as a "no-LLM" reference point in the
comparison.

PIPELINE:
    1. Tokenize note into sentences
    2. Embed each sentence using Bio_ClinicalBERT (mean pooling)
    3. Compute document centroid (average of sentence embeddings)
    4. Score each sentence by cosine similarity to the centroid
    5. Select top-K sentences based on target compression ratio
    6. Reorder selected sentences to match original order

NO PREREQUISITE:
    Unlike other baselines, this does NOT need an SGLang server - it loads
    the BERT model directly via HuggingFace transformers.

OUTPUT:
    results/bert_centroid/<model_name>/
        ├── summaries.csv               note_id, clinical_notes, summary, ...
        ├── summaries_detailed.json     Per-sentence scores and selection details
        ├── statistics.json             Aggregate compression stats
        └── experiment_metrics.csv      Runtime, memory (no API/token tracking)

OUTPUT COLUMN NAMES (standardized for evaluator):
    clinical_notes  Original note text (matches all other methods)
    summary         Concatenated selected sentences

NOTE ON MODEL NAMING:
    The "model" here is the BERT model used for embedding (e.g., Bio_ClinicalBERT).
    Output uses the short name (last path segment) so the evaluator lists this
    baseline alongside LLM-based methods with consistent naming.
"""

import json
import os
import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import psutil

import nltk
import torch
from nltk.tokenize import sent_tokenize
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoModel, AutoTokenizer

# Ensure NLTK sentence tokenizer is available (try both punkt and punkt_tab,
# the latter is needed for NLTK 3.8.2+)
for resource in ['punkt_tab', 'punkt']:
    try:
        nltk.data.find(f'tokenizers/{resource}')
    except LookupError:
        try:
            nltk.download(resource, quiet=True)
        except Exception:
            pass


# ============================================================
# Metrics (same shape as LLM baselines for fair comparison)
# ============================================================

@dataclass
class ExperimentMetrics:
    """Runtime and memory tracker. No API/token tracking (no LLM used)."""
    model_name: str
    method: str = "bert_centroid_extractive"
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    total_duration: Optional[float] = None
    peak_memory_mb: float = 0.0
    # API/token fields included for cross-method compatibility (always 0 here).
    api_calls: int = 0
    total_tokens: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    
    def update_memory(self):
        process = psutil.Process(os.getpid())
        current_mb = process.memory_info().rss / 1024 / 1024
        self.peak_memory_mb = max(self.peak_memory_mb, current_mb)
    
    def finalize(self):
        self.end_time = time.time()
        self.total_duration = self.end_time - self.start_time


# ============================================================
# Summarizer
# ============================================================

class BERTCentroidSummarizer:
    """BERT-Centroid extractive summarization with adaptive compression."""
    
    def __init__(self,
                 bert_model: str = 'emilyalsentzer/Bio_ClinicalBERT',
                 output_dir: str = './results/bert_centroid/Bio_ClinicalBERT',
                 device: Optional[str] = None):
        """
        Args:
            bert_model: HuggingFace model name for sentence embeddings.
                Bio_ClinicalBERT is recommended for clinical text.
            output_dir: Where to save outputs.
            device: 'cuda', 'cpu', or None to auto-detect.
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.bert_model_name = bert_model
        
        # FIXED: original code referenced self.device before assigning it
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        print(f"Using device: {self.device}")
        print(f"Loading BERT model: {bert_model}")
        
        # Short model name for outputs/comparison
        self.model_name = bert_model.split('/')[-1]
        self.metrics = ExperimentMetrics(model_name=self.model_name)
        
        self.tokenizer = AutoTokenizer.from_pretrained(bert_model)
        self.model = AutoModel.from_pretrained(bert_model)
        self.model.to(self.device)
        self.model.eval()
        
        print("✓ BERT model loaded")
    
    # ----------------------------------------
    # Text processing
    # ----------------------------------------
    
    def preprocess_text(self, text: str) -> str:
        """Light cleanup before sentence tokenization."""
        text = re.sub(r'\s+', ' ', text)
        # Keep punctuation that helps sentence boundaries
        text = re.sub(r'[^\w\s\.\,\-\/\:\;\(\)]', '', text)
        return text.strip()
    
    def extract_sentences(self, text: str) -> List[str]:
        """Split text into sentences, dropping very short fragments."""
        text = self.preprocess_text(text)
        if not text:
            return []
        sentences = sent_tokenize(text)
        # Drop short fragments (less than 4 words) - usually section headers
        sentences = [s for s in sentences if len(s.split()) > 3]
        return sentences
    
    def get_bert_embeddings(self, sentences: List[str]) -> np.ndarray:
        """Embed sentences using mean pooling over token embeddings.
        
        Mean pooling captures contributions from all tokens, which works
        better than [CLS] for clinical text dense with medical terminology.
        
        Returns:
            numpy array of shape (n_sentences, embedding_dim)
        """
        if not sentences:
            return np.array([])
        
        embeddings = []
        
        with torch.no_grad():
            for sentence in sentences:
                inputs = self.tokenizer(
                    sentence,
                    return_tensors='pt',
                    padding=True,
                    truncation=True,
                    max_length=512
                ).to(self.device)
                
                outputs = self.model(**inputs)
                
                # Mean pooling over real tokens (ignoring padding)
                attention_mask = inputs['attention_mask']
                token_embeddings = outputs.last_hidden_state
                
                mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
                sum_embeddings = torch.sum(token_embeddings * mask_expanded, 1)
                sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
                mean_pooled = (sum_embeddings / sum_mask).cpu().numpy()
                embeddings.append(mean_pooled[0])
        
        return np.array(embeddings)
    
    # ----------------------------------------
    # Summarization
    # ----------------------------------------
    
    def summarize(self, text: str,
                  target_ratio: float = 0.5,
                  min_sentences: int = 3,
                  max_sentences: int = 15) -> Dict[str, Any]:
        """Generate a centroid-based extractive summary.
        
        Args:
            text: Clinical note to summarize.
            target_ratio: Desired compression ratio (0.5 = ~50% of original).
            min_sentences: Floor on summary length.
            max_sentences: Ceiling on summary length.
        """
        self.metrics.update_memory()
        
        if not text or not text.strip():
            return self._empty_result(text, target_ratio, reason='empty_input')
        
        sentences = self.extract_sentences(text)
        
        # Edge case: very short notes - return as-is
        if len(sentences) <= min_sentences:
            summary = ' '.join(sentences)
            return self._build_result(
                summary=summary,
                selected_sentences=sentences,
                scores=[1.0] * len(sentences),
                num_selected=len(sentences),
                total_sentences=len(sentences),
                target_ratio=target_ratio,
                text=text,
                selection_method='all_sentences_kept',
            )
        
        # Compute target sentence count from compression ratio
        avg_sentence_length = len(text) / len(sentences)
        target_length = len(text) * target_ratio
        num_sentences = int(target_length / max(avg_sentence_length, 1))
        num_sentences = max(min_sentences, min(num_sentences, max_sentences, len(sentences)))
        
        # Embed → centroid → score by similarity
        embeddings = self.get_bert_embeddings(sentences)
        doc_embedding = np.mean(embeddings, axis=0, keepdims=True)
        similarities = cosine_similarity(embeddings, doc_embedding).flatten()
        
        # Top-K by similarity, then resort to original order for coherence
        top_indices = similarities.argsort()[-num_sentences:][::-1]
        top_indices_sorted = sorted(top_indices)
        
        selected_sentences = [sentences[i] for i in top_indices_sorted]
        selected_scores = [float(similarities[i]) for i in top_indices_sorted]
        summary = ' '.join(selected_sentences)
        
        return self._build_result(
            summary=summary,
            selected_sentences=selected_sentences,
            scores=selected_scores,
            num_selected=num_sentences,
            total_sentences=len(sentences),
            target_ratio=target_ratio,
            text=text,
            selection_method='bert_centroid',
        )
    
    def _build_result(self, summary: str, selected_sentences: List[str],
                     scores: List[float], num_selected: int, total_sentences: int,
                     target_ratio: float, text: str,
                     selection_method: str) -> Dict[str, Any]:
        """Build the standard result dict."""
        return {
            'method': 'BERT-Centroid',
            'summary': summary,
            'selected_sentences': selected_sentences,
            'scores': scores,
            'num_sentences_selected': num_selected,
            'total_sentences': total_sentences,
            'target_ratio': target_ratio,
            'actual_compression_ratio': len(summary) / max(len(text), 1),
            'selection_method': selection_method,
            # Standard fields shared across all baselines
            'summary_word_count': len(summary.split()),
            'summary_char_count': len(summary),
            'original_word_count': len(text.split()),
            'original_char_count': len(text),
            'compression_ratio': len(summary) / max(len(text), 1),
            'word_reduction_ratio': len(summary.split()) / max(len(text.split()), 1),
        }
    
    def _empty_result(self, text: str, target_ratio: float, reason: str) -> Dict[str, Any]:
        """Return a zero-filled result for empty/invalid input."""
        return {
            'method': 'BERT-Centroid',
            'summary': '',
            'selected_sentences': [],
            'scores': [],
            'num_sentences_selected': 0,
            'total_sentences': 0,
            'target_ratio': target_ratio,
            'actual_compression_ratio': 0.0,
            'selection_method': reason,
            'summary_word_count': 0,
            'summary_char_count': 0,
            'original_word_count': len(text.split()) if text else 0,
            'original_char_count': len(text) if text else 0,
            'compression_ratio': 0.0,
            'word_reduction_ratio': 0.0,
        }
    
    # ----------------------------------------
    # Metrics
    # ----------------------------------------
    
    def save_metrics(self):
        """Save final metrics (runtime + memory)."""
        self.metrics.finalize()
        
        metrics_dict = {
            'model': self.metrics.model_name,
            'method': self.metrics.method,
            'total_duration_seconds': self.metrics.total_duration,
            'total_duration_minutes': self.metrics.total_duration / 60,
            'total_duration_hours': self.metrics.total_duration / 3600,
            'peak_memory_mb': self.metrics.peak_memory_mb,
            'peak_memory_gb': self.metrics.peak_memory_mb / 1024,
            'total_api_calls': 0,
            'total_tokens': 0,
            'prompt_tokens': 0,
            'completion_tokens': 0,
            'avg_tokens_per_call': 0,
        }
        
        pd.DataFrame([metrics_dict]).to_csv(
            self.output_dir / 'experiment_metrics.csv', index=False
        )
        
        print(f"\n{'='*60}")
        print(f"BASELINE METRICS: BERT-Centroid (no LLM)")
        print(f"{'='*60}")
        print(f"Duration: {self.metrics.total_duration/60:.2f} min")
        print(f"Peak Memory: {self.metrics.peak_memory_mb/1024:.2f} GB")


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    # ---------- Configuration ----------
    
    BERT_MODEL = 'emilyalsentzer/Bio_ClinicalBERT'
    
    OUTPUT_DIR = f"./results/base_extractive/bert_centroid"
    
    # Compression parameters
    TARGET_RATIO = 0.5
    MIN_SENTENCES = 3
    MAX_SENTENCES = 15
    
    TEST_DATA_PATH = os.environ.get("TEST_DATA_PATH", "datasets/cluster_testing_data.csv")
    
    # ---------- Run ----------
    
    print("="*80)
    print(f"BASELINE: BERT-Centroid Extractive Summarization")
    print(f"Model:  {BERT_MODEL}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Target compression: {TARGET_RATIO:.0%}")
    print(f"Sentence bounds: [{MIN_SENTENCES}, {MAX_SENTENCES}]")
    print("="*80)
    
    try:
        summarizer = BERTCentroidSummarizer(
            bert_model=BERT_MODEL,
            output_dir=OUTPUT_DIR,
            device=None,  # Auto-detect
        )
        
        test_data = pd.read_csv(TEST_DATA_PATH, index_col=0)
        print(f"\nLoaded {len(test_data)} test notes")
        
        results = []
        print(f"\n=== Summarizing {len(test_data)} notes ===")
        
        for idx, note in enumerate(test_data['clinical_notes'].values):
            print(f"  Note {idx + 1}/{len(test_data)}", end=' ')
            result = summarizer.summarize(
                text=note,
                target_ratio=TARGET_RATIO,
                min_sentences=MIN_SENTENCES,
                max_sentences=MAX_SENTENCES,
            )
            result['note_id'] = idx
            # Standardized: 'clinical_notes' matches all other methods
            result['clinical_notes'] = note
            results.append(result)
            print(f"  {result['total_sentences']} → {result['num_sentences_selected']} "
                  f"sentences ({result['actual_compression_ratio']:.1%})")
        
        # CSV: drop list/array columns (not CSV-friendly)
        # JSON: full detail with per-sentence scores
        results_df = pd.DataFrame(results)
        csv_columns = [c for c in results_df.columns
                      if c not in ('selected_sentences', 'scores')]
        results_df[csv_columns].to_csv(
            summarizer.output_dir / 'summaries.csv', index=False
        )
        
        with open(summarizer.output_dir / 'summaries_detailed.json', 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        stats = {
            'total_notes': len(results_df),
            'target_compression_ratio': TARGET_RATIO,
            'min_sentences': MIN_SENTENCES,
            'max_sentences': MAX_SENTENCES,
            'avg_compression_ratio': float(results_df['compression_ratio'].mean()),
            'std_compression_ratio': float(results_df['compression_ratio'].std()),
            'min_compression_ratio': float(results_df['compression_ratio'].min()),
            'max_compression_ratio': float(results_df['compression_ratio'].max()),
            'avg_sentences_selected': float(results_df['num_sentences_selected'].mean()),
            'avg_total_sentences': float(results_df['total_sentences'].mean()),
            'avg_original_word_count': float(results_df['original_word_count'].mean()),
            'avg_summary_word_count': float(results_df['summary_word_count'].mean()),
        }
        
        with open(summarizer.output_dir / 'statistics.json', 'w') as f:
            json.dump(stats, f, indent=2)
        
        print("\n=== Statistics ===")
        for k, v in stats.items():
            print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
        
        summarizer.save_metrics()
        print("\n✓ Baseline experiment completed!")
        print(f"\nNext step: python run_evaluation.py")
        
    except Exception as e:
        print(f"\n✗ Failed: {e}")
        import traceback
        traceback.print_exc()
