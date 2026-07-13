"""
Baseline: Standard Abstractive Summarization

Single-pass LLM summarization of clinical notes. The simplest LLM baseline -
no iteration, no decomposition, no schema. Just "read this, write a summary".

PREREQUISITE:
    Start the SGLang server BEFORE running this script:
        python -m sglang.launch_server \
            --model-path meta-llama/Llama-3.1-8B-Instruct \
            --port 30000

OUTPUT:
    results/standard_abstractive/<model_name>/
        ├── summaries.csv               note_id, clinical_notes, summary, ...
        ├── statistics.json             Aggregate compression stats
        └── experiment_metrics.csv      Runtime, memory, tokens

OUTPUT COLUMN NAMES (standardized for evaluator):
    clinical_notes  Original note text (matches schema methods)
    summary         Generated summary text
"""

import json
import time
import psutil
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

import pandas as pd
from sglang_client import SGLangClient


# ============================================================
# Metrics
# ============================================================

@dataclass
class ExperimentMetrics:
    model_name: str
    method: str = "standard_abstractive"
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    total_duration: Optional[float] = None
    peak_memory_mb: float = 0.0
    api_calls: int = 0
    total_tokens: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    
    def update_memory(self):
        process = psutil.Process(os.getpid())
        current_mb = process.memory_info().rss / 1024 / 1024
        self.peak_memory_mb = max(self.peak_memory_mb, current_mb)
    
    def update_from_response(self, response: Dict[str, Any]):
        self.api_calls += 1
        usage = response.get("usage", {})
        self.total_tokens += usage.get("total_tokens", 0)
        self.prompt_tokens += usage.get("prompt_tokens", 0)
        self.completion_tokens += usage.get("completion_tokens", 0)
        self.update_memory()
    
    def finalize(self):
        self.end_time = time.time()
        self.total_duration = self.end_time - self.start_time


# ============================================================
# Summarizer
# ============================================================

class StandardAbstractiveSummarizer:
    """Single-pass abstractive summarization. No memory between notes."""
    
    def __init__(self, model_name: str, output_dir: str,
                 base_url: str = "http://127.0.0.1:30000/v1"):
        self.model_name = model_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.metrics = ExperimentMetrics(model_name=model_name)
        self.client = SGLangClient(model=model_name, base_url=base_url, temperature=0.1)
        
        self.system_prompt = (
            "You are an expert clinical documentation specialist. "
            "Create concise, accurate summaries of clinical notes while "
            "preserving all medically relevant information."
        )
    
    def summarize(self, clinical_note: str) -> Dict[str, Any]:
        """Generate a single-pass abstractive summary."""
        prompt = f"""Summarize the following clinical note. Preserve all medically relevant information including:
- Patient demographics and chief complaint
- Key findings and diagnoses
- Vital signs and lab values
- Medications and treatment plans
- Critical observations

Output only the summary in plain English. No preamble, no explanation.

CLINICAL NOTE:
{clinical_note}"""
        
        try:
            response = self.client.generate(
                prompt,
                system_prompt=self.system_prompt,
                use_memory=False,  # Each summary is independent
            )
            self.metrics.update_from_response(response)
            
            summary = response['reply'].strip()
            
            return {
                'method': 'Standard Abstractive',
                'summary': summary,
                'summary_word_count': len(summary.split()),
                'summary_char_count': len(summary),
                'original_word_count': len(clinical_note.split()),
                'original_char_count': len(clinical_note),
                'compression_ratio': len(summary) / max(len(clinical_note), 1),
                'word_reduction_ratio': len(summary.split()) / max(len(clinical_note.split()), 1),
            }
        except Exception as e:
            print(f"  Error: {e}")
            return {
                'method': 'Standard Abstractive',
                'summary': '',
                'error': str(e),
                'summary_word_count': 0,
                'summary_char_count': 0,
                'original_word_count': len(clinical_note.split()),
                'original_char_count': len(clinical_note),
                'compression_ratio': 0,
                'word_reduction_ratio': 0,
            }
    
    def save_metrics(self):
        """Save experiment metrics."""
        self.metrics.finalize()
        
        metrics_dict = {
            'model': self.metrics.model_name,
            'method': self.metrics.method,
            'total_duration_seconds': self.metrics.total_duration,
            'total_duration_minutes': self.metrics.total_duration / 60,
            'total_duration_hours': self.metrics.total_duration / 3600,
            'peak_memory_mb': self.metrics.peak_memory_mb,
            'peak_memory_gb': self.metrics.peak_memory_mb / 1024,
            'total_api_calls': self.metrics.api_calls,
            'total_tokens': self.metrics.total_tokens,
            'prompt_tokens': self.metrics.prompt_tokens,
            'completion_tokens': self.metrics.completion_tokens,
            'avg_tokens_per_call': self.metrics.total_tokens / max(self.metrics.api_calls, 1),
        }
        
        pd.DataFrame([metrics_dict]).to_csv(self.output_dir / 'experiment_metrics.csv', index=False)
        
        print(f"\n{'='*60}")
        print(f"BASELINE METRICS: Standard Abstractive")
        print(f"{'='*60}")
        print(f"Duration: {self.metrics.total_duration/60:.2f} min")
        print(f"Peak Memory: {self.metrics.peak_memory_mb/1024:.2f} GB")
        print(f"API Calls: {self.metrics.api_calls}")
        print(f"Total Tokens: {self.metrics.total_tokens:,}")


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    # ---------- Configuration ----------
    
    # Model name as known to the running SGLang server.
    MODEL_NAME = os.environ.get("MODEL_NAME", "Llama-4-Scout-17B-16E-Instruct-FP8")
    
    # SGLang server (must already be running)
    SGLANG_BASE_URL = "http://127.0.0.1:30000/v1"
    
    # Output: results/standard_abstractive/<model_name>/
    OUTPUT_DIR = f"./results/standard_abstractive/{MODEL_NAME}"
    
    TEST_DATA_PATH = os.environ.get("TEST_DATA_PATH", "datasets/cluster_testing_data.csv")
    
    # ---------- Run ----------
    
    print("="*80)
    print(f"BASELINE: Standard Abstractive Summarization")
    print(f"Model:  {MODEL_NAME}")
    print(f"Server: {SGLANG_BASE_URL}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Note: SGLang server must already be running.")
    print("="*80)
    
    try:
        test_data = pd.read_csv(TEST_DATA_PATH, index_col=0)
        print(f"\nLoaded {len(test_data)} test notes")
        
        summarizer = StandardAbstractiveSummarizer(
            model_name=MODEL_NAME,
            output_dir=OUTPUT_DIR,
            base_url=SGLANG_BASE_URL,
        )
        
        results = []
        print(f"\n=== Summarizing {len(test_data)} notes ===")
        
        for idx, note in enumerate(test_data['clinical_notes'].values):
            print(f"  Note {idx + 1}/{len(test_data)}")
            result = summarizer.summarize(note)
            result['note_id'] = idx
            # Standardized: 'clinical_notes' matches schema methods
            result['clinical_notes'] = note
            results.append(result)
        
        results_df = pd.DataFrame(results)
        results_df.to_csv(summarizer.output_dir / 'summaries.csv', index=False)
        
        stats = {
            'total_notes': len(results_df),
            'avg_compression_ratio': float(results_df['compression_ratio'].mean()),
            'avg_word_reduction_ratio': float(results_df['word_reduction_ratio'].mean()),
            'avg_original_word_count': float(results_df['original_word_count'].mean()),
            'avg_summary_word_count': float(results_df['summary_word_count'].mean()),
            'min_compression_ratio': float(results_df['compression_ratio'].min()),
            'max_compression_ratio': float(results_df['compression_ratio'].max()),
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