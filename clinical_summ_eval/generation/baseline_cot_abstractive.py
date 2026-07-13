"""
Baseline: Chain-of-Thought Abstractive Summarization

Two-stage summarization:
1. Analyze the clinical note (identify key information)
2. Generate summary based on analysis

Used as a stronger baseline than single-pass summarization.

PREREQUISITE:
    Start the SGLang server BEFORE running this script:
        python -m sglang.launch_server \
            --model-path meta-llama/Llama-3.1-8B-Instruct \
            --port 30000

OUTPUT:
    results/cot_abstractive/<model_name>/
        ├── summaries.csv               note_id, clinical_notes, analysis, summary, ...
        ├── statistics.json             Aggregate compression stats
        └── experiment_metrics.csv      Runtime, memory, tokens

OUTPUT COLUMN NAMES (standardized for evaluator):
    clinical_notes  Original note text (matches all other methods)
    analysis        Stage 1 reasoning output
    summary         Stage 2 final summary
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
    method: str = "cot_abstractive"
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

class ChainOfThoughtSummarizer:
    """
    Two-stage CoT summarization.
    
    Memory IS used WITHIN one note (so stage 2 can reference stage 1's analysis)
    but is RESET between notes (each note is independent).
    """
    
    def __init__(self, model_name: str, output_dir: str,
                 base_url: str = "http://127.0.0.1:30000/v1"):
        self.model_name = model_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.metrics = ExperimentMetrics(model_name=model_name)
        self.client = SGLangClient(model=model_name, base_url=base_url, temperature=0.1)
        
        self.system_prompt = (
            "You are an expert clinical documentation specialist. "
            "You analyze clinical notes systematically and create accurate summaries. "
            "You think step-by-step to ensure all critical information is captured."
        )
    
    def summarize(self, clinical_note: str) -> Dict[str, Any]:
        """Two-stage chain-of-thought summarization."""
        try:
            # Stage 1: Analyze
            analysis = self._analyze(clinical_note)
            
            # Stage 2: Summarize based on analysis
            summary = self._summarize_from_analysis()
            
            # Reset between notes (each note is independent)
            self.client.reset_conversation()
            
            return {
                'method': 'Chain-of-Thought',
                'analysis': analysis,
                'summary': summary,
                'analysis_word_count': len(analysis.split()),
                'summary_word_count': len(summary.split()),
                'summary_char_count': len(summary),
                'original_word_count': len(clinical_note.split()),
                'original_char_count': len(clinical_note),
                'compression_ratio': len(summary) / max(len(clinical_note), 1),
                'word_reduction_ratio': len(summary.split()) / max(len(clinical_note.split()), 1),
            }
        except Exception as e:
            print(f"  Error: {e}")
            self.client.reset_conversation()
            return {
                'method': 'Chain-of-Thought',
                'error': str(e),
                'analysis': '',
                'summary': '',
                'analysis_word_count': 0,
                'summary_word_count': 0,
                'summary_char_count': 0,
                'original_word_count': len(clinical_note.split()),
                'original_char_count': len(clinical_note),
                'compression_ratio': 0,
                'word_reduction_ratio': 0,
            }
    
    def _analyze(self, clinical_note: str) -> str:
        """Stage 1: Analyze clinical note systematically."""
        prompt = f"""Analyze this clinical note systematically. Identify:

1. Patient information (demographics, presenting concern)
2. Clinical events (what happened, when)
3. Findings (vitals, exam, lab results)
4. Decisions made (diagnoses, treatments)
5. Critical information (red flags, key concerns)

Provide a structured analysis with clear observations.

CLINICAL NOTE:
{clinical_note}"""
        
        response = self.client.generate(
            prompt,
            system_prompt=self.system_prompt,
            use_memory=True,  # Stage 2 will reference this
        )
        self.metrics.update_from_response(response)
        return response['reply'].strip()
    
    def _summarize_from_analysis(self) -> str:
        """Stage 2: Generate summary based on analysis (still in memory)."""
        prompt = """Based on your analysis, create a concise clinical summary.

Requirements:
- Preserve all medically relevant information
- Include critical findings
- Use proper medical terminology
- Plain English narrative format

Output only the summary. No preamble, no explanation."""
        
        response = self.client.generate(
            prompt,
            system_prompt=self.system_prompt,
            use_memory=True,  # Reference the analysis
        )
        self.metrics.update_from_response(response)
        return response['reply'].strip()
    
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
        print(f"BASELINE METRICS: Chain-of-Thought")
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
    
    # Output: results/cot_abstractive/<model_name>/
    OUTPUT_DIR = f"./results/cot_abstractive/{MODEL_NAME}"
    
    TEST_DATA_PATH = os.environ.get("TEST_DATA_PATH", "datasets/cluster_testing_data.csv")
    
    # ---------- Run ----------
    
    print("="*80)
    print(f"BASELINE: Chain-of-Thought Abstractive Summarization")
    print(f"Model:  {MODEL_NAME}")
    print(f"Server: {SGLANG_BASE_URL}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Note: SGLang server must already be running.")
    print("="*80)
    
    try:
        test_data = pd.read_csv(TEST_DATA_PATH, index_col=0)
        print(f"\nLoaded {len(test_data)} test notes")
        
        summarizer = ChainOfThoughtSummarizer(
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
            'avg_analysis_word_count': float(results_df['analysis_word_count'].mean()),
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