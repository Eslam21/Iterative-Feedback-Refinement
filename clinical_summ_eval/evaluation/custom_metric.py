import json
from typing import List, Optional, Union
import asyncio

from deepeval.test_case import (
    LLMTestCase,
    LLMTestCaseParams,
    ConversationalTestCase,
)
from deepeval.metrics import BaseMetric
from deepeval.models import DeepEvalBaseLLM
from deepeval.utils import get_or_create_event_loop, prettify_list
from deepeval.metrics.utils import (
    construct_verbose_logs,
    trimAndLoadJson,
    check_llm_test_case_params,
    initialize_model,
)
from deepeval.metrics.summarization.template import SummarizationTemplate
from deepeval.metrics.faithfulness.template import FaithfulnessTemplate
from deepeval.metrics.indicator import metric_progress_indicator
from deepeval.metrics.summarization.schema import *
from deepeval.metrics.faithfulness.schema import *

from pydantic import BaseModel, Field

from eval_logging import get_logger

log = get_logger(__name__)

required_params: List[LLMTestCaseParams] = [
    LLMTestCaseParams.INPUT,
    LLMTestCaseParams.ACTUAL_OUTPUT,
]


# ─────────────────────────────────────────────────────────────
# Safe helpers — these are the core of the NaN fix. Every place
# that used to `raise ValueError` on a length mismatch or build a
# Pydantic model directly from raw judge JSON now degrades instead.
# ─────────────────────────────────────────────────────────────
def _reconcile_pair(a: list, b: list, pad="idk"):
    """Make two lists the same length without ever raising.

    Length mismatches between question lists and answer lists are EXPECTED
    on messy clinical text / flattened JSON schemas — the judge drops or
    merges items. Truncating to the shorter list and padding the answer
    side with a neutral 'idk' keeps scoring honest (an unanswered question
    counts as not-covered) while never crashing the metric.
    """
    if len(a) == len(b):
        return a, b
    log.warn("length mismatch %d vs %d -> reconciling", len(a), len(b))
    n = min(len(a), len(b))
    a2, b2 = list(a[:n]), list(b[:n])
    # If one side was longer, keep its extra questions but mark unanswered.
    longer = a if len(a) > len(b) else b
    for _ in range(abs(len(a) - len(b))):
        a2.append(longer[len(a2)] if a is longer else pad)
        b2.append(pad if b is longer else longer[len(b2)])
    return a2[: max(len(a), len(b))], b2[: max(len(a), len(b))]


def _safe_complex_questions(raw_list) -> "List[ComplexQuestion]":
    """Build ComplexQuestion objects, skipping/repairing malformed entries.

    A missing or non-int `importance`, or a non-dict element, no longer
    takes down the whole metric (and, via the shared async batch, the
    sibling GEval metrics for the same row).
    """
    out: List[ComplexQuestion] = []
    for e in raw_list or []:
        try:
            if not isinstance(e, dict):
                continue
            imp = e.get("importance", 3)
            try:
                imp = int(imp)
            except (TypeError, ValueError):
                imp = 3
            imp = min(5, max(1, imp))
            q = str(e.get("question", "")).strip()
            ans = str(e.get("answer", "")).strip()
            if not q:
                continue
            out.append(ComplexQuestion(question=q, answer=ans or "idk", importance=imp))
        except Exception as exc:  # noqa: BLE001 - never propagate
            log.warn("dropping malformed complex question: %s", exc)
    return out


def _safe_complex_verdicts(raw_list, answers, questions) -> "List[ComplexQuestionVerdict]":
    """Build ComplexQuestionVerdict objects defensively.

    Reconciles verdict count to the answers list and clamps scores so a
    stray string/out-of-range score from the judge can't raise.
    """
    out: List[ComplexQuestionVerdict] = []
    raw_list = raw_list or []
    n = min(len(raw_list), len(answers), len(questions))
    for i in range(n):
        e = raw_list[i]
        try:
            d = e if isinstance(e, dict) else e.dict()
        except Exception:  # noqa: BLE001
            d = {}
        try:
            score = int(d.get("score", 0))
        except (TypeError, ValueError):
            score = 0
        score = min(5, max(0, score))
        out.append(
            ComplexQuestionVerdict(
                score=score,
                reason=str(d.get("reason", "") or ""),
                original_answer=answers[i]["original_answer"],
                summary_answer=answers[i]["summary_answer"],
                question=questions[i].question,
            )
        )
    # Any answered questions with no verdict -> score 0 (conservative).
    for i in range(n, min(len(answers), len(questions))):
        out.append(
            ComplexQuestionVerdict(
                score=0,
                reason="no verdict returned by judge",
                original_answer=answers[i]["original_answer"],
                summary_answer=answers[i]["summary_answer"],
                question=questions[i].question,
            )
        )
    return out

class ComplexQuestion(BaseModel):
    question: str
    answer: str
    importance: int = Field(description="1-5, with 1 being not important and 5 being most important")

class ComplexQuestions(BaseModel):
    questions: List[ComplexQuestion]

class ComplexQuestionVerdictOutput(BaseModel):
    score: int
    reason: str

class ComplexQuestionVerdict(ComplexQuestionVerdictOutput):
    original_answer: str
    summary_answer: str
    question: str

class ComplexQuestionsVerdictsOutputs(BaseModel):
    verdicts: List[ComplexQuestionVerdictOutput]


class CustomSummarizationMetric(BaseMetric):
    def __init__(
        self,
        threshold: float = 0.5,
        n: int = 5,
        n_complex_questions: int = 5,
        model: Optional[Union[str, DeepEvalBaseLLM]] = None,
        assessment_questions: Optional[List[str]] = None,
        include_reason: bool = True,
        async_mode=True,
        strict_mode: bool = False,
        verbose_mode: bool = False,
        truths_extraction_limit: Optional[int] = None,
        fmt: str = "text",
        skip_simple_coverage: bool = True,
    ):
        self.threshold = 1 if strict_mode else threshold
        self.model, self.using_native_model = initialize_model(model)
        self.evaluation_model = self.model.get_model_name()
        # "text" = prose summary, "json" = filled clinical schema. Drives
        # schema-fair prompt wording so structured key:value output is not
        # penalised for "not reading like a summary".
        self.fmt = fmt
        # The simple 5-question coverage path (SummarizationTemplate) does
        # NOT enter the final score — score = F1(alignment, complex
        # coverage). It only enriched the `reason`. Skipping it removes ~3
        # judge calls per note with zero effect on summ_score. Default on.
        self.skip_simple_coverage = skip_simple_coverage

        if assessment_questions is not None and len(assessment_questions) == 0:
            self.assessment_questions = None
        else:
            self.assessment_questions = assessment_questions

        self.complex_assessment_questions = None
        self.include_reason = include_reason
        self.n = n
        self.n_complex_questions = n_complex_questions
        self.async_mode = async_mode
        self.strict_mode = strict_mode
        self.verbose_mode = verbose_mode

        self.truths_extraction_limit = truths_extraction_limit
        if self.truths_extraction_limit is not None:
            self.truths_extraction_limit = max(self.truths_extraction_limit, 0)

    def measure(
        self,
        test_case: Union[LLMTestCase, ConversationalTestCase],
        _show_indicator: bool = True, *args, **kwargs
    ) -> float:
        if isinstance(test_case, ConversationalTestCase):
            test_case = test_case.turns[0]
        check_llm_test_case_params(
            test_case,
            required_params,
            input_image_count=0,
            actual_output_image_count=0,
            metric=self
        )

        self.evaluation_cost = 0 if self.using_native_model else None
        with metric_progress_indicator(self, _show_indicator=_show_indicator):
            if self.async_mode:
                loop = get_or_create_event_loop()
                loop.run_until_complete(
                    self.a_measure(test_case, _show_indicator=False)
                )
            else:
                self.truths: str = self._generate_truths(test_case.input)
                self.claims: List[str] = self._generate_claims(
                    test_case.actual_output
                )
                # Simple coverage is skipped by default — it never entered
                # the final score and was always None on disk. Keep the sync
                # path consistent with a_measure.
                if self.skip_simple_coverage:
                    self.coverage_verdicts: List[SummarizationCoverageVerdict] = []
                    if self.assessment_questions is None:
                        self.assessment_questions = []
                else:
                    self.coverage_verdicts = (
                        self._generate_coverage_verdicts(test_case)
                    )
                self.alignment_verdicts: List[SummarizationAlignmentVerdict] = (
                    self._generate_alignment_verdicts()
                )
                self.complex_coverage_verdicts: List[ComplexQuestionVerdict] = (
                    self._generate_complex_coverage_verdicts(test_case)
                )

                alignment_score = self._calculate_score(ScoreType.ALIGNMENT)
                complex_coverage_scores = [e.score / 5 for e in self.complex_coverage_verdicts]
                complex_coverage_score = (
                    sum(complex_coverage_scores) / len(complex_coverage_scores)
                    if complex_coverage_scores else 0.0
                )

                self.score_breakdown = {
                    ScoreType.ALIGNMENT.value: alignment_score,
                    "complex_coverage": complex_coverage_score,
                }

                # F1 between alignment (precision) and complex coverage
                # (recall) — identical to the async path so sync/async never
                # disagree on what summ_score means.
                precision = alignment_score
                recall = complex_coverage_score
                self.score = (
                    2 * (precision * recall) / (precision + recall)
                    if (precision + recall) > 0 else 0
                )
                self.reason = self._generate_reason()
                self.success = self.score >= self.threshold

                logs = {
                    'claims': self.claims,
                    'assessment_questions': self.assessment_questions,
                    'complex_assessment_questions': [e.dict() for e in self.complex_assessment_questions],
                    'alignment_verdicts': [v.dict() for v in self.alignment_verdicts],
                    'complex_coverage_verdicts': [v.dict() for v in self.complex_coverage_verdicts],
                    'alignment_score': alignment_score,
                    'complex_coverage_score': complex_coverage_score,
                    'score': self.score,
                    'reason': self.reason,
                    'success': self.success,
                }
                self.verbose_logs = json.dumps(logs)

                return self.score

    async def a_measure(
        self,
        test_case: Union[LLMTestCase, ConversationalTestCase],
        _show_indicator: bool = True, *args, **kwargs
    ) -> float:
        if isinstance(test_case, ConversationalTestCase):
            test_case = test_case.turns[0]
        check_llm_test_case_params(
            test_case,
            required_params,
            input_image_count=0,
            actual_output_image_count=0,
            metric=self
        )

        self.evaluation_cost = 0 if self.using_native_model else None
        with metric_progress_indicator(
            self,
            async_mode=True,
            _show_indicator=_show_indicator,
        ):
            try:
                self.truths, self.claims = await asyncio.gather(
                    self._a_generate_truths(test_case.input),
                    self._a_generate_claims(test_case.actual_output),
                )

                if self.skip_simple_coverage:
                    # Simple coverage does not affect the score; skip its
                    # ~3 judge calls. coverage_verdicts stays empty, which
                    # _a_generate_reason already guards against.
                    (
                        self.complex_coverage_verdicts,
                        self.alignment_verdicts,
                    ) = await asyncio.gather(
                        self._a_generate_complex_coverage_verdicts(test_case),
                        self._a_generate_alignment_verdicts(),
                    )
                    self.coverage_verdicts = []
                    if self.assessment_questions is None:
                        self.assessment_questions = []
                else:
                    (
                        self.complex_coverage_verdicts, # List[ComplexQuestionVerdict]
                        self.coverage_verdicts, # List[SummarizationCoverageVerdict]
                        self.alignment_verdicts, # List[SummarizationAlignmentVerdict]
                    ) = await asyncio.gather(
                        self._a_generate_complex_coverage_verdicts(test_case),
                        self._a_generate_coverage_verdicts(test_case),
                        self._a_generate_alignment_verdicts(),
                    )

                alignment_score = self._calculate_score(ScoreType.ALIGNMENT)
                complex_coverage_scores = [e.score / 5 for e in self.complex_coverage_verdicts]
                complex_coverage_score = (
                    sum(complex_coverage_scores) / len(complex_coverage_scores)
                    if complex_coverage_scores else 0.0
                )

                # Simple (5-question) coverage is intentionally skipped — it
                # never entered the score and was always None on disk. Recall
                # is carried entirely by complex coverage.
                self.score_breakdown = {
                    ScoreType.ALIGNMENT.value: alignment_score,
                    "complex_coverage": complex_coverage_score,
                }

                # F1 between alignment (precision) and complex coverage (recall).
                precision = alignment_score
                recall = complex_coverage_score
                self.score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                self.reason = await self._a_generate_reason()
                self.success = self.score >= self.threshold

                logs = {
                    'claims': self.claims,
                    'assessment_questions': self.assessment_questions,
                    'complex_assessment_questions': [e.dict() for e in self.complex_assessment_questions],
                    'alignment_verdicts': [v.dict() for v in self.alignment_verdicts],
                    'complex_coverage_verdicts': [v.dict() for v in self.complex_coverage_verdicts],
                    'alignment_score': alignment_score,
                    'complex_coverage_score': complex_coverage_score,
                    'score': self.score,
                    'reason': self.reason,
                    'success': self.success,
                }
                self.verbose_logs = json.dumps(logs)
                return self.score
            except Exception as exc:  # noqa: BLE001
                # An unrecoverable judge failure for THIS case (e.g.
                # trimAndLoadJson could not parse a truncated response, or
                # the judge JSON was missing a required key). Report it
                # through DeepEval's expected `error` channel so the case
                # STAYS in test_results with score=None instead of being
                # silently dropped from the result set entirely. This is
                # what keeps row counts aligned and makes the failure
                # visible in the logs rather than a phantom missing row.
                log.error("custom metric failed for a case: %s: %s",
                          type(exc).__name__, str(exc)[:200])
                self.error = f"{type(exc).__name__}: {str(exc)[:200]}"
                self.score = None
                self.success = False
                self.reason = None
                self.verbose_logs = None
                return None

    async def _a_generate_reason(self) -> str:
        if self.include_reason is False:
            return None

        contradictions = []
        redundancies = []
        for verdict in self.alignment_verdicts:
            if verdict.verdict.strip().lower() == "no":
                contradictions.append(verdict.reason)
            elif verdict.verdict.strip().lower() == "idk":
                redundancies.append(verdict.reason)

        questions = []
        if self.coverage_verdicts:
            for verdict in self.coverage_verdicts:
                if (
                    verdict.original_verdict.strip().lower() == "yes"
                    and verdict.summary_verdict.strip().lower() == "no"
                ):
                    questions.append(verdict.question)

        prompt: dict = SummarizationTemplate.generate_reason(
            contradictions=contradictions,
            redundancies=redundancies,
            questions=questions,
            score=format(self.score, ".2f"),
        )

        if len(questions) > 0:
            prompt += f"""Questions the original text can answer but not the summary:
{questions}

"""
        prompt += """JSON:
"""

        if self.using_native_model:
            res, cost = await self.model.a_generate(prompt)
            self.evaluation_cost += cost
            data = trimAndLoadJson(res, self)
            return data["reason"]
        else:
            try:
                res: Reason = await self.model.a_generate(prompt, schema=Reason)
                return res.reason
            except TypeError:
                res = await self.model.a_generate(prompt)
                data = trimAndLoadJson(res, self)
                return data["reason"]

    def _generate_reason(self) -> str:
        if self.include_reason is False:
            return None

        contradictions = []
        redundancies = []
        for verdict in self.alignment_verdicts:
            if verdict.verdict.strip().lower() == "no":
                contradictions.append(verdict.reason)
            elif verdict.verdict.strip().lower() == "idk":
                redundancies.append(verdict.reason)

        questions = []
        if self.coverage_verdicts:
            for verdict in self.coverage_verdicts:
                if (
                    verdict.original_verdict.strip().lower() == "yes"
                    and verdict.summary_verdict.strip().lower() == "no"
                ):
                    questions.append(verdict.question)

        prompt: dict = SummarizationTemplate.generate_reason(
            contradictions=contradictions,
            redundancies=redundancies,
            questions=questions,
            score=format(self.score, ".2f"),
        )

        if len(questions) > 0:
            prompt += f"""Questions the original text can answer but not the summary:
{questions}

"""
        prompt += """JSON:
"""

        if self.using_native_model:
            res, cost = self.model.generate(prompt)
            self.evaluation_cost += cost
            data = trimAndLoadJson(res, self)
            return data["reason"]
        else:
            try:
                res: Reason = self.model.generate(prompt, schema=Reason)
                return res.reason
            except TypeError:
                res = self.model.generate(prompt)
                data = trimAndLoadJson(res, self)
                return data["reason"]

    def _calculate_score(self, score_type: ScoreType) -> float:
        if score_type == ScoreType.ALIGNMENT:
            total = len(self.alignment_verdicts)
            if total == 0:
                return 0
            faithfulness_count = 0
            for verdict in self.alignment_verdicts:
                # Different from the faithfulness score, this
                # penalizes 'idk' (full of fluff) summaries
                if verdict.verdict.strip().lower() == "yes":
                    faithfulness_count += 1

            score = faithfulness_count / total

        else:
            if self.assessment_questions is None:
                return 1
            total = 0
            coverage_count = 0
            for verdict in self.coverage_verdicts:
                if verdict.original_verdict.strip().lower() == "yes":
                    total += 1
                    if verdict.summary_verdict.strip().lower() == "yes":
                        coverage_count += 1

            if total == 0:
                return 0

            score = coverage_count / total

        return 0 if self.strict_mode and score < self.threshold else score

    async def _a_generate_answers(self, text: str) -> List[str]:
        prompt = SummarizationTemplate.generate_answers(
            questions=self.assessment_questions, text=text
        )
        if self.using_native_model:
            res, cost = await self.model.a_generate(prompt)
            self.evaluation_cost += cost
            data = trimAndLoadJson(res, self)
            return data["answers"]
        else:
            try:
                res: Answers = await self.model.a_generate(
                    prompt, schema=Answers
                )
                return res.answers
            except TypeError:
                res = await self.model.a_generate(prompt)
                data = trimAndLoadJson(res, self)
                return data["answers"]

    def _generate_answers(self, text: str) -> List[str]:
        prompt = SummarizationTemplate.generate_answers(
            questions=self.assessment_questions, text=text
        )
        if self.using_native_model:
            res, cost = self.model.generate(prompt)
            self.evaluation_cost += cost
            data = trimAndLoadJson(res, self)
            return data["answers"]
        else:
            try:
                res: Answers = self.model.generate(prompt, schema=Answers)
                return res.answers
            except TypeError:
                res = self.model.generate(prompt)
                data = trimAndLoadJson(res, self)
                return data["answers"]
            
    async def _a_generate_complex_answers(self, text: str) -> List[str]:
        prompt = generate_complex_answers(self.complex_assessment_questions, text)
        if self.using_native_model:
            res, cost = await self.model.a_generate(prompt)
            self.evaluation_cost += cost
            data = trimAndLoadJson(res, self)
            return data["answers"]
        else:
            try:
                res: Answers = await self.model.a_generate(prompt, schema=Answers)
                return res.answers
            except TypeError:
                res = await self.model.a_generate(prompt)
                data = trimAndLoadJson(res, self)
                return data["answers"]
            
    def _generate_complex_answers(self, text: str) -> List[str]:
        prompt = generate_complex_answers(self.complex_assessment_questions, text)
        if self.using_native_model:
            res, cost = self.model.generate(prompt)
            self.evaluation_cost += cost
            data = trimAndLoadJson(res, self)
            return data["answers"]
        else:
            try:
                res: Answers = self.model.generate(prompt, schema=Answers)
                return res.answers
            except TypeError:
                res = self.model.generate(prompt)
                data = trimAndLoadJson(res, self)
                return data["answers"]

    async def _a_generate_assessment_questions(self, text: str):
        prompt = SummarizationTemplate.generate_questions(text=text, n=self.n)
        if self.using_native_model:
            res, cost = await self.model.a_generate(prompt)
            self.evaluation_cost += cost
            data = trimAndLoadJson(res, self)
            return data["questions"]
        else:
            try:
                res: Questions = await self.model.a_generate(
                    prompt, schema=Questions
                )
                return res.questions
            except TypeError:
                res = await self.model.a_generate(prompt)
                data = trimAndLoadJson(res, self)
                return data["questions"]

    def _generate_assessment_questions(self, text: str):
        prompt = SummarizationTemplate.generate_questions(text=text, n=self.n)
        if self.using_native_model:
            res, cost = self.model.generate(prompt)
            self.evaluation_cost += cost
            data = trimAndLoadJson(res, self)
            return data["questions"]
        else:
            try:
                res: Questions = self.model.generate(prompt, schema=Questions)
                return res.questions
            except TypeError:
                res = self.model.generate(prompt)
                data = trimAndLoadJson(res, self)
                return data["questions"]
            
    async def _a_generate_complex_assessment_questions(self, text: str) -> List[ComplexQuestion]:
        prompt = generate_complex_questions(text, self.n_complex_questions, fmt=self.fmt)
        if self.using_native_model:
            res, cost = await self.model.a_generate(prompt)
            self.evaluation_cost += cost
            data = trimAndLoadJson(res, self)
            return _safe_complex_questions(data.get("questions"))
        try:
            res = await self.model.a_generate(prompt, schema=ComplexQuestions)
            return res.questions
        except TypeError:
            res = await self.model.a_generate(prompt)
            data = trimAndLoadJson(res, self)
            return _safe_complex_questions(data.get("questions"))
            
    def _generate_complex_assessment_questions(self, text: str) -> List[ComplexQuestion]:
        prompt = generate_complex_questions(text, self.n_complex_questions, fmt=self.fmt)
        if self.using_native_model:
            res, cost = self.model.generate(prompt)
            self.evaluation_cost += cost
            data = trimAndLoadJson(res, self)
            return _safe_complex_questions(data.get("questions"))
        else:
            try:
                res = self.model.generate(prompt, schema=ComplexQuestions)
                return res.questions
            except TypeError:
                res = self.model.generate(prompt)
                data = trimAndLoadJson(res, self)
                return _safe_complex_questions(data.get("questions"))




    async def _a_generate_coverage_verdicts(
        self, test_case: LLMTestCase
    ) -> List[SummarizationCoverageVerdict]:
        if self.assessment_questions is None:
            self.assessment_questions = (
                await self._a_generate_assessment_questions(test_case.input)
            )

        tasks = [
            self._a_generate_answers(test_case.input),
            self._a_generate_answers(test_case.actual_output),
        ]
        results = await asyncio.gather(*tasks)
        original_answers, summary_answers = _reconcile_pair(results[0], results[1])

        coverage_veridcts: List[SummarizationCoverageVerdict] = []
        n = min(len(original_answers), len(summary_answers), len(self.assessment_questions))
        for i in range(n):
            coverage_veridcts.append(
                SummarizationCoverageVerdict(
                    summary_verdict=summary_answers[i],
                    original_verdict=original_answers[i],
                    question=self.assessment_questions[i],
                )
            )
        return coverage_veridcts

    def _generate_coverage_verdicts(
        self, test_case: LLMTestCase
    ) -> List[SummarizationCoverageVerdict]:
        if self.assessment_questions is None:
            self.assessment_questions = self._generate_assessment_questions(
                test_case.input
            )

        original_answers = self._generate_answers(test_case.input)
        summary_answers = self._generate_answers(test_case.actual_output)

        original_answers, summary_answers = _reconcile_pair(original_answers, summary_answers)

        coverage_veridcts: List[SummarizationCoverageVerdict] = []
        for i in range(len(original_answers)):
            coverage_veridcts.append(
                SummarizationCoverageVerdict(
                    summary_verdict=summary_answers[i],
                    original_verdict=original_answers[i],
                    question=self.assessment_questions[i],
                )
            )

        return coverage_veridcts
    
    def _generate_complex_coverage_verdicts(self, test_case: LLMTestCase) -> List[ComplexQuestionVerdict]:
        if self.complex_assessment_questions is None:
            self.complex_assessment_questions: List[ComplexQuestion] = self._generate_complex_assessment_questions(test_case.input)

        if not self.complex_assessment_questions:
            log.warn("no complex questions generated; complex coverage = 0")
            return []

        original_answers = [e.answer for e in self.complex_assessment_questions]
        summary_answers: List[str] = self._generate_complex_answers(test_case.actual_output)

        original_answers, summary_answers = _reconcile_pair(original_answers, summary_answers)
        questions = self.complex_assessment_questions[: len(original_answers)]
        answers = [{'original_answer': o, 'summary_answer': s}
                   for o, s in zip(original_answers, summary_answers)]

        prompt = generate_complex_verdicts(answers)
        if self.using_native_model:
            res, cost = self.model.generate(prompt)
            self.evaluation_cost += cost
            data = trimAndLoadJson(res, self)
            return _safe_complex_verdicts(data.get("verdicts"), answers, questions)
        else:
            try:
                res: ComplexQuestionsVerdictsOutputs = self.model.generate(prompt, schema=ComplexQuestionsVerdictsOutputs)
                return _safe_complex_verdicts(res.verdicts, answers, questions)
            except TypeError:
                res = self.model.generate(prompt)
                data = trimAndLoadJson(res, self)
                return _safe_complex_verdicts(data.get("verdicts"), answers, questions)
    
    async def _a_generate_complex_coverage_verdicts(self, test_case: LLMTestCase) -> List[ComplexQuestionVerdict]:
        if self.complex_assessment_questions is None:
            self.complex_assessment_questions = await self._a_generate_complex_assessment_questions(test_case.input)

        if not self.complex_assessment_questions:
            log.warn("no complex questions generated; complex coverage = 0")
            return []
        log.debug("generated %d complex questions", len(self.complex_assessment_questions))

        original_answers = [e.answer for e in self.complex_assessment_questions]
        summary_answers: List[str] = await self._a_generate_complex_answers(test_case.actual_output)

        original_answers, summary_answers = _reconcile_pair(original_answers, summary_answers)
        questions = self.complex_assessment_questions[: len(original_answers)]
        answers = [{'original_answer': o, 'summary_answer': s}
                   for o, s in zip(original_answers, summary_answers)]

        prompt = generate_complex_verdicts(answers)
        if self.using_native_model:
            res, cost = await self.model.a_generate(prompt)
            self.evaluation_cost += cost
            data = trimAndLoadJson(res, self)
            return _safe_complex_verdicts(data.get("verdicts"), answers, questions)
        try:
            res: ComplexQuestionsVerdictsOutputs = await self.model.a_generate(
                prompt, schema=ComplexQuestionsVerdictsOutputs)
            return _safe_complex_verdicts(res.verdicts, answers, questions)
        except TypeError:
            res = await self.model.a_generate(prompt)
            data = trimAndLoadJson(res, self)
            return _safe_complex_verdicts(data.get("verdicts"), answers, questions)


    async def _a_generate_alignment_verdicts(
        self,
    ) -> List[SummarizationAlignmentVerdict]:
        if len(self.claims) == 0:
            return []

        verdicts: List[SummarizationAlignmentVerdict] = []
        prompt = SummarizationTemplate.generate_alignment_verdicts(
            summary_claims=self.claims, original_text=self.truths
        )
        if self.using_native_model:
            res, cost = await self.model.a_generate(prompt)
            self.evaluation_cost += cost
            data = trimAndLoadJson(res, self)
            verdicts = [
                SummarizationAlignmentVerdict(**item)
                for item in data["verdicts"]
            ]
            return verdicts
        else:
            try:
                res: Verdicts = await self.model.a_generate(
                    prompt, schema=Verdicts
                )
                verdicts = [item for item in res.verdicts]
                return verdicts
            except TypeError:
                res = await self.model.a_generate(prompt)
                data = trimAndLoadJson(res, self)
                verdicts = [
                    SummarizationAlignmentVerdict(**item)
                    for item in data["verdicts"]
                ]
                return verdicts

    def _generate_alignment_verdicts(
        self,
    ) -> List[SummarizationAlignmentVerdict]:
        if len(self.claims) == 0:
            return []

        verdicts: List[SummarizationAlignmentVerdict] = []
        prompt = SummarizationTemplate.generate_alignment_verdicts(
            summary_claims=self.claims, original_text=self.truths
        )
        if self.using_native_model:
            res, cost = self.model.generate(prompt)
            self.evaluation_cost += cost
            data = trimAndLoadJson(res, self)
            verdicts = [
                SummarizationAlignmentVerdict(**item)
                for item in data["verdicts"]
            ]
            return verdicts
        else:
            try:
                res: Verdicts = self.model.generate(prompt, schema=Verdicts)
                verdicts = [item for item in res.verdicts]
                return verdicts
            except TypeError:
                res = self.model.generate(prompt)
                data = trimAndLoadJson(res, self)
                verdicts = [
                    SummarizationAlignmentVerdict(**item)
                    for item in data["verdicts"]
                ]
                return verdicts

    async def _a_generate_truths(self, text: str) -> str:
        return text
    def _generate_truths(self, text: str) -> str:
        return text
    
    async def _a_generate_claims(self, text: str) -> List[str]:
        # Borrow faithfulness template, but for filled schemas prepend a
        # schema-fair instruction so empty/placeholder fields and structural
        # tokens are NOT extracted as (false) clinical claims.
        prompt = FaithfulnessTemplate.generate_claims(text)
        if self.fmt == "json":
            prompt = (
                "The following text is a FILLED STRUCTURED CLINICAL SCHEMA "
                "rendered as key: value lines. Treat each POPULATED field as a "
                "clinical claim. IGNORE empty, 'N/A', 'none', 'not documented', "
                "or placeholder fields — absence of a value is NOT a claim. Do "
                "NOT invent narrative connective claims.\n\n" + prompt
            )
        if self.using_native_model:
            res, cost = await self.model.a_generate(prompt)
            self.evaluation_cost += cost
            data = trimAndLoadJson(res, self)
            return data["claims"]
        else:
            try:
                res: Claims = await self.model.a_generate(prompt, schema=Claims)
                return res.claims
            except TypeError:
                res = await self.model.a_generate(prompt)
                data = trimAndLoadJson(res, self)
                return data["claims"]


    def _generate_claims(self, text: str) -> List[str]:
        # Borrow faithfulness template
        prompt = FaithfulnessTemplate.generate_claims(text)
        if self.using_native_model:
            res, cost = self.model.generate(prompt)
            self.evaluation_cost += cost
            data = trimAndLoadJson(res, self)
            return data["claims"]
        else:
            try:
                res: Claims = self.model.generate(prompt, schema=Claims)
                return res.claims
            except TypeError:
                res = self.model.generate(prompt)
                data = trimAndLoadJson(res, self)
                return data["claims"]

    def is_successful(self) -> bool:
        if self.error is not None:
            self.success = False
        elif self.score is None:
            self.success = False
        else:
            try:
                self.success = self.score >= self.threshold
            except (TypeError, ValueError):
                self.success = False
        return self.success

    @property
    def __name__(self):
        return "Custom Summarization Metric"
    


##### PROMPT


def generate_complex_answers(questions, text):
        return f"""Based on the list of questions, generate a JSON with key 'answers', which answers the questions in order using information from the text.
        If the text does not contain enough information to answer the question, return 'idk'.
        The answer should be a concise 1-sentence long answer, which does not need to be in full sentences.

The length of 'answers' SHOULD BE STRICTLY EQUAL to that of questions.

Text:
{text}

Questions:
{questions}

JSON:
"""

def generate_complex_verdicts(answers):
    return f"""You are given a list of JSON objects. Each contains 'original_answer' and 'summary_answer'.
    Original answer is the correct answer to a question. 
    Your job is to assess if the summary answer is correct, based on the model answer which is the original answer.
    Give a score from 0 to 5, with 0 being completely wrong, and 5 being completely correct.
    If the 'summary_answer' is 'idk', return a score of 0.

    Return a JSON object with the key 'verdicts', which is a list of JSON objects, with the keys: 'score', and 'reason': a concise 1 sentence explanation for the score.

The length of the list SHOULD BE STRICTLY EQUAL to that of the answers list.

Answers:
{answers}

JSON:
"""

def generate_complex_questions(text, n, fmt: str = "text"):
        source_hint = ""
        if fmt == "json":
            source_hint = (
                "\nNote: the SUMMARY being evaluated elsewhere is a FILLED "
                "STRUCTURED CLINICAL SCHEMA (key/value fields), not prose. "
                "Generate questions about clinical CONTENT only (conditions, "
                "history, medications, findings, diagnosis, management). Do NOT "
                "ask about narrative style, prose flow, section wording, or "
                "presence/absence of headings — those are irrelevant to a "
                "structured record.\n"
            )
        return f"""Based on the given clinical text, generate a list of {n} questions that can be answered with the information in this document.
        The questions should target the main clinical points: patient background, reason for visit, relevant history, findings, diagnosis, and management/follow-up.
        Then, provide a concise 1 sentence answer to the question, using only information that can be found in the document.
        Answer concisely; the answer does not need to be in full sentences.
        Make sure the questions are different from each other.
        They should cover a combination of background, findings, diagnosis, and management.
{source_hint}
        Lastly, rate the importance of this question to the document on a scale of 1 to 5, with 1 being not important and 5 being most important.
        An important question relates to an essential clinical point such that not knowing the answer means the reader has missed something clinically significant.
        A less important question asks about a minor detail.

** IMPORTANT
Return a JSON object with the key 'questions', which is a list of {n} JSON objects, each with the keys 'question', 'answer', and 'importance'.
**
Text:
{text}

JSON:
"""