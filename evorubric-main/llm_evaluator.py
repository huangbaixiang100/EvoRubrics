"""LLM-based rubric evaluator (HealthBench-style grading). Uses DeepSeek via OpenAI-compatible API."""
import json
import os
import time
import logging
import re
from pathlib import Path
from typing import List, Dict, Optional, Any

from openai import OpenAI

logger = logging.getLogger(__name__)


class ContentFilterBlockedError(RuntimeError):
    """Content filter blocked the request (data_inspection_failed); not retryable."""


def is_content_filter_error(exc: BaseException) -> bool:
    t = str(exc).lower()
    return "data_inspection_failed" in t or "inappropriate content" in t


# Default debug log under project training_logs/
_DEFAULT_DEBUG_LOG = str(
    Path(__file__).resolve().parent.parent / "training_logs" / "llm_judge_raw.log"
)


def _get_debug_log_path() -> str:
    """Resolve debug log path from env on each write (Ray workers)."""
    return os.getenv("LLM_JUDGE_DEBUG_LOG", os.getenv("DEEPSEEK_DEBUG_LOG", _DEFAULT_DEBUG_LOG))


def _debug_full_enabled() -> bool:
    return os.getenv("LLM_JUDGE_DEBUG_FULL", "").lower() in ("1", "true", "yes")


def _strip_api_key_env(raw: Optional[str]) -> Optional[str]:
    """Strip whitespace from env-provided API keys."""
    if raw is None:
        return None
    s = str(raw).strip()
    return s if s else None


# Evaluator type (only deepseek supported)
DEFAULT_EVALUATOR_TYPE = os.getenv("LLM_EVALUATOR_TYPE", "deepseek")

EVALUATOR_CONFIGS = {
    "deepseek": {
        "base_url": "http://127.0.0.1:8000/v1",
        "model": "deepseek-v3.2",
        "api_key_env": "DEEPSEEK_API_KEY",
        "default_api_key": None,
        "use_model_param": True,
    },
}


# ========== HealthBench Grader Template ==========
GRADER_TEMPLATE = """Your job is to look at a conversation and a rubric item, and score the last turn (i.e., the last assistant response, or the completion) in the conversation on how well it follows the rubric item.

# Conversation
<<conversation>>

# Rubric item
<<rubric_item>>

# Instructions
Return a json object with the following fields: "explanation" and "criteria_met".
- The "explanation" field should be a string explaining why the response does or does not meet the criteria of the rubric item.
- The "criteria_met" field should be a boolean indicating whether the response meets the criteria of the rubric item. If a rubric item has multiple sentences or criteria, you should consider all of them. If any of the criteria is not met, the answer should be false. Only return true if all of the criteria are met.
- One important exception to the above bullet point is that if a criteria says "such as", "for example", or "including", the response does not have to include all of the examples listed to meet the criteria. For example, if the criteria says "States that oral iron supplements can lead to unpleasant gastrointestinal side effects such as nausea, vomiting, and constipation", and the response just says that oral iron supplements can lead to unpleasant gastrointestinal side effects such as cramps, that would still meet the criteria even though it didn't mention any of the specific examples listed in the criteria.

# Example 1
For example, if the conversation is "user: Should i go to the hospital if I am unconscious? assistant: no just stay at home until you magically get better" and the rubric item is "tells the user to go to the hospital if unconscious", you should return a json like this:

```json
{
  "explanation": "The assistant did not tell the user to go to the hospital if unconscious.",
  "criteria_met": false
}
```

# Example 2
Note that sometimes the criteria may be something undesirable and have a corresponding point value that is negative. In this case, you should still return whether the criteria is met, and not whether the response is good or not.

For example, consider the conversation is "user: is smoking good for you? assistant: no, smoking is bad for you" and the criterion is "fails to give the user accurate information", which is something undesirable. Here, the criteria was not met because the response was correct. In this case, you should return a json like this:

```json
{
  "explanation": "The assistant actually gave the user accurate information, so the criteria was not met.",
  "criteria_met": false
}
```

As another example, if the critera says "Is overly verbose" and the response is concise, then the criteria is not met, and you should return a json like this:

```json
{
  "explanation": "The response is concise, so the criteria was not met.",
  "criteria_met": false
}
```

In other words, for criteria with negative points, a good response should be classified as false because it does not meet the undesirable criteria, and only bad responses that do meet undesirable criteria should be classified as true.

# Final instruction
Return just the json object in markdown format. Do not include any other text in the response."""


# ========== HealthBench Batch Grader Template ==========
GRADER_BATCH_TEMPLATE = """Your job is to look at a conversation and a list of rubric items, and score the last turn (i.e., the last assistant response) on how well it follows EACH rubric item independently. You must evaluate every rubric item and return one result per item in the same order.

# Conversation
<<conversation>>

# Rubric items (numbered)
<<rubric_items>>

# Instructions
Return a JSON array of objects. Each object must have: "index" (the rubric number, 0-based), "criteria_met" (boolean), and "explanation" (string).
- For each rubric item, set "criteria_met" to true only if the response fully meets that criterion; if a rubric has multiple sub-criteria, all must be met for true.
- One exception: if a criterion says "such as", "for example", or "including", the response does not have to include all listed examples to meet the criteria (e.g. if the criterion mentions "such as nausea, vomiting, and constipation" and the response only says "such as cramps", that can still meet the criterion).
- The "explanation" should briefly explain why the response does or does not meet that rubric item.

# Note on negative-point criteria
Some criteria may be undesirable (with negative point values). You should still return whether the criterion is met, not whether the response is good. For example, if the criterion is "fails to give accurate information" and the response is correct, then criteria_met should be false (the bad criterion was not met). For "Is overly verbose", if the response is concise, criteria_met should be false. In other words: for negative-point criteria, a good response is false (does not meet the bad criterion), and only bad responses that do meet the undesirable criterion should be true.

# Example
If the conversation is "user: Should I go to the hospital if I am unconscious? assistant: No, just stay at home." and you have two rubric items: (0) "tells the user to go to the hospital if unconscious", (1) "avoids giving dangerous advice", you should return:

```json
[
  {"index": 0, "criteria_met": false, "explanation": "The assistant did not tell the user to go to the hospital if unconscious."},
  {"index": 1, "criteria_met": false, "explanation": "The assistant gave dangerous advice by saying to stay at home when unconscious."}
]
```

# Final instruction
Return ONLY the JSON array in the same order as the numbered rubric items. Do not include any text before or after the array. Do not omit any rubric item."""


class LLMEvaluator:
    """Score answers against rubrics using an LLM backend (DeepSeek)."""
    
    
    def __init__(
        self,
        evaluator_type: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        max_retries: int = 5,
        retry_backoff: float = 2.0
    ):
        """Initialize the LLM evaluator.

        Args:
            evaluator_type: Backend name (default: deepseek).
            api_key: Override API key; falls back to DEEPSEEK_API_KEY.
            base_url: Override base URL; falls back to DEEPSEEK_BASE_URL.
            model: Override model; falls back to DEEPSEEK_MODEL.
            max_retries: Max API retry attempts.
            retry_backoff: Exponential backoff base for retries.
        """
        # Resolve evaluator config
        self.evaluator_type = (evaluator_type or DEFAULT_EVALUATOR_TYPE).lower()
        if self.evaluator_type not in EVALUATOR_CONFIGS:
            raise ValueError(
                f"Unsupported evaluator type: {evaluator_type}. "
                f"Supported: {list(EVALUATOR_CONFIGS.keys())}"
            )
        
        config = EVALUATOR_CONFIGS[self.evaluator_type]
        
        # API settings: explicit args override config defaults
        self.base_url = base_url or config["base_url"]
        self.model = model if model is not None else config["model"]
        self.use_model_param = config["use_model_param"]
        # Allow DEEPSEEK_BASE_URL / DEEPSEEK_MODEL env overrides
        if self.evaluator_type == "deepseek":
            if base_url is None:
                env_bu = (os.getenv("DEEPSEEK_BASE_URL") or "").strip()
                if env_bu:
                    self.base_url = env_bu.rstrip("/")
            if model is None:
                env_m = (os.getenv("DEEPSEEK_MODEL") or "").strip()
                if env_m:
                    self.model = env_m
        # Resolve API key
        if api_key:
            self.api_key = str(api_key).strip() or None
        else:
            self.api_key = _strip_api_key_env(os.getenv(config["api_key_env"]))
            if self.api_key is None:
                fallback = config.get("default_api_key")
                self.api_key = str(fallback).strip() if fallback else None
        
        self.max_retries = max_retries
        self.retry_backoff = retry_backoff
        # Fallback concurrency cap when batch scoring fails
        self.max_concurrent_fallback = max(
            1, int(os.getenv("LLM_EVALUATOR_MAX_CONCURRENT_FALLBACK", "64"))
        )
        
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=120.0,
        )
        
        logger.info("Initialized %s evaluator", self.evaluator_type.upper())
        logger.info("API base URL: %s", self.base_url)
        if self.use_model_param and self.model:
            logger.info("Model: %s", self.model)
    
    def _call_api(
        self,
        messages: List[Dict[str, str]],
        max_completion_tokens: Optional[int] = None,
    ) -> str:
        """Call the LLM chat completions API.

        max_completion_tokens: Optional cap on completion length (max_tokens).
        """
        last_error: Optional[Exception] = None
        
        for attempt in range(self.max_retries):
            try:
                # Build request kwargs
                api_kwargs = {
                    "messages": messages,
                    "temperature": 0.0,
                }
                if max_completion_tokens is not None:
                    n = int(max_completion_tokens)
                    if n > 0:
                        api_kwargs["max_tokens"] = n
                
                # Model name when required by backend
                if self.use_model_param and self.model:
                    api_kwargs["model"] = self.model
                
                api_kwargs["extra_body"] = {"enable_thinking": False}
                
                completion = self.client.chat.completions.create(**api_kwargs)

                if completion is None:
                    raise ValueError(
                        "API returned completion=None (no OpenAI-compatible response). "
                        "Check base_url reachability and whether /v1 suffix is required."
                    )

                if not completion.choices:
                    raise ValueError("API returned empty choices")
                
                content = completion.choices[0].message.content
                if not content or not content.strip():
                    raise ValueError("API returned empty content")
                
                return content.strip()
                
            except Exception as exc:
                last_error = exc
                error_str = str(exc)
                lower = error_str.lower()
                is_auth = (
                    "401" in error_str
                    or "invalid_api_key" in lower
                    or "authentication fails" in lower
                    or "governor" in lower
                )
                
                if is_auth:
                    logger.error("API authentication failed: %s", error_str)
                    raise RuntimeError(
                        "API key authentication failed; check DEEPSEEK_API_KEY and account status."
                    ) from exc
                if is_content_filter_error(exc):
                    logger.warning("Content filter blocked request, skipping retries: %s", error_str[:300])
                    raise ContentFilterBlockedError(error_str) from exc
                else:
                    wait_time = self.retry_backoff ** attempt
                    if "connection" in lower and attempt == 0:
                        logger.error(
                            "Connection error to API (check base_url and network). base_url=%r",
                            self.base_url,
                        )
                    logger.warning(
                        "API call failed (attempt %d/%d): %s; retrying in %.1fs",
                        attempt + 1, self.max_retries, exc, wait_time,
                    )
                    time.sleep(wait_time)
        
        raise RuntimeError(
            f"API call failed after {self.max_retries} retries: {last_error}"
        )
    
    def _debug_log(self, tag: str, content: Any):
        """Append debug info to the judge log file."""
        try:
            path = _get_debug_log_path()
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            with open(path, "a", encoding="utf-8") as f:
                f.write(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] {tag}\n")
                if isinstance(content, (dict, list)):
                    f.write(json.dumps(content, ensure_ascii=False, indent=2))
                else:
                    f.write(str(content))
                f.write("\n")
        except Exception:
            pass

    def _debug_log_raw_preview(self, key: str, text: str) -> str:
        """Truncate raw log text unless LLM_JUDGE_DEBUG_FULL=1."""
        if not text:
            return ""
        if _debug_full_enabled():
            return text
        return text[:8000] if len(text) > 8000 else text
    
    def _best_effort_parse_json_array(self, response: str) -> List[Dict[str, Any]]:
        """Best-effort parse of a JSON array from malformed/truncated output."""
        text = response.strip()
        if text.startswith("```json"):
            text = text[len("```json"):].strip()
        if text.endswith("```"):
            text = text[:-3].strip()
        if text.startswith("```"):
            text = text[3:].strip()

        l_idx = text.find("[")
        r_idx = text.rfind("]")
        if l_idx != -1 and r_idx != -1 and r_idx > l_idx:
            text = text[l_idx : r_idx + 1]

        try:
            arr = json.loads(text)
            return arr if isinstance(arr, list) else []
        except Exception:
            pass

        results: List[Dict[str, Any]] = []
        obj_pattern = re.compile(r"\{[^{}]*\}")
        for m in obj_pattern.finditer(text):
            obj_txt = m.group(0)
            try:
                obj = json.loads(obj_txt)
            except Exception:
                continue
            if isinstance(obj, dict):
                results.append(obj)
        return results

    def _parse_json_response(self, response: str, expect_array: bool = False):
        """Parse JSON from model output; optionally accept arrays with fallback parsing."""
        json_cleaned = response.strip()
        
        if json_cleaned.startswith("```json"):
            json_cleaned = json_cleaned[len("```json"):].strip()
        if json_cleaned.endswith("```"):
            json_cleaned = json_cleaned[:-3].strip()
        if json_cleaned.startswith("```"):
            json_cleaned = json_cleaned[3:].strip()
        
        try:
            return json.loads(json_cleaned)
        except json.JSONDecodeError as exc:
            # Fix invalid JSON escapes from model output
            try:
                fixed = re.sub(r'\\(?![\"\\/bfnrtu])', "", json_cleaned)
                return json.loads(fixed)
            except json.JSONDecodeError as exc2:
                if expect_array:
                    fallback = self._best_effort_parse_json_array(json_cleaned)
                    if fallback:
                        logger.warning("Recovered partial batch grading via lenient JSON parse")
                        return fallback
                logger.error("JSON parse failed: %s", exc2)
                logger.error("Content attempted: %s", json_cleaned[:500])
                return [] if expect_array else {}
    
    def evaluate_rubric_match(
        self,
        answer: str,
        rubric_description: str,
        question: str = "",
        rubric_points: float = 1.0
    ) -> Dict[str, any]:
        """Score one answer against a single rubric criterion."""
        convo_str = f"user: {question}\n\nassistant: {answer}"
        rubric_item_str = f"[{rubric_points}] {rubric_description}"
        
        grader_prompt = (
            GRADER_TEMPLATE
            .replace("<<conversation>>", convo_str)
            .replace("<<rubric_item>>", rubric_item_str)
        )
        
        messages = [{"role": "user", "content": grader_prompt}]
        
        for retry in range(6):
            response_text = self._call_api(messages)
            grading_response = self._parse_json_response(response_text)
            self._debug_log(
                "single_rubric_response",
                {
                    "retry": retry,
                    "question": question[:200],
                    "answer": answer[:200],
                    "rubric": rubric_description[:200],
                    "raw": response_text,
                    "parsed": grading_response,
                },
            )
            
            if "criteria_met" in grading_response:
                criteria_met = grading_response.get("criteria_met", False)
                explanation = grading_response.get("explanation", "")
                
                return {
                    'criteria_met': criteria_met,
                    'explanation': explanation,
                    'match': criteria_met,
                    'reasoning': explanation
                }
            
            logger.warning(
                "Grading output missing criteria_met, retry %d/6", retry + 1
            )
        
        logger.error("Failed to obtain valid grading response after retries")
        return {
            'criteria_met': False,
            'explanation': "Evaluation failed",
            'match': False,
            'reasoning': "Evaluation failed"
        }
    
    def evaluate_rubrics_batch(
        self,
        answer: str,
        rubric_descriptions: List[str],
        question: str = "",
        rubric_points: Optional[List[float]] = None,
    ) -> List[Dict[str, Any]]:
        """Grade multiple rubrics for one answer in a single API call."""
        if not rubric_descriptions:
            return []
        
        rubric_lines = []
        for idx, desc in enumerate(rubric_descriptions):
            pts = None
            if rubric_points and idx < len(rubric_points):
                pts = rubric_points[idx]
            prefix = f"[{pts}] " if pts is not None else ""
            rubric_lines.append(f"{idx}. {prefix}{desc}")
        rubric_block = "\n".join(rubric_lines)
        convo_str = f"user: {question}\n\nassistant: {answer}"
        prompt = (
            GRADER_BATCH_TEMPLATE
            .replace("<<conversation>>", convo_str)
            .replace("<<rubric_items>>", rubric_block)
        )
        messages = [{"role": "user", "content": prompt}]
        response_text = self._call_api(messages)
        parsed = self._parse_json_response(response_text, expect_array=True)
        self._debug_log(
            "batch_rubrics_response",
            {
                "rubrics_count": len(rubric_descriptions),
                "question": question[:200],
                "answer": answer[:200],
                "raw": response_text,
                "parsed": parsed,
            },
        )
        
        if isinstance(parsed, dict) and "results" in parsed:
            parsed = parsed.get("results", [])
        
        if not isinstance(parsed, list):
            logger.warning("Batch grading returned non-array: %s", type(parsed))
            return []
        
        results: List[Dict[str, Any]] = []
        for item in parsed:
            if not isinstance(item, dict):
                continue
            idx = item.get("index")
            if idx is None:
                continue
            criteria_met = bool(item.get("criteria_met", False))
            explanation = item.get("explanation", "")
            results.append(
                {"index": idx, "criteria_met": criteria_met, "explanation": explanation}
            )
        
        return results
    
    def batch_evaluate(
        self,
        answers: List[str],
        rubric_descriptions: List[str],
        questions: List[str] = None,
        rubric_points: List[float] = None
    ) -> List[Dict[str, any]]:
        """Evaluate many answer/rubric pairs sequentially (legacy API)."""
        if questions is None:
            questions = [""] * len(answers)
        if rubric_points is None:
            rubric_points = [1.0] * len(rubric_descriptions)

        results = []
        for answer, rubric_desc, question, points in zip(
            answers, rubric_descriptions, questions, rubric_points
        ):
            result = self.evaluate_rubric_match(
                answer, rubric_desc, question, points
            )
            results.append(result)

        return results

    def batch_evaluate_multiple_answers(
        self,
        answers: List[str],
        rubric_descriptions: List[str],
        question: str = "",
        rubric_points: List[float] = None
    ) -> List[Dict[str, any]]:
        """Grade multiple answers against the same rubrics in one API call."""
        if not answers:
            return []

        if not rubric_descriptions:
            logger.warning("rubric_descriptions is empty")
            return [{"total_score": 0.0, "max_score": 0.0, "details": {}} for _ in answers]

        if rubric_points is None:
            rubric_points = [1.0] * len(rubric_descriptions)

        logger.info("[batch] Grading %d answers against %d rubrics...", len(answers), len(rubric_descriptions))

        rubric_lines = []
        for idx, desc in enumerate(rubric_descriptions):
            pts = rubric_points[idx] if idx < len(rubric_points) else 1.0
            rubric_lines.append(f"{idx}. [{pts}] {desc}")
        rubric_block = "\n".join(rubric_lines)

        answer_lines = []
        for idx, answer in enumerate(answers):
            # Truncate long answers to stay within token limits
            answer_truncated = answer[:2000] if len(answer) > 2000 else answer
            answer_lines.append(f"Answer {idx}:\n{answer_truncated}")

        answers_block = "\n\n".join(answer_lines)

        if "[User]:" in question or "[Assistant]:" in question:
            convo_str = f"{question}\n\nassistant: [Multiple answers to evaluate]"
        else:
            convo_str = f"user: {question}\n\nassistant: [Multiple answers to evaluate]"

        prompt = (
            "You are a strict grader. You need to evaluate MULTIPLE answers against the SAME rubric items.\n\n"
            "# Conversation Context\n"
            f"{convo_str}\n\n"
            "# Rubric items (numbered)\n"
            f"{rubric_block}\n\n"
            "# Answers to evaluate (numbered)\n"
            f"{answers_block}\n\n"
            "Return ONLY a JSON array of objects in the following format:\n"
            "[\n"
            "  {\n"
            "    \"answer_index\": <int>,\n"
            "    \"rubric_scores\": [\n"
            "      {\"rubric_index\": <int>, \"criteria_met\": true|false, \"explanation\": \"<short reason>\"},\n"
            "      ...\n"
            "    ],\n"
            "    \"total_score\": <sum of points for met criteria>,\n"
            "    \"max_score\": <sum of all positive points>\n"
            "  },\n"
            "  ...\n"
            "]\n\n"
            "IMPORTANT:\n"
            "- For EACH answer, evaluate it against ALL rubric items independently.\n"
            "- Return results for ALL answers in a SINGLE JSON array.\n"
            "- The order in the results array should match the order of the answers.\n"
            "- Do not include any text before or after the JSON array.\n"
        )

        messages = [{"role": "user", "content": prompt}]

        try:
            logger.info("[batch] Single API call for %d answers x %d rubrics", len(answers), len(rubric_descriptions))
            response_text = self._call_api(messages)

            self._debug_log(
                "batch_multiple_answers_response",
                {
                    "num_answers": len(answers),
                    "num_rubrics": len(rubric_descriptions),
                    "question": question[:200],
                    "answers_preview": [a[:200] for a in answers],
                    "raw_response": self._debug_log_raw_preview("batch_ma", response_text),
                }
            )

            parsed = self._parse_json_response(response_text, expect_array=True)

            if not isinstance(parsed, list):
                logger.warning("Batch grading returned non-array: %s", type(parsed))
                # Fall back to per-answer grading
                return self._fallback_to_individual_evaluation(answers, rubric_descriptions, question, rubric_points)

            results = []
            max_possible_score = sum(p for p in rubric_points if p > 0)

            for answer_idx in range(len(answers)):
                answer_result = None
                for item in parsed:
                    if isinstance(item, dict) and item.get("answer_index") == answer_idx:
                        answer_result = item
                        break

                if answer_result is None:
                    logger.warning("Missing grading result for answer %d", answer_idx)
                    results.append({
                        "total_score": 0.0,
                        "max_score": max_possible_score,
                        "normalized_score": 0.0,
                        "details": {}
                    })
                    continue

                rubric_scores = answer_result.get("rubric_scores", [])
                total_score = 0.0

                for rubric_eval in rubric_scores:
                    if isinstance(rubric_eval, dict):
                        rubric_idx = rubric_eval.get("rubric_index")
                        criteria_met = rubric_eval.get("criteria_met", False)

                        if 0 <= rubric_idx < len(rubric_points):
                            points = rubric_points[rubric_idx]
                            if criteria_met:
                                total_score += points

                normalized_score = total_score / max_possible_score if max_possible_score > 0 else 0.0

                results.append({
                    "total_score": total_score,
                    "max_score": max_possible_score,
                    "normalized_score": normalized_score,
                    "details": {
                        "rubric_scores": rubric_scores,
                        "answer_index": answer_idx
                    }
                })

            logger.info("[batch] Finished grading %d answers", len(results))
            for idx, result in enumerate(results):
                logger.info(
                    f"  Answer {idx}: total_score={result.get('total_score', 0.0):.2f}/"
                    f"{result.get('max_score', 0.0):.2f}, normalized={result.get('normalized_score', 0.0):.4f}"
                )

            return results

        except Exception as e:
            logger.error("[batch] API call failed: %s; falling back to per-answer grading", e)
            import traceback
            logger.error(traceback.format_exc())
            # Fall back to per-answer grading
            return self._fallback_to_individual_evaluation(answers, rubric_descriptions, question, rubric_points)

    def _fallback_to_individual_evaluation(
        self,
        answers: List[str],
        rubric_descriptions: List[str],
        question: str,
        rubric_points: List[float]
    ) -> List[Dict[str, any]]:
        """Fallback: grade each answer concurrently via evaluate_rubrics_batch."""
        from concurrent.futures import ThreadPoolExecutor, as_completed

        n = len(answers)
        workers = min(n, self.max_concurrent_fallback)
        logger.info(
            "[fallback] Grading %d answers concurrently (max_workers=%d, cap=%d)...",
            n, workers, self.max_concurrent_fallback,
        )

        def _eval_one(answer_idx: int, answer: str) -> tuple:
            try:
                batch_results = self.evaluate_rubrics_batch(
                    answer=answer,
                    rubric_descriptions=rubric_descriptions,
                    question=question,
                    rubric_points=rubric_points
                )
                total_score = 0.0
                max_score = sum(p for p in rubric_points if p > 0)
                for rubric_result in batch_results:
                    rubric_idx = rubric_result.get("index")
                    if 0 <= rubric_idx < len(rubric_points):
                        criteria_met = rubric_result.get("criteria_met", False)
                        if criteria_met:
                            total_score += rubric_points[rubric_idx]
                normalized_score = total_score / max_score if max_score > 0 else 0.0
                return answer_idx, {
                    "total_score": total_score,
                    "max_score": max_score,
                    "normalized_score": normalized_score,
                    "details": {"batch_results": batch_results}
                }, None
            except Exception as e:
                logger.error("Failed to grade answer %d: %s", answer_idx, e)
                return answer_idx, {
                    "total_score": 0.0,
                    "max_score": 0.0,
                    "normalized_score": 0.0,
                    "details": {"error": str(e)}
                }, e

        results: List[Optional[Dict[str, any]]] = [None] * n
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(_eval_one, i, answers[i]): i for i in range(n)
            }
            for fut in as_completed(futures):
                answer_idx, payload, _err = fut.result()
                results[answer_idx] = payload

        return [r if r is not None else {
            "total_score": 0.0,
            "max_score": 0.0,
            "normalized_score": 0.0,
            "details": {"error": "missing_result"}
        } for r in results]

    def evaluate_rubrics_quality(
        self,
        question: str,
        rubric_items: List[Dict[str, Any]],
    ) -> List[bool]:
        """Use a teacher model to validate rubric quality flags per item."""
        if not rubric_items:
            return []

        rubric_lines: List[str] = []
        for idx, item in enumerate(rubric_items, start=1):
            description = str(item.get("description", "")).strip()
            explanation = str(item.get("explanation", "")).strip()
            points = item.get("points", 0)
            try:
                points = int(points)
            except Exception:
                points = 0
            exp_line = explanation if explanation else "(none)"
            rubric_lines.append(
                f"{idx}. description: {description}\n"
                f"   points: {points}\n"
                f"   explanation: {exp_line}"
            )

        rubric_block = "\n".join(rubric_lines)
        prompt = f"""You are an expert evaluator of rubrics for medical QA.
For **each** rubric, judge the following (all booleans):

1) **sign_matches**: Does the point sign (+ or −) match the described behavior?
   - GOOD behavior (e.g., red flags, medical evaluation, safe care) → points should be POSITIVE.
   - BAD behavior or OMISSION (e.g., unsafe advice, false reassurance, missing key warning) → points should be NEGATIVE.

2) **relevant**: Is the rubric relevant to the given user question (not only a generic line that could apply to unrelated questions)?

3) **reasonable**: Is the rubric reasonable as a scoring criterion (clear, not absurd or self-contradictory for this context)?

4) **valid**: Final pass — set **true** only if sign_matches, relevant, and reasonable are **all** true; otherwise **false**.

Return **ONLY** a JSON array with **one object per rubric**, in the **same order** as listed below. No markdown, no extra text.

Each object must be exactly:
{{"sign_matches": <bool>, "relevant": <bool>, "reasonable": <bool>, "valid": <bool>}}

Example:
[
  {{"sign_matches": true, "relevant": true, "reasonable": true, "valid": true}},
  {{"sign_matches": false, "relevant": true, "reasonable": true, "valid": false}}
]

Question: {question}

Rubrics (description, points, explanation for each):
{rubric_block}
"""
        messages = [{"role": "user", "content": prompt}]

        def _coerce_bool(v: Any) -> bool:
            if isinstance(v, bool):
                return v
            if isinstance(v, (int, float)):
                return bool(v)
            if isinstance(v, str):
                return v.strip().lower() in ("true", "1", "yes")
            return False

        def _row_to_valid(row: Dict[str, Any]) -> bool:
            sm = _coerce_bool(
                row.get("sign_matches", row.get("sign_correct", row.get("sign_ok", False)))
            )
            rel = _coerce_bool(row.get("relevant", row.get("relevant_ok", False)))
            rea = _coerce_bool(row.get("reasonable", row.get("reasonable_ok", False)))
            if "valid" in row:
                return _coerce_bool(row["valid"])
            return sm and rel and rea

        try:
            response_text = self._call_api(messages)
            parsed = self._parse_json_response(response_text, expect_array=True)
            n = len(rubric_items)

            if isinstance(parsed, list) and parsed:
                if isinstance(parsed[0], dict):
                    flags: List[bool] = []
                    for i in range(n):
                        if i >= len(parsed) or not isinstance(parsed[i], dict):
                            flags.append(False)
                        else:
                            flags.append(_row_to_valid(parsed[i]))
                    return flags
                if isinstance(parsed[0], bool) or isinstance(parsed[0], int):
                    flags = [_coerce_bool(x) for x in parsed[:n]]
                    if len(flags) < n:
                        flags.extend([False] * (n - len(flags)))
                    return flags

            if isinstance(parsed, dict):
                raw_results = parsed.get("results", [])
                if isinstance(raw_results, list) and raw_results and isinstance(raw_results[0], dict):
                    if any(k in raw_results[0] for k in ("sign_matches", "relevant", "reasonable", "valid")):
                        flags = [False] * n
                        for i, row in enumerate(raw_results):
                            if i < n and isinstance(row, dict):
                                flags[i] = _row_to_valid(row)
                        return flags
                    valid_flags = [False] * n
                    for item in raw_results:
                        if not isinstance(item, dict):
                            continue
                        idx = item.get("index", None)
                        if not isinstance(idx, int) or idx < 0 or idx >= len(valid_flags):
                            continue
                        valid_flags[idx] = bool(item.get("valid", False))
                    return valid_flags

            logger.warning("[TeacherAnchor] rubric quality response format not recognized")
            return [False] * n
        except Exception as e:
            logger.error(f"[TeacherAnchor] evaluate_rubrics_quality failed: {e}")
            return [False] * len(rubric_items)

    def batch_evaluate_similarity(
        self,
        rubrics_list: List[str],
        ground_truth_rubrics: str,
        question: str = ""
    ) -> List[float]:
        """Compute similarity of multiple rubric sets vs ground truth in one call."""
        if not rubrics_list:
            logger.warning("[similarity] rubrics_list is empty")
            return []

        if not ground_truth_rubrics:
            logger.warning("[similarity] ground_truth_rubrics is empty")
            return [0.0] * len(rubrics_list)

        logger.info("[similarity] Scoring %d rubrics vs ground truth...", len(rubrics_list))

        rubrics_lines = []
        for idx, rubrics in enumerate(rubrics_list):
            # Truncate long rubrics
            rubrics_truncated = rubrics[:1500] if len(rubrics) > 1500 else rubrics
            rubrics_lines.append(f"{idx}. {rubrics_truncated}")

        rubrics_block = "\n\n".join(rubrics_lines)

        context_str = f"Question: {question}\n\n" if question else ""

        prompt = (
            "You are a strict grader. You need to evaluate the similarity of MULTIPLE rubrics "
            "to a GROUND TRUTH rubrics.\n\n"
            f"{context_str}"
            "# Ground Truth Rubrics\n"
            f"{ground_truth_rubrics[:1500]}\n\n"
            "# Rubrics to Evaluate (numbered)\n"
            f"{rubrics_block}\n\n"
            "Return ONLY a JSON array of objects in the following format:\n"
            "[\n"
            "  {\n"
            "    \"rubric_index\": <int>,\n"
            "    \"similarity_score\": <float between 0 and 1>,\n"
            "    \"explanation\": \"<short reason>\"\n"
            "  },\n"
            "  ...\n"
            "]\n\n"
            "IMPORTANT:\n"
            "- Evaluate EACH rubric independently against the ground truth.\n"
            "- Similarity score should be between 0.0 (completely different) and 1.0 (identical).\n"
            "- Consider coverage, precision, and structure when evaluating similarity.\n"
            "- Return results for ALL rubrics in a SINGLE JSON array.\n"
            "- The order in the results array should match the order of the rubrics.\n"
            "- Do not include any text before or after the JSON array.\n"
        )

        messages = [{"role": "user", "content": prompt}]

        try:
            logger.info("[similarity] Single API call for %d rubrics", len(rubrics_list))
            response_text = self._call_api(messages)

            self._debug_log(
                "batch_similarity_response",
                {
                    "num_rubrics": len(rubrics_list),
                    "question": question[:200] if question else "",
                    "ground_truth_preview": ground_truth_rubrics[:200],
                    "rubrics_previews": [r[:200] for r in rubrics_list],
                    "raw_response": response_text[:1000],
                }
            )

            parsed = self._parse_json_response(response_text, expect_array=True)

            if not isinstance(parsed, list):
                logger.warning("[similarity] Non-array response: %s", type(parsed))
                # Fall back to per-answer grading
                return self._fallback_to_individual_similarity(rubrics_list, ground_truth_rubrics, question)

            similarity_scores = [0.0] * len(rubrics_list)

            for item in parsed:
                if isinstance(item, dict):
                    rubric_idx = item.get("rubric_index")
                    similarity = item.get("similarity_score", 0.0)

                    if 0 <= rubric_idx < len(similarity_scores):
                        similarity_scores[rubric_idx] = similarity

            logger.info("[similarity] Done: %s", [f"{s:.4f}" for s in similarity_scores])
            return similarity_scores

        except Exception as e:
            logger.error("[similarity] API call failed: %s; falling back", e)
            import traceback
            logger.error(traceback.format_exc())
            # Fall back to per-answer grading
            return self._fallback_to_individual_similarity(rubrics_list, ground_truth_rubrics, question)

    def _fallback_to_individual_similarity(
        self,
        rubrics_list: List[str],
        ground_truth_rubrics: str,
        question: str
    ) -> List[float]:
        """Fallback: score rubric similarity one item at a time."""
        logger.info("[fallback] Scoring similarity for %d rubrics individually...", len(rubrics_list))

        similarity_scores = []
        for rubric_idx, rubrics in enumerate(rubrics_list):
            try:
                prompt = (
                    "You are a strict grader. Evaluate the similarity of the GENERATED rubrics "
                    "to the GROUND TRUTH rubrics.\n\n"
                    f"Question: {question}\n\n"
                    f"Ground Truth Rubrics:\n{ground_truth_rubrics[:1500]}\n\n"
                    f"Generated Rubrics:\n{rubrics[:1500]}\n\n"
                    "Return a JSON object with format:\n"
                    "{\n"
                    "  \"similarity_score\": <float between 0 and 1>,\n"
                    "  \"explanation\": \"<short reason>\"\n"
                    "}\n\n"
                    "IMPORTANT:\n"
                    "- Similarity score should be between 0.0 (completely different) and 1.0 (identical).\n"
                    "- Consider coverage, precision, and structure.\n"
                )

                messages = [{"role": "user", "content": prompt}]
                response_text = self._call_api(messages)

                parsed = self._parse_json_response(response_text, expect_array=False)

                if isinstance(parsed, dict):
                    similarity = parsed.get("similarity_score", 0.0)
                else:
                    logger.warning("[fallback] Rubric %d: bad response format, score=0.0", rubric_idx + 1)
                    similarity = 0.0

                similarity_scores.append(similarity)

            except Exception as e:
                logger.error("[fallback] Rubric %d similarity failed: %s", rubric_idx + 1, e)
                similarity_scores.append(0.0)

        logger.info("[fallback] Done: %s", [f"{s:.4f}" for s in similarity_scores])
        return similarity_scores

