"""EvoRubrics training module."""
import logging
import re
import os
import numpy as np
from typing import List, Dict, Any, Tuple, Union, Optional
from bert_score import score as bert_score
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    from models import REFLECT_PROMPT_TEMPLATE
except ImportError:
    REFLECT_PROMPT_TEMPLATE = """You are a medical expert answering questions.

**ORIGINAL QUESTION:**
{question}

**INITIAL ANSWER:**
{baseline_answer}

**EVALUATION RUBRICS:**
{rubrics}

**YOUR TASK:**
Please provide an IMPROVED answer to the ORIGINAL QUESTION above.

**REQUIREMENTS:**
1. Review the initial answer against each rubric
2. Identify what is missing or weak
3. Write a COMPLETE, IMPROVED answer that addresses those issues
4. Your output must be a direct answer to the question, NOT rubrics or evaluation criteria
5. Do NOT output numbered evaluation points or scoring criteria

**IMPORTANT:** Output ONLY the improved answer as if you are directly answering the original question. Do NOT generate evaluation rubrics.

**IMPROVED ANSWER:**"""

logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(levelname)s:%(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logger.setLevel(logging.INFO)
_SENTENCE_TRANSFORMER_MODEL = None

def get_sentence_transformer_model(model_name: str = "sentence-transformers/all-MiniLM-L6-v2", device: str = "cpu"):
    """English docstring."""
    global _SENTENCE_TRANSFORMER_MODEL
    
    if _SENTENCE_TRANSFORMER_MODEL is None:
        try:
            from sentence_transformers import SentenceTransformer
            import torch
            
            logger.info("[log message omitted]")
            
            if device.startswith("cuda"):
                if not torch.cuda.is_available():
                    logger.warning("[log message omitted]")
                    device = "cpu"
                else:
                    device_idx = int(device.split(":")[1]) if ":" in device else 0
                    if device_idx >= torch.cuda.device_count():
                        logger.warning("[log message omitted]")
                        device = "cuda:0"
            
            _SENTENCE_TRANSFORMER_MODEL = SentenceTransformer(model_name)
            
            if device.startswith("cuda"):
                _SENTENCE_TRANSFORMER_MODEL = _SENTENCE_TRANSFORMER_MODEL.to(device)
            else:
                _SENTENCE_TRANSFORMER_MODEL = _SENTENCE_TRANSFORMER_MODEL.to("cpu")
            
            logger.info("[log message omitted]")
        except ImportError:
            logger.error("[log message omitted]")
            raise ImportError("")
        except Exception as e:
            logger.error("[log message omitted]")
            error_str = str(e).lower()
            if "meta tensor" in error_str or "no data" in error_str or "cuda" in error_str:
                if device != "cpu":
                    logger.warning("[log message omitted]")
                    try:
                        _SENTENCE_TRANSFORMER_MODEL = SentenceTransformer(model_name)
                        _SENTENCE_TRANSFORMER_MODEL = _SENTENCE_TRANSFORMER_MODEL.to("cpu")
                        logger.info("[log message omitted]")
                    except Exception as e2:
                        logger.error("[log message omitted]")
                        raise
                else:
                    raise
            else:
                logger.error("[log message omitted]")
            raise
    
    return _SENTENCE_TRANSFORMER_MODEL


class RubricItem:
    """English docstring."""
    def __init__(self, number: int, description: str, points: int, axis: str, explanation: str = ""):
        self.number = number
        self.description = description
        self.points = points
        self.axis = axis
        self.explanation = explanation
    
    def __repr__(self):
        return f"RubricItem({self.number}, points={self.points}, axis={self.axis})"


def normalize_score_to_minus1_1(total_score: float, positive_sum: float, negative_sum: float) -> float:
    """English docstring."""
    max_score = float(positive_sum)
    min_score = float(negative_sum)
    denom = max_score - min_score
    if denom <= 0.0:
        return 0.0
    normalized = (float(total_score) - min_score) / denom
    return float(np.clip(normalized, 0.0, 1.0))


def normalize_score_by_max_possible(total_score: float, positive_sum: float) -> float:
    """English docstring."""
    max_score = float(positive_sum)
    if max_score == 0.0:
        return 0.0
    return float(total_score) / max_score


def _rankdata_average(values: np.ndarray) -> np.ndarray:
    """Rank values with average rank for ties (1-based)."""
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    i = 0
    while i < len(values):
        j = i
        while j + 1 < len(values) and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0
        ranks[order[i : j + 1]] = avg_rank
        i = j + 1
    return ranks


def parse_rubrics(rubrics_text: str) -> List[RubricItem]:
    """English docstring."""
    if not rubrics_text or not rubrics_text.strip():
        logger.warning("[log message omitted]")
        return []
    
    rubric_items = []
    
    first_rubric_patterns = [
        r'(\d+)(?:\\\.|\.)\s*([^\n]+?)\[points?:(-?\d+)\]',
        r'(\d+)(?:\\\.|\.)\s*([^\n]+?)\s*,\s*points?:\s*(-?\d+)',
        r'(\d+)(?:\\\.|\.)\s*([^\n]+?)\s*\n\s*[Pp]oints?:\s*(-?\d+)',
        r'\[(\d+)\]\.\s*([^\n]+?)\s*\n\s*[Pp]oints?:\s*(-?\d+)',
    ]
    
    first_match = None
    for pattern in first_rubric_patterns:
        first_match = re.search(pattern, rubrics_text, re.MULTILINE)
        if first_match:
            break
    
    if first_match:
        start_pos = first_match.start()
        logger.debug("[log message omitted]")
        text_to_parse = rubrics_text[start_pos:]
    else:
        logger.debug("[log message omitted]")
        text_to_parse = rubrics_text
    
    
    
    new_format_patterns = [
        r'(\d+)(?:\\\.|\.)\s*([^\n]+?)\[points?:(-?\d+)\]',
        r'(\d+)(?:\\\.|\.)\s*([^\n]+?)\s*,\s*points?:\s*(-?\d+)',
    ]
    new_format_matches = []
    for pattern in new_format_patterns:
        matches = list(re.finditer(pattern, text_to_parse, re.MULTILINE | re.IGNORECASE))
        if matches:
            new_format_matches = matches
            break
    
    if new_format_matches:
        logger.debug("[log message omitted]")
        for match in new_format_matches:
            number = int(match.group(1))
            description = match.group(2).strip()
            points = int(match.group(3))
            axis = "unknown"
            explanation = ""
            
            description = description.rstrip('.')
            description = description.strip()
            if not description:
                continue
            
            rubric_item = RubricItem(number, description, points, axis, explanation)
            rubric_items.append(rubric_item)

    if not rubric_items:
        rubric_start_pattern = r'(?:^|\n)\s*(?:\[?(\d+)\]?(?:\\\.|\.))'
        rubric_starts = []
        for match in re.finditer(rubric_start_pattern, text_to_parse, re.MULTILINE):
            rubric_starts.append((match.start(), int(match.group(1))))

        if not rubric_starts:
            logger.debug("[log message omitted]")
            pattern = (
                r'(\d+)(?:\\\.|\.)\s*([^\n]+?)\s*\n\s*[Pp]oints?:\s*(-?\d+)'
                r'(?:\s*\n\s*axis:\s*([A-Za-z_]+))?'
                r'(?:\s*\n\s*explanation:\s*(.+?))?(?=\n\s*(?:\d+(?:\\\.|\.)|\[\d+\]\.)|\Z)'
            )
            matches = list(re.finditer(pattern, text_to_parse, re.MULTILINE))
            for match in matches:
                number = int(match.group(1))
                description = match.group(2).strip()
                points = int(match.group(3))
                axis = (match.group(4) or "unknown").strip()
                explanation = (match.group(5) or "").strip()
                rubric_item = RubricItem(number, description, points, axis, explanation)
                rubric_items.append(rubric_item)
        else:
            for i, (start_pos, number) in enumerate(rubric_starts):
                if i + 1 < len(rubric_starts):
                    end_pos = rubric_starts[i + 1][0]
                else:
                    end_pos = len(text_to_parse)

                item_text = text_to_parse[start_pos:end_pos]

                points_match = re.search(r'points?:\s*(-?\d+)', item_text, re.IGNORECASE)
                axis_match = re.search(r'axis:\s*([A-Za-z_]+)', item_text, re.IGNORECASE)
                explanation_match = re.search(r'explanation:\s*(.+)', item_text, re.IGNORECASE)

                if points_match:
                    points = int(points_match.group(1))
                    axis = (axis_match.group(1) if axis_match else "unknown").strip()
                    explanation = (explanation_match.group(1) if explanation_match else "").strip()

                    number_match = re.search(r'(?:\[?\d+\]?(?:\\\.|\.))\s*', item_text)
                    if number_match:
                        desc_start = number_match.end()
                    else:
                        desc_start = 0

                    desc_end = points_match.start()

                    description = item_text[desc_start:desc_end].strip()
                    description = description.rstrip(',').strip()

                    description = re.sub(r'\*\*([^*]+)\*\*', r'\1', description)  # **bold**
                    description = re.sub(r'\*([^*]+)\*', r'\1', description)  # *italic*
                    description = re.sub(r'^-\s+', '', description, flags=re.MULTILINE)

                    description = re.sub(r'\n\s*\n+', ' ', description)
                    description = re.sub(r'\n\s+', ' ', description)
                    description = re.sub(r'\s+', ' ', description)
                    description = description.strip()

                    rubric_item = RubricItem(number, description, points, axis, explanation)
                    rubric_items.append(rubric_item)
    
    if not rubric_items:
        logger.warning("[log message omitted]")
    else:
        logger.debug("[log message omitted]")
    
    return rubric_items


def score_answer_with_rubrics(
    answer: str, 
    rubrics: List[RubricItem],
    llm_evaluator=None,
    question: str = ""
) -> Tuple[float, Dict[str, Any]]:
    """English docstring."""
    if not rubrics:
        logger.warning("[log message omitted]")
        return 0.0, {}
    
    if llm_evaluator is None:
        logger.error("[log message omitted]")
        return 0.0, {}
    
    total_score = 0.0
    max_possible_score = 0.0
    details = {
        'positive_scores': [],
        'negative_scores': [],
        'by_axis': {}
    }
    
    logger.info("[log message omitted]")
    
    result_map = {}
    
    try:
        batch_results = llm_evaluator.evaluate_rubrics_batch(
            answer=answer,
            rubric_descriptions=[r.description for r in rubrics],
            question=question,
            rubric_points=[r.points for r in rubrics],
        )
        result_map = {item.get("index"): item for item in batch_results if isinstance(item, dict) and "index" in item}
        logger.info("[log message omitted]")
    except Exception as e:
        logger.warning("[log message omitted]")
        result_map = {}
    
    missing_indices = [idx for idx in range(len(rubrics)) if idx not in result_map]
    
    if missing_indices:
        logger.info("[log message omitted]")
        
        def evaluate_single_rubric(idx):
            """English docstring."""
            rubric = rubrics[idx]
            try:
                eval_result = llm_evaluator.evaluate_rubric_match(
                    answer=answer,
                    rubric_description=rubric.description,
                    question=question,
                    rubric_points=rubric.points
                )
                return idx, eval_result, None
            except Exception as e:
                logger.warning("[log message omitted]")
                eval_result = {
                    "criteria_met": False,
                    "explanation": "",
                }
                return idx, eval_result, e
        
        
        with ThreadPoolExecutor(max_workers=16) as executor:
            future_to_idx = {
                executor.submit(evaluate_single_rubric, idx): idx 
                for idx in missing_indices
            }
            
            for future in as_completed(future_to_idx):
                try:
                    idx, eval_result, error = future.result()
                    result_map[idx] = {
                        "criteria_met": eval_result.get("criteria_met", False),
                        "explanation": eval_result.get("explanation", ""),
                    }
                except Exception as e:
                    computed_idx = future_to_idx[future]
                    logger.error("[log message omitted]")
                    result_map[computed_idx] = {
                        "criteria_met": False,
                        "explanation": "",
                    }
        
        logger.info("[log message omitted]")
    
    for idx, rubric in enumerate(rubrics):
        eval_result = None
        if idx in result_map:
            eval_result = {
                "criteria_met": result_map[idx].get("criteria_met", False),
                "explanation": result_map[idx].get("explanation", ""),
            }
        else:
            logger.warning("[log message omitted]")
            eval_result = {
                "criteria_met": False,
                "explanation": "",
            }

        criteria_met = bool(eval_result.get('criteria_met', False))
        score = rubric.points if criteria_met else 0
        total_score += score
        
        entry = {
            'rubric': rubric.description,
            'points': rubric.points,
            'scored': score,
            'reasoning': eval_result.get('explanation', ''),
            'criteria_met': criteria_met
        }
        if rubric.points > 0:
            max_possible_score += rubric.points
            details['positive_scores'].append(entry)
        else:
            details['negative_scores'].append(entry)
        
        if rubric.axis not in details['by_axis']:
            details['by_axis'][rubric.axis] = []
        details['by_axis'][rubric.axis].append({
            'rubric': rubric.description,
            'points': rubric.points,
            'scored': score,
            'criteria_met': criteria_met
        })

    details['total_score'] = total_score
    details['max_possible_score'] = max_possible_score

    return total_score, details


def calculate_rubrics_generator_reward(
    generated_rubrics: str,
    ground_truth_rubrics: str,
    policy_answers: List[str],
    llm_evaluator=None,
    question: str = "",
    all_rubrics_list: List[str] = None,
    policy_llm=None,
    similarity_weight: float = 0.25,
    discrimination_weight: float = 0.25,
    diversity_weight: float = 0.25,
    reflect_weight: float = 0.0,
    return_details: bool = False,
    use_llm_similarity: bool = True,
    llm_similarity_api_key: str | None = None,
    llm_similarity_base_url: str | None = None,
    enable_similarity: bool = True,
    enable_discrimination: bool = True,
    enable_diversity: bool = True,
    enable_reflect: bool = False,
    skip_similarity_reason: str | None = None,
    skip_discrimination_reason: str | None = None,
    skip_diversity_reason: str | None = None,
    skip_reflect_reason: str | None = None,
    reflect_answers: Dict[int, str] = None,
    similarity_penalty_weight: float = 0.0,
    similarity_penalty_threshold: float = 0.8,
    intra_rubric_penalty_precomputed: float | None = None,
) -> float:
    """Compute a single Rubrics Generator reward composed of:

    1. similarity to ground-truth rubrics (LLM judge with BERT-score fallback);
    2. discrimination (variance of scoring multiple policy answers);
    3. diversity (semantic distance vs. other rubrics in the batch);
    4. reflect (gain from using rubrics to revise a baseline answer).

    Returns the scalar reward or, when ``return_details=True``, a dict with
    component breakdowns. The intra-rubric similarity penalty is subtracted
    from the weighted sum.
    """
    logger.info("Computing Rubrics Generator reward for one rubric group")

    similarity_reward = 0.0
    discrimination_reward = 0.0
    diversity_reward = 0.0
    reflect_reward = 0.0
    similarity_penalty_value = 0.0

    # 1. Similarity to ground-truth rubrics (optional).
    if enable_similarity:
        llm_similarity_score = None
        llm_similarity_reason = ""
        llm_call_succeeded = False
        if use_llm_similarity:
            llm_similarity_score, llm_similarity_reason = _similarity_via_llm(
                text_a=generated_rubrics,
                text_b=ground_truth_rubrics,
                api_key=llm_similarity_api_key,
                base_url=llm_similarity_base_url,
            )
            if llm_similarity_score is not None:
                llm_call_succeeded = True
                similarity_reward = llm_similarity_score
                logger.info(
                    ""
                    f"explanation={llm_similarity_reason}"
                )
        
        if not llm_call_succeeded:
            logger.info("[log message omitted]")
            try:
                import torch
                P, R, F1 = bert_score(
                    [generated_rubrics],
                    [ground_truth_rubrics],
                    lang='en',
                    model_type='xlnet-base-cased',
                    verbose=False,
                    device='cuda' if torch.cuda.is_available() else 'cpu'
                )
                similarity_reward = float(F1.item())
                logger.info(f"BERT Score F1: {similarity_reward:.4f}")
            except Exception as e:
                logger.warning("[log message omitted]")
                gen_words = set(generated_rubrics.lower().split())
                gt_words = set(ground_truth_rubrics.lower().split())
                if len(gt_words) == 0:
                    similarity_reward = 0.0
                else:
                    intersection = len(gen_words & gt_words)
                    union = len(gen_words | gt_words)
                    similarity_reward = intersection / union if union > 0 else 0.0
                logger.info("[log message omitted]")
    else:
        if skip_similarity_reason:
            logger.info("[log message omitted]")
        else:
            logger.info("[log message omitted]")
    
    if enable_discrimination:
        logger.info("[log message omitted]")
        discrimination_reward = calculate_discrimination_reward(
            generated_rubrics,
            policy_answers,
            llm_evaluator,
            question
        )
    else:
        if skip_discrimination_reason:
            logger.info("[log message omitted]")
        else:
            logger.info("[log message omitted]")
    
    if enable_diversity:
        if all_rubrics_list and len(all_rubrics_list) > 1:
            logger.info("[log message omitted]")
            diversity_reward = calculate_semantic_distance_reward(
                generated_rubrics,
                all_rubrics_list
            )
        else:
            logger.info("[log message omitted]")
    else:
        if skip_diversity_reason:
            logger.info("[log message omitted]")
        else:
            logger.info("[log message omitted]")
    
    # 4. Reflect reward (optional).
    if enable_reflect:
        logger.info("[log message omitted]")
        logger.info(f"[R_reflect] reflect_answers type: {type(reflect_answers)}")
        if reflect_answers:
            logger.info(f"[R_reflect] reflect_answers keys: {list(reflect_answers.keys())}")
            logger.info("[log message omitted]")
        else:
            logger.warning("[log message omitted]")
        
        logger.info(f"[R_reflect] policy_llm: {type(policy_llm)}")
        
        if reflect_answers is None and policy_llm is None:
            logger.warning("[log message omitted]")
        else:
            logger.info("[log message omitted]")
            reflect_reward = calculate_reflect_reward(
                generated_rubrics=generated_rubrics,
                baseline_answers=policy_answers,
                policy_llm=policy_llm,
                llm_evaluator=llm_evaluator,
                question=question,
                reflect_answers=reflect_answers
            )
    else:
        if skip_reflect_reason:
            logger.info("[log message omitted]")
        else:
            logger.info("[log message omitted]")

    # Intra-rubric similarity penalty (subtracted at the end).
    if similarity_penalty_weight > 0.0:
        if intra_rubric_penalty_precomputed is not None:
            similarity_penalty_value = float(intra_rubric_penalty_precomputed)
        else:
            similarity_penalty_value, _ = intra_rubric_similarity_penalty_single(
                generated_rubrics, similarity_penalty_threshold
            )

    total_reward = (
        similarity_weight * similarity_reward
        + discrimination_weight * discrimination_reward
        + diversity_weight * diversity_reward
        + reflect_weight * reflect_reward
        - similarity_penalty_weight * similarity_penalty_value
    )

    logger.info(
        f"Rubrics Generator reward: "
        f"similarity={similarity_reward:.4f}(w={similarity_weight}, on={enable_similarity}), "
        f"discrimination={discrimination_reward:.4f}(w={discrimination_weight}, on={enable_discrimination}), "
        f"diversity={diversity_reward:.4f}(w={diversity_weight}, on={enable_diversity}), "
        f"reflect={reflect_reward:.4f}(w={reflect_weight}, on={enable_reflect}), "
        f"intra_rubric_penalty={similarity_penalty_value:.4f}(w={similarity_penalty_weight}), "
        f"total={total_reward:.4f}"
    )

    if return_details:
        return {
            'total_reward': total_reward,
            'similarity_reward': similarity_reward,
            'discrimination_reward': discrimination_reward,
            'diversity_reward': diversity_reward,
            'reflect_reward': reflect_reward,
            'similarity_penalty': similarity_penalty_value,
        }
    return total_reward


def calculate_discrimination_reward(
    rubrics_text: str,
    policy_answers: List[str],
    llm_evaluator=None,
    question: str = "",
    return_baseline_scores: bool = False
) -> float | tuple[float, Dict[int, float]]:
    """English docstring."""
    if not policy_answers:
        logger.warning("[log message omitted]")
        if return_baseline_scores:
            return 0.0, {}
        return 0.0
    
    rubrics = parse_rubrics(rubrics_text)
    if not rubrics:
        logger.warning("[log message omitted]")
        if return_baseline_scores:
            return 0.0, {}
        return 0.0
    
    positive_sum = sum(r.points for r in rubrics if r.points > 0)
    negative_sum = sum(r.points for r in rubrics if r.points < 0)
    if positive_sum == 0 and negative_sum == 0:
        logger.warning("[log message omitted]")
        if return_baseline_scores:
            return 0.0, {}
        return 0.0
    
    normalized_scores = []
    raw_scores = []
    baseline_scores_dict = {}  # {baseline_idx: normalized_score}
    
    def score_single_answer(i, answer):
        """English docstring."""
        total_score, _ = score_answer_with_rubrics(
            answer,
            rubrics,
            llm_evaluator=llm_evaluator,
            question=question
        )
        normalized_score = normalize_score_to_minus1_1(total_score, positive_sum, negative_sum)
        return i, total_score, normalized_score
    
    with ThreadPoolExecutor(max_workers=min(len(policy_answers),16)) as executor:
        future_to_idx = {
            executor.submit(score_single_answer, i, answer): i 
            for i, answer in enumerate(policy_answers)
        }
        
        results = {}
        for future in as_completed(future_to_idx):
            try:
                i, total_score, normalized_score = future.result()
                results[i] = (total_score, normalized_score)
            except Exception as e:
                idx = future_to_idx[future]
                logger.error("[log message omitted]")
                results[idx] = (0.0, 0.0)
        
        for i in range(len(policy_answers)):
            total_score, normalized_score = results[i]
            raw_scores.append(total_score)
            baseline_scores_dict[i] = normalized_score
            normalized_scores.append(normalized_score)
            
            logger.debug(
                ""
                ""
            )
    
    if len(normalized_scores) < 2:
        logger.warning("[log message omitted]")
        if return_baseline_scores:
            return 0.0, baseline_scores_dict
        return 0.0
    
    std_dev = np.std(normalized_scores)
    
    logger.info(
        ""
        ""
        ""
        f"{[f'{s:.4f}' for s in normalized_scores]}\n"
        ""
        ""
    )
    
    if return_baseline_scores:
        return float(std_dev), baseline_scores_dict
    return float(std_dev)


def batch_calculate_discrimination_rewards(
    rubrics_list: List[str],
    policy_answers: List[str],
    llm_evaluator=None,
    question: str = "",
    return_baseline_scores_dict: bool = False
) -> Tuple[List[float], Dict[int, Dict[int, float]]] | List[float]:
    """English docstring."""
    if not policy_answers:
        logger.warning("[log message omitted]")
        if return_baseline_scores_dict:
            return [0.0] * len(rubrics_list), {}
        return [0.0] * len(rubrics_list)

    if not rubrics_list:
        logger.warning("[log message omitted]")
        if return_baseline_scores_dict:
            return [], {}
        return []

    logger.info("[log message omitted]")

    parsed_rubrics_list = []
    for i, rubrics_text in enumerate(rubrics_list):
        rubrics = parse_rubrics(rubrics_text)
        if not rubrics:
            logger.warning("[log message omitted]")
            continue
        positive_sum = sum(r.points for r in rubrics if r.points > 0)
        negative_sum = sum(r.points for r in rubrics if r.points < 0)
        if positive_sum == 0 and negative_sum == 0:
            logger.warning("[log message omitted]")
            parsed_rubrics_list.append({
                'index': i,
                'rubrics': rubrics,
                'positive_sum': positive_sum,
                'negative_sum': negative_sum,
                'is_valid': False
            })
        else:
            parsed_rubrics_list.append({
                'index': i,
                'rubrics': rubrics,
                'positive_sum': positive_sum,
                'negative_sum': negative_sum,
                'is_valid': True
            })

    if not parsed_rubrics_list:
        logger.warning("[log message omitted]")
        if return_baseline_scores_dict:
            return [0.0] * len(rubrics_list), {}
        return [0.0] * len(rubrics_list)

    logger.info("[log message omitted]")

    discrimination_rewards = [0.0] * len(rubrics_list)
    all_baseline_scores = {}  # {rubrics_idx: {answer_idx: normalized_score}}

    for rubrics_info in parsed_rubrics_list:
        rubrics_idx = rubrics_info['index']
        rubrics = rubrics_info['rubrics']
        positive_sum = rubrics_info['positive_sum']
        negative_sum = rubrics_info['negative_sum']
        is_valid = rubrics_info['is_valid']

        if not is_valid:
            discrimination_rewards[rubrics_idx] = 0.0
            all_baseline_scores[rubrics_idx] = {}
            continue

        try:
            rubric_descriptions = [r.description for r in rubrics]
            rubric_points = [r.points for r in rubrics]

            if hasattr(llm_evaluator, 'batch_evaluate_multiple_answers'):
                batch_results = llm_evaluator.batch_evaluate_multiple_answers(
                    answers=policy_answers,
                    rubric_descriptions=rubric_descriptions,
                    question=question,
                    rubric_points=rubric_points
                )

                normalized_scores = []
                baseline_scores_for_rubrics = {}

                for answer_idx, result in enumerate(batch_results):
                    total_score = result.get('total_score', 0.0)
                    normalized_score = normalize_score_to_minus1_1(
                        total_score, positive_sum, negative_sum
                    )

                    normalized_scores.append(normalized_score)
                    baseline_scores_for_rubrics[answer_idx] = normalized_score

                if len(normalized_scores) >= 2:
                    std_dev = float(np.std(normalized_scores))
                else:
                    logger.warning("[log message omitted]")
                    std_dev = 0.0

                discrimination_rewards[rubrics_idx] = std_dev
                all_baseline_scores[rubrics_idx] = baseline_scores_for_rubrics

                logger.info(
                    ""
                    f"std={std_dev:.4f}, "
                    f"scores={[f'{s:.4f}' for s in normalized_scores]}"
                )

            else:
                logger.warning("[log message omitted]")
                normalized_scores = []
                baseline_scores_for_rubrics = {}

                for answer_idx, answer in enumerate(policy_answers):
                    total_score, _ = score_answer_with_rubrics(
                        answer,
                        rubrics,
                        llm_evaluator=llm_evaluator,
                        question=question
                    )
                    normalized_score = normalize_score_to_minus1_1(
                        total_score, positive_sum, negative_sum
                    )
                    normalized_scores.append(normalized_score)
                    baseline_scores_for_rubrics[answer_idx] = normalized_score

                if len(normalized_scores) >= 2:
                    std_dev = float(np.std(normalized_scores))
                else:
                    std_dev = 0.0

                discrimination_rewards[rubrics_idx] = std_dev
                all_baseline_scores[rubrics_idx] = baseline_scores_for_rubrics

        except Exception as e:
            logger.error("[log message omitted]")
            discrimination_rewards[rubrics_idx] = 0.0
            all_baseline_scores[rubrics_idx] = {}

    logger.info("[log message omitted]")

    if return_baseline_scores_dict:
        return discrimination_rewards, all_baseline_scores
    return discrimination_rewards


def calculate_semantic_distance_reward(
    generated_rubrics: str,
    all_rubrics_list: List[str],
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
) -> float:
    """English docstring."""
    try:
        rubrics_items = parse_rubrics(generated_rubrics)
        use_sentences = False
        if len(rubrics_items) >= 2:
            texts = [r.description for r in rubrics_items]
        else:
            logger.warning("[log message omitted]")
            sentences = [s.strip() for s in re.split(r'[\n\.]+', generated_rubrics) if s.strip()]
            if len(sentences) < 2:
                logger.warning("[log message omitted]")
                return 0.0
            texts = sentences
            use_sentences = True

        model = get_sentence_transformer_model(embedding_model_name)
        embeddings = model.encode(texts, convert_to_numpy=True)
        
        n = len(embeddings)
        distances = []
        for i in range(n):
            for j in range(i + 1, n):
                a = embeddings[i]
                b = embeddings[j]
                cos_sim = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
                cos_dist = 1.0 - cos_sim
                distances.append(cos_dist)
        
        if not distances:
            return 0.0
        
        avg_distance = float(np.mean(distances))
        normalized_reward = avg_distance / 2.0
        logger.info(
            ""
            f"pair_count={len(distances)}, avg_dist={avg_distance:.4f}, "
            f"normalized={normalized_reward:.4f}"
        )
        return normalized_reward
    except Exception as e:
        logger.error("[log message omitted]")
        return 0.0


def batch_calculate_diversity_rewards(
    rubrics_list: List[str],
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
) -> List[float]:
    """English docstring."""
    if not rubrics_list:
        logger.warning("[log message omitted]")
        return []
    
    logger.info("[log message omitted]")
    
    try:
        model = get_sentence_transformer_model(embedding_model_name)
        
        all_texts = []
        text_group_info = []
        
        for rubrics_idx, generated_rubrics in enumerate(rubrics_list):
            rubrics_items = parse_rubrics(generated_rubrics)
            use_sentences = False
            
            if len(rubrics_items) >= 2:
                texts = [r.description for r in rubrics_items]
            else:
                logger.debug("[log message omitted]")
                sentences = [s.strip() for s in re.split(r'[\n\.]+', generated_rubrics) if s.strip()]
                if len(sentences) < 2:
                    logger.warning("[log message omitted]")
                    text_group_info.append({
                        'rubrics_idx': rubrics_idx,
                        'start_idx': len(all_texts),
                        'num_texts': 0,
                        'use_sentences': False
                    })
                    continue
                texts = sentences
                use_sentences = True
            
            start_idx = len(all_texts)
            all_texts.extend(texts)
            text_group_info.append({
                'rubrics_idx': rubrics_idx,
                'start_idx': start_idx,
                'num_texts': len(texts),
                'use_sentences': use_sentences
            })
        
        if not all_texts:
            logger.warning("[log message omitted]")
            return [0.0] * len(rubrics_list)
        
        logger.info("[log message omitted]")
        import time
        t_encode_start = time.time()
        all_embeddings = model.encode(all_texts, convert_to_numpy=True, batch_size=32)
        t_encode = time.time() - t_encode_start
        logger.info("[log message omitted]")
        
        diversity_rewards = [0.0] * len(rubrics_list)
        
        for group_info in text_group_info:
            rubrics_idx = group_info['rubrics_idx']
            start_idx = group_info['start_idx']
            num_texts = group_info['num_texts']
            use_sentences = group_info['use_sentences']
            
            if num_texts < 2:
                diversity_rewards[rubrics_idx] = 0.0
                continue
            
            end_idx = start_idx + num_texts
            embeddings = all_embeddings[start_idx:end_idx]
            
            n = len(embeddings)
            distances = []
            for i in range(n):
                for j in range(i + 1, n):
                    a = embeddings[i]
                    b = embeddings[j]
                    cos_sim = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
                    cos_dist = 1.0 - cos_sim
                    distances.append(cos_dist)
            
            if distances:
                avg_distance = float(np.mean(distances))
                normalized_reward = avg_distance / 2.0
                diversity_rewards[rubrics_idx] = normalized_reward
                logger.debug(
                    ""
                    f"source={'sentences' if use_sentences else 'rubrics'}, "
                    f"pair_count={len(distances)}, avg_dist={avg_distance:.4f}, "
                    f"normalized={normalized_reward:.4f}"
                )
            else:
                diversity_rewards[rubrics_idx] = 0.0
        
        logger.info("[log message omitted]")
        return diversity_rewards
        
    except Exception as e:
        logger.error("[log message omitted]")
        import traceback
        logger.error(traceback.format_exc())
        return [0.0] * len(rubrics_list)


def batch_calculate_intra_rubric_similarity_penalties(
    rubrics_list: List[str],
    threshold: float,
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
) -> List[Tuple[float, float]]:
    """English docstring."""
    if not rubrics_list:
        return []
    empty = [(0.0, 0.0)] * len(rubrics_list)
    logger.info("[log message omitted]")
    try:
        model = get_sentence_transformer_model(embedding_model_name)
        all_texts: List[str] = []
        text_group_info: List[Dict[str, Any]] = []

        for rubrics_idx, generated_rubrics in enumerate(rubrics_list):
            rubrics_items = parse_rubrics(generated_rubrics)
            use_sentences = False
            if len(rubrics_items) >= 2:
                texts = [r.description for r in rubrics_items]
            else:
                sentences = [s.strip() for s in re.split(r"[\n\.]+", generated_rubrics) if s.strip()]
                if len(sentences) < 2:
                    text_group_info.append(
                        {
                            "rubrics_idx": rubrics_idx,
                            "start_idx": len(all_texts),
                            "num_texts": 0,
                            "use_sentences": False,
                        }
                    )
                    continue
                texts = sentences
                use_sentences = True
            start_idx = len(all_texts)
            all_texts.extend(texts)
            text_group_info.append(
                {
                    "rubrics_idx": rubrics_idx,
                    "start_idx": start_idx,
                    "num_texts": len(texts),
                    "use_sentences": use_sentences,
                }
            )

        if not all_texts:
            return empty

        logger.info("[log message omitted]")
        all_embeddings = model.encode(all_texts, convert_to_numpy=True, batch_size=32)
        out: List[Tuple[float, float]] = [(0.0, 0.0)] * len(rubrics_list)

        for group_info in text_group_info:
            rubrics_idx = group_info["rubrics_idx"]
            start_idx = group_info["start_idx"]
            num_texts = group_info["num_texts"]
            if num_texts < 2:
                continue
            end_idx = start_idx + num_texts
            embeddings = all_embeddings[start_idx:end_idx]
            sims: List[float] = []
            nloc = len(embeddings)
            for i in range(nloc):
                for j in range(i + 1, nloc):
                    a = embeddings[i]
                    b = embeddings[j]
                    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
                    if denom <= 0:
                        continue
                    sims.append(float(np.dot(a, b) / denom))
            if not sims:
                continue
            avg_sim = float(np.mean(sims))
            penalty = max(0.0, avg_sim - float(threshold))
            out[rubrics_idx] = (penalty, avg_sim)
            logger.debug(
                ""
            )

        logger.info("[log message omitted]")
        return out
    except Exception as e:
        logger.error("[log message omitted]")
        import traceback

        logger.error(traceback.format_exc())
        return empty


def filter_similar_rubric_items_by_semantic_similarity(
    generated_rubrics: str,
    threshold: float = 0.75,
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
) -> Tuple[str, int, int]:
    """English docstring."""
    rubrics_items = parse_rubrics(generated_rubrics)
    before_count = len(rubrics_items)
    if before_count <= 1:
        return generated_rubrics, before_count, before_count

    try:
        model = get_sentence_transformer_model(embedding_model_name)
        descriptions = [r.description.strip() for r in rubrics_items]
        embeddings = model.encode(descriptions, convert_to_numpy=True, batch_size=32)

        keep = [True] * before_count
        for i in range(before_count):
            if not keep[i]:
                continue
            for j in range(i + 1, before_count):
                if not keep[j]:
                    continue
                a = embeddings[i]
                b = embeddings[j]
                denom = float(np.linalg.norm(a) * np.linalg.norm(b))
                if denom <= 0:
                    continue
                sim = float(np.dot(a, b) / denom)
                if sim > float(threshold):
                    len_i = len(descriptions[i])
                    len_j = len(descriptions[j])
                    if len_i >= len_j:
                        keep[j] = False
                    else:
                        keep[i] = False
                        break

        kept_items = [rubrics_items[idx] for idx, flag in enumerate(keep) if flag]
        if not kept_items:
            longest_idx = int(np.argmax([len(d) for d in descriptions]))
            kept_items = [rubrics_items[longest_idx]]

        after_count = len(kept_items)
        filtered_lines = []
        for idx, item in enumerate(kept_items, start=1):
            filtered_lines.append(f"{idx}. {item.description.strip()}")
            filtered_lines.append(f"points:{item.points}")
            filtered_lines.append(f"axis:{(item.axis or 'unknown').strip()}")
            if getattr(item, "explanation", ""):
                filtered_lines.append(f"explanation:{item.explanation.strip()}")
        filtered_rubrics_text = "\n".join(filtered_lines)
        return filtered_rubrics_text, before_count, after_count
    except Exception as e:
        logger.warning("[log message omitted]")
        return generated_rubrics, before_count, before_count


def intra_rubric_similarity_penalty_single(
    generated_rubrics: str,
    threshold: float,
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
) -> Tuple[float, float]:
    """English docstring."""
    r = batch_calculate_intra_rubric_similarity_penalties(
        [generated_rubrics], threshold, embedding_model_name=embedding_model_name
    )
    return r[0] if r else (0.0, 0.0)


def _similarity_via_llm(
    text_a: str,
    text_b: str,
    api_key: str | None = None,
    base_url: str | None = None,
    model: str = "deepseek-v3.2-20251201-160k-local",
) -> tuple[float | None, str]:
    """English docstring."""
    resolved_api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
    resolved_base_url = base_url or os.getenv("DEEPSEEK_BASE_URL") or "http://127.0.0.1:8000/v1"
    resolved_model = os.getenv("DEEPSEEK_MODEL") or model
    if not resolved_api_key:
        return None, ""
    try:
        from openai import OpenAI

        client = OpenAI(api_key=resolved_api_key, base_url=resolved_base_url, timeout=120.0)
        # Prompt: ask the model to rate rubric similarity (0-10) and provide one-sentence explanation.
        prompt = (
            "You are an evaluator tasked with determining the similarity between two evaluation rubrics. "
            "Please score the similarity between the two rubrics on a scale from 0 to 10. "
            "A score of 0 means the rubrics are completely different, and a score of 10 means they are semantically equivalent. "
            "Consider both the structure and the content when evaluating. "
            "Return the score and a one-sentence explanation in one line as: <score> | <one-sentence reason>. "
            "Keep the explanation concise.\n\n"
            f"Rubric A:\n{text_a}\n\nRubric B:\n{text_b}"
        )
        resp = client.chat.completions.create(
            model=resolved_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=20,
            extra_body={"enable_thinking": False}
        )
        content = (resp.choices[0].message.content or "").strip()
        matches = re.findall(r"[-+]?\d*\.?\d+", content)
        score = float(matches[0]) if matches else 0.0
        score = max(0.0, min(10.0, score))

        explanation = ""
        if matches:
            first = matches[0]
            idx = content.find(first)
            if idx >= 0:
                explanation = content[idx + len(first) :].lstrip(" |:-\n\t").strip()
        if not explanation and "|" in content:
            parts = content.split("|", 1)
            if len(parts) == 2:
                explanation = parts[1].strip()

        return score / 10.0, explanation
    except Exception as e:
        logger.warning("[log message omitted]")
        return None, ""


def batch_calculate_similarity_rewards(
    rubrics_list: List[str],
    ground_truth_rubrics: str,
    llm_evaluator=None,
    question: str = ""
) -> List[float]:
    """English docstring."""
    if not rubrics_list:
        logger.warning("[log message omitted]")
        return []

    if not ground_truth_rubrics:
        logger.warning("[log message omitted]")
        return [0.0] * len(rubrics_list)

    logger.info("[log message omitted]")

    try:
        if hasattr(llm_evaluator, 'batch_evaluate_similarity'):
            similarity_scores = llm_evaluator.batch_evaluate_similarity(
                rubrics_list=rubrics_list,
                ground_truth_rubrics=ground_truth_rubrics,
                question=question
            )

            logger.info("[log message omitted]")
            return similarity_scores

        else:
            logger.warning("[log message omitted]")
            similarity_scores = []
            for rubrics in rubrics_list:
                score, reason = _similarity_via_llm(
                    text_a=rubrics,
                    text_b=ground_truth_rubrics
                )
                if score is not None:
                    similarity_scores.append(score)
                else:
                    try:
                        import torch
                        P, R, F1 = bert_score(
                            [rubrics],
                            [ground_truth_rubrics],
                            lang='en',
                            model_type='xlnet-base-cased',
                            verbose=False,
                            device='cuda' if torch.cuda.is_available() else 'cpu'
                        )
                        similarity_scores.append(float(F1.item()))
                    except Exception as e:
                        logger.warning("[log message omitted]")
                        gen_words = set(rubrics.lower().split())
                        gt_words = set(ground_truth_rubrics.lower().split())
                        if len(gt_words) == 0:
                            similarity_scores.append(0.0)
                        else:
                            intersection = len(gen_words & gt_words)
                            union = len(gen_words | gt_words)
                            similarity_scores.append(intersection / union if union > 0 else 0.0)

            logger.info("[log message omitted]")
            return similarity_scores

    except Exception as e:
        logger.error("[log message omitted]")
        import traceback
        logger.error(traceback.format_exc())
        return [0.0] * len(rubrics_list)


def batch_calculate_reflect_rewards(
    rubrics_list: List[str],
    ground_truth_rubrics_list: List[str],
    baseline_answers_list: List[List[str]],
    reflect_answers_per_rubrics: List[Dict[int, str]],
    llm_evaluator,
    questions: List[str],
    max_concurrent: int = 64
) -> List[float]:
    """English docstring."""
    num_rubrics = len(rubrics_list)
    if num_rubrics == 0:
        return []
    
    logger.info("[log message omitted]")
    
    all_tasks = []
    
    for rubrics_idx in range(num_rubrics):
        scoring_rubrics_text = rubrics_list[rubrics_idx] if rubrics_idx < len(rubrics_list) else ""
        baseline_answers = baseline_answers_list[rubrics_idx] if rubrics_idx < len(baseline_answers_list) else []
        reflect_answers = reflect_answers_per_rubrics[rubrics_idx] if rubrics_idx < len(reflect_answers_per_rubrics) else {}
        question = questions[rubrics_idx] if rubrics_idx < len(questions) else ""
        
        if not scoring_rubrics_text or not baseline_answers:
            continue
        
        gt_rubrics = parse_rubrics(scoring_rubrics_text)
        if not gt_rubrics:
            continue
        
        positive_sum = sum(r.points for r in gt_rubrics if r.points > 0)
        negative_sum = sum(r.points for r in gt_rubrics if r.points < 0)
        if positive_sum == 0 and negative_sum == 0:
            continue
        
        for baseline_idx, baseline_answer in enumerate(baseline_answers):
            all_tasks.append({
                'task_type': 'baseline',
                'rubrics_idx': rubrics_idx,
                'baseline_idx': baseline_idx,
                'answer': baseline_answer,
                'gt_rubrics': gt_rubrics,
                'question': question,
                'positive_sum': positive_sum,
                'negative_sum': negative_sum
            })
            
            if isinstance(reflect_answers, dict) and baseline_idx in reflect_answers:
                revised_answer = reflect_answers[baseline_idx]
                all_tasks.append({
                    'task_type': 'revised',
                    'rubrics_idx': rubrics_idx,
                    'baseline_idx': baseline_idx,
                    'answer': revised_answer,
                    'gt_rubrics': gt_rubrics,
                    'question': question,
                    'positive_sum': positive_sum,
                    'negative_sum': negative_sum
                })
    
    baseline_scores = {}  # {(rubrics_idx, baseline_idx): score}
    revised_scores = {}   # {(rubrics_idx, baseline_idx): score}
    
    if all_tasks:
        num_baseline = sum(1 for t in all_tasks if t['task_type'] == 'baseline')
        num_revised = sum(1 for t in all_tasks if t['task_type'] == 'revised')
        logger.info("[log message omitted]")
        
        def evaluate_answer(task):
            """English docstring."""
            try:
                total_score, _ = score_answer_with_rubrics(
                    answer=task['answer'],
                    rubrics=task['gt_rubrics'],
                    llm_evaluator=llm_evaluator,
                    question=task['question']
                )
                normalized_score = normalize_score_to_minus1_1(
                    total_score, task['positive_sum'], task['negative_sum']
                )
                return task['task_type'], task['rubrics_idx'], task['baseline_idx'], normalized_score, None
            except Exception as e:
                return task['task_type'], task['rubrics_idx'], task['baseline_idx'], 0.0, e
        
        with ThreadPoolExecutor(max_workers=min(len(all_tasks), max_concurrent)) as executor:
            futures = {executor.submit(evaluate_answer, task): task for task in all_tasks}
            for future in as_completed(futures):
                try:
                    task_type, rubrics_idx, baseline_idx, score, error = future.result()
                    key = (rubrics_idx, baseline_idx)
                    
                    if task_type == 'baseline':
                        baseline_scores[key] = score
                    else:  # revised
                        revised_scores[key] = score
                    
                    if error:
                        logger.warning("[log message omitted]")
                except Exception as e:
                    logger.error("[log message omitted]")
    
    reflect_rewards = []
    for rubrics_idx in range(num_rubrics):
        improvements = []
        baseline_answers = baseline_answers_list[rubrics_idx] if rubrics_idx < len(baseline_answers_list) else []
        
        for baseline_idx in range(len(baseline_answers)):
            key = (rubrics_idx, baseline_idx)
            baseline_score = baseline_scores.get(key, 0.0)
            revised_score = revised_scores.get(key, 0.0)
            improvement = revised_score - baseline_score
            improvements.append(improvement)
        
        if improvements:
            reflect_reward = float(np.mean(improvements))
        else:
            reflect_reward = 0.0
        
        reflect_rewards.append(reflect_reward)
        logger.info("[log message omitted]")
    
    logger.info("[log message omitted]")
    return reflect_rewards


def calculate_reflect_reward(
    generated_rubrics: str,
    baseline_answers: List[str],
    policy_llm,
    llm_evaluator,
    question: str,
    reflect_answers: Dict[int, str] = None
) -> float:
    """English docstring."""
    if not baseline_answers:
        logger.warning("[log message omitted]")
        return 0.0
    
    if reflect_answers is None and policy_llm is None:
        logger.error("[log message omitted]")
        return 0.0
    
    logger.info(f"[R_reflect Debug] reflect_answers type: {type(reflect_answers)}")
    if reflect_answers:
        logger.info(f"[R_reflect Debug] reflect_answers keys: {list(reflect_answers.keys())}")
        logger.info("[log message omitted]")
    
    logger.info("[log message omitted]")
    
    rubrics = parse_rubrics(generated_rubrics)
    if not rubrics:
        logger.warning("[log message omitted]")
        return 0.0
    
    positive_sum = sum(r.points for r in rubrics if r.points > 0)
    negative_sum = sum(r.points for r in rubrics if r.points < 0)
    if positive_sum == 0 and negative_sum == 0:
        logger.warning("[log message omitted]")
        return 0.0
    
    score_improvements = []
    
    baseline_answers_to_score = []
    revised_answers_to_score = []
    indices_to_compute = []
    
    for i, baseline_answer in enumerate(baseline_answers):
        logger.info("[log message omitted]")
        
        if reflect_answers is not None and i in reflect_answers:
            revised_answer = reflect_answers[i]
            logger.info("[log message omitted]")
        elif policy_llm is not None:
            logger.info("[log message omitted]")
            reflect_prompt = REFLECT_PROMPT_TEMPLATE.format(
                question=question,
                baseline_answer=baseline_answer,
                rubrics=generated_rubrics
            )
            
            if hasattr(policy_llm, 'tokenizer') and policy_llm.tokenizer is not None:
                if hasattr(policy_llm.tokenizer, 'apply_chat_template') and policy_llm.tokenizer.chat_template is not None:
                    try:
                        from models import POLICY_LLM_SYSTEM_PROMPT
                    except ImportError:
                        POLICY_LLM_SYSTEM_PROMPT = "You are a helpful assistant.Your task is to answer the questions posed by the user."
                    
                    messages = [
                        {"role": "system", "content": POLICY_LLM_SYSTEM_PROMPT},
                        {"role": "user", "content": reflect_prompt}
                    ]
                    reflect_prompt = policy_llm.tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True,
                        enable_thinking=False
                    )
            
            try:
                revised_answers = policy_llm.generate_answers(
                    question=reflect_prompt,
                    num_answers=1,
                    max_length=512,
                    do_sample=True,  
                    sampling_profile="diverse"
                )
                revised_answer = revised_answers[0] if revised_answers else baseline_answer
            except Exception as e:
                logger.error("[log message omitted]")
                revised_answer = baseline_answer
        else:
            logger.warning("[log message omitted]")
            revised_answer = baseline_answer
        
        baseline_answers_to_score.append((i, baseline_answer))
        revised_answers_to_score.append((i, revised_answer))
        indices_to_compute.append(i)
    
    def score_baseline_for_reflect(idx, answer):
        """English docstring."""
        total_score, _ = score_answer_with_rubrics(answer, rubrics, llm_evaluator=llm_evaluator, question=question)
        normalized_score = normalize_score_to_minus1_1(total_score, positive_sum, negative_sum)
        return idx, normalized_score
    
    logger.info("[log message omitted]")
    computed_baseline_scores = {}
    with ThreadPoolExecutor(max_workers=min(len(baseline_answers_to_score), 16)) as executor:
        future_to_idx = {
            executor.submit(score_baseline_for_reflect, idx, answer): idx 
            for idx, answer in baseline_answers_to_score
        }
        
        for future in as_completed(future_to_idx):
            try:
                idx, normalized_score = future.result()
                computed_baseline_scores[idx] = normalized_score
                logger.info("[log message omitted]")
            except Exception as e:
                computed_idx = future_to_idx[future]
                logger.error("[log message omitted]")
                computed_baseline_scores[computed_idx] = 0.0
    
    def score_revised_answer(idx, answer):
        """English docstring."""
        total_score, _ = score_answer_with_rubrics(answer, rubrics, llm_evaluator=llm_evaluator, question=question)
        normalized_score = normalize_score_to_minus1_1(total_score, positive_sum, negative_sum)
        return idx, normalized_score
    
    logger.info("[log message omitted]")
    computed_revised_scores = {}
    with ThreadPoolExecutor(max_workers=min(len(revised_answers_to_score), 16)) as executor:
        future_to_idx = {
            executor.submit(score_revised_answer, idx, answer): idx 
            for idx, answer in revised_answers_to_score
        }
        
        for future in as_completed(future_to_idx):
            try:
                idx, normalized_score = future.result()
                computed_revised_scores[idx] = normalized_score
            except Exception as e:
                computed_idx = future_to_idx[future]
                logger.error("[log message omitted]")
                computed_revised_scores[computed_idx] = 0.0
    
    for i in indices_to_compute:
        s_0 = computed_baseline_scores.get(i, 0.0)
        
        if i in computed_revised_scores:
            s_1 = computed_revised_scores[i]
        else:
            logger.warning("[log message omitted]")
            s_1 = 0.0
        
        improvement = s_1 - s_0
        score_improvements.append(improvement)
        logger.info("[log message omitted]")
    
    r_reflect = np.mean(score_improvements)
    
    logger.info("[log message omitted]")
    
    return float(r_reflect)


def calculate_policy_llm_reward(
    answer: str,
    rubrics_list: List[str],
    llm_evaluator=None,
    question: str = ""
) -> Tuple[float, Dict[str, Any]]:
    """English docstring."""
    logger.info("[log message omitted]"
                 "")
    
    parsed_rubrics_list = []
    for i, rubrics_text in enumerate(rubrics_list):
        logger.info("[log message omitted]")
        rubrics = parse_rubrics(rubrics_text)
        if not rubrics:
            logger.warning("[log message omitted]")
            continue
        parsed_rubrics_list.append((i, rubrics))
    
    if not parsed_rubrics_list:
        logger.warning("[log message omitted]")
        return 0.0, {}
    
    logger.info("[log message omitted]")
    
    def score_single_rubrics(idx, rubrics):
        """English docstring."""
        try:
            total_score, score_details = score_answer_with_rubrics(
                answer,
                rubrics,
                llm_evaluator=llm_evaluator,
                question=question
            )
            positive_sum = sum(r.points for r in rubrics if r.points > 0)
            negative_sum = sum(r.points for r in rubrics if r.points < 0)
            reward = normalize_score_to_minus1_1(total_score, positive_sum, negative_sum)
            
            return idx, reward, total_score, positive_sum, negative_sum, score_details
        except Exception as e:
            logger.error("[log message omitted]")
            positive_sum = sum(r.points for r in rubrics if r.points > 0)
            negative_sum = sum(r.points for r in rubrics if r.points < 0)
            return idx, 0.0, 0.0, positive_sum, negative_sum, {}
    
    max_workers = min(len(parsed_rubrics_list), 16) 
    rewards = []
    details_list = []
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {
            executor.submit(score_single_rubrics, idx, rubrics): idx 
            for idx, rubrics in parsed_rubrics_list
        }
        
        for future in as_completed(future_to_idx):
            computed_idx = future_to_idx[future]
            try:
                idx, reward, total_score, positive_sum, negative_sum, score_details = future.result()
                rewards.append(reward)
                details_list.append({
                    'rubrics_index': computed_idx,
                    'reward': reward,
                    'total_score': total_score,
                    'max_possible_score': positive_sum,
                    'min_possible_score': negative_sum,
                    'score_details': score_details
                })
            except Exception as e:
                logger.exception("[log message omitted]")
                details_list.append({
                    'rubrics_index': computed_idx,
                    'reward': 0.0,
                    'total_score': 0.0,
                    'max_possible_score': 0.0,
                    'min_possible_score': 0.0,
                    'score_details': {}
                })
    
    if not rewards:
        logger.warning("[log message omitted]")
        return 0.0, {}
    
    avg_reward = np.mean(rewards)
    
    details = {
        'rewards': rewards,
        'avg_reward': avg_reward,
        'min_reward': np.min(rewards),
        'max_reward': np.max(rewards),
        'std_reward': np.std(rewards),
        'details_list': details_list
    }
    
    logger.debug("[log message omitted]"
                 ""
                 ""
                 "")

    return avg_reward, details


def batch_calculate_policy_llm_rewards(
    answers: List[str],
    rubrics_list: List[str],
    llm_evaluator=None,
    question: str = "",
    max_concurrent_api_calls: int = 64,
) -> Tuple[List[float], Dict[str, Any]]:
    """English docstring."""
    cap = max(1, min(int(max_concurrent_api_calls), 64))
    logger.info("[log message omitted]")

    if not answers:
        logger.warning("[log message omitted]")
        return [], {}

    if not rubrics_list:
        logger.warning("[log message omitted]")
        return [0.0] * len(answers), {}

    parsed_rubrics_list = []
    for i, rubrics_text in enumerate(rubrics_list):
        logger.info("[log message omitted]")
        rubrics = parse_rubrics(rubrics_text)
        if not rubrics:
            logger.warning("[log message omitted]")
            continue
        parsed_rubrics_list.append({
            'index': i,
            'rubrics': rubrics,
            'max_positive_score': sum(r.points for r in rubrics if r.points > 0),
            'min_negative_score': sum(r.points for r in rubrics if r.points < 0),
        })

    if not parsed_rubrics_list:
        logger.warning("[log message omitted]")
        return [0.0] * len(answers), {}

    logger.info("[log message omitted]")


    answers_total_scores = [0.0] * len(answers)
    answers_count = [0] * len(answers)

    all_details = []


    logger.info("[log message omitted]")

    def evaluate_single_rubrics_group(rubrics_info):
        """English docstring."""
        rubrics_idx = rubrics_info['index']
        rubrics = rubrics_info['rubrics']
        max_positive_score = rubrics_info['max_positive_score']
        min_negative_score = rubrics_info['min_negative_score']

        if max_positive_score == 0 and min_negative_score == 0:
            logger.warning("[log message omitted]")
            return None

        logger.info("[log message omitted]")

        try:
            rubric_descriptions = [r.description for r in rubrics]
            rubric_points = [r.points for r in rubrics]

            if hasattr(llm_evaluator, 'batch_evaluate_multiple_answers'):
                batch_results = llm_evaluator.batch_evaluate_multiple_answers(
                    answers=answers,
                    rubric_descriptions=rubric_descriptions,
                    question=question,
                    rubric_points=rubric_points
                )

                local_details = []
                for answer_idx, result in enumerate(batch_results):
                    total_score = result.get('total_score', 0.0)
                    normalized_score = result.get('normalized_score', 0.0)

                    reward = normalize_score_to_minus1_1(
                        total_score, max_positive_score, min_negative_score
                    )

                    local_details.append({
                        'rubrics_index': rubrics_idx,
                        'answer_index': answer_idx,
                        'reward': reward,
                        'total_score': total_score,
                        'max_score': max_positive_score,
                        'min_score': min_negative_score,
                        'batch_result': result
                    })

                logger.info("[log message omitted]")
                return (rubrics_idx, local_details)

            else:
                logger.warning("[log message omitted]")
                local_details = []
                for answer_idx, answer in enumerate(answers):
                    total_score, score_details = score_answer_with_rubrics(
                        answer,
                        rubrics,
                        llm_evaluator=llm_evaluator,
                        question=question
                    )
                    reward = normalize_score_to_minus1_1(
                        total_score, max_positive_score, min_negative_score
                    )

                    local_details.append({
                        'rubrics_index': rubrics_idx,
                        'answer_index': answer_idx,
                        'reward': reward,
                        'total_score': total_score,
                        'max_score': max_positive_score,
                        'min_score': min_negative_score,
                    })

                return (rubrics_idx, local_details)

        except Exception as e:
            logger.error("[log message omitted]")
            local_details = []
            for answer_idx in range(len(answers)):
                local_details.append({
                    'rubrics_index': rubrics_idx,
                    'answer_index': answer_idx,
                    'reward': 0.0,
                    'total_score': 0.0,
                    'max_score': max_positive_score,
                    'min_score': min_negative_score,
                    'error': str(e)
                })
            return (rubrics_idx, local_details)

    max_concurrent = min(len(parsed_rubrics_list), cap)
    logger.info("[log message omitted]")

    from concurrent.futures import ThreadPoolExecutor, as_completed

    with ThreadPoolExecutor(max_workers=max_concurrent) as executor:
        future_to_rubrics = {
            executor.submit(evaluate_single_rubrics_group, rubrics_info): rubrics_info
            for rubrics_info in parsed_rubrics_list
        }

        for future in as_completed(future_to_rubrics):
            try:
                rubrics_idx, details = future.result()
                if details is not None:
                    for detail in details:
                        answer_idx = detail['answer_index']
                        reward = detail['reward']

                        answers_total_scores[answer_idx] += reward
                        answers_count[answer_idx] += 1

                        all_details.append(detail)
            except Exception as e:
                logger.error("[log message omitted]")
                import traceback
                logger.error(traceback.format_exc())

    rewards = []
    for answer_idx in range(len(answers)):
        if answers_count[answer_idx] > 0:
            avg_reward = answers_total_scores[answer_idx] / answers_count[answer_idx]
        else:
            avg_reward = 0.0
        rewards.append(avg_reward)

    details = {
        'rewards': rewards,
        'num_rubrics_groups': len(parsed_rubrics_list),
        'all_details': all_details
    }

    logger.info("[log message omitted]")
    for idx, reward in enumerate(rewards):
        logger.info("[log message omitted]")

    return rewards, details


