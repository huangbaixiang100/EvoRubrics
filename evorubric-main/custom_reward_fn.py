"""Adversarial reward functions used by the EvoRubric trainer.

This module exposes:

* ``AdversarialRewardState`` / ``GLOBAL_REWARD_STATE`` — shared mutable state
  that holds the rubrics / baseline answers / reward weights used across calls.
* ``set_reward_mode(...)`` — update the global state from the trainer.
* ``adversarial_reward_fn_batch`` — entry point loaded by verl's
  ``BatchRewardManager`` (configured in the training scripts).
* ``adversarial_reward_fn`` / ``adversarial_reward_fn_dataproto`` /
  ``compute_adversarial_reward`` — single-sample and DataProto-style helpers.

The reward consists of similarity / discrimination / diversity / reflect
components plus an optional intra-rubric similarity penalty.
"""
from __future__ import annotations

import logging
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Dict, List

import torch

logger = logging.getLogger(__name__)

try:
    from verl import DataProto  # noqa: F401
    VERL_AVAILABLE = True
except ImportError:
    print("[Warning] VERL not available, using dict interface")
    VERL_AVAILABLE = False
    DataProto = Dict  

from reward_calculator import (
    calculate_policy_llm_reward,
    calculate_rubrics_generator_reward,
    parse_rubrics,
)
from llm_evaluator import LLMEvaluator


@dataclass
class AdversarialRewardState:
    """Per-process state holding inputs and weights for reward computation."""

    mode: str = "policy_llm" 
    rubrics_list: List[str] = field(default_factory=list) 
    ground_truth_rubrics: str = ""  
    policy_answers: List[str] = field(default_factory=list) 
    all_rubrics_list: List[str] = field(default_factory=list)
    question: str = ""  
    sub_rewards_history: List[Dict[str, float]] = field(default_factory=list)  

    # Rubrics Generator reward weights.
    similarity_weight: float = 0.25
    discrimination_weight: float = 0.25
    diversity_weight: float = 0.25
    reflect_weight: float = 0.0
    
    # Component on/off toggles.
    enable_similarity: bool = True
    enable_discrimination: bool = True
    enable_diversity: bool = True
    enable_reflect: bool = False

    policy_llm: Any = None
    # {baseline_idx: {rubrics_idx: revised_answer}}; consumed by R_reflect.
    reflect_answers_per_baseline: Dict[int, Dict[int, str]] = field(default_factory=dict)

    # Intra-rubric similarity penalty (subtracted from the weighted sum).
    similarity_penalty_weight: float = 0.0
    similarity_penalty_threshold: float = 0.8

    max_concurrent_api_calls: int = 64


GLOBAL_REWARD_STATE = AdversarialRewardState(mode="policy_llm")
LLM_EVALUATOR: LLMEvaluator | None = None


def initialize_llm_evaluator() -> LLMEvaluator:
    global LLM_EVALUATOR
    if LLM_EVALUATOR is None:
        LLM_EVALUATOR = LLMEvaluator()
    return LLM_EVALUATOR


def set_reward_mode(
    mode: str,
    rubrics_list: List[str] | None = None,
    ground_truth_rubrics: str | None = None,
    policy_answers: List[str] | None = None,
    all_rubrics_list: List[str] | None = None,
    question: str | None = None,
    policy_llm: Any = None,
    similarity_weight: float | None = None,
    discrimination_weight: float | None = None,
    diversity_weight: float | None = None,
    reflect_weight: float | None = None,
    enable_similarity: bool | None = None,
    enable_discrimination: bool | None = None,
    enable_diversity: bool | None = None,
    enable_reflect: bool | None = None,
    reflect_answers_per_baseline: Dict[int, Dict[int, str]] | None = None,
    similarity_penalty_weight: float | None = None,
    similarity_penalty_threshold: float | None = None,
    max_concurrent_api_calls: int | None = None,
) -> None:
    """Update the global reward state used by reward function entry points."""
    global GLOBAL_REWARD_STATE
    GLOBAL_REWARD_STATE.mode = mode
    if rubrics_list is not None:
        GLOBAL_REWARD_STATE.rubrics_list = rubrics_list
    if ground_truth_rubrics is not None:
        GLOBAL_REWARD_STATE.ground_truth_rubrics = ground_truth_rubrics
    if policy_answers is not None:
        GLOBAL_REWARD_STATE.policy_answers = policy_answers
    if all_rubrics_list is not None:
        GLOBAL_REWARD_STATE.all_rubrics_list = all_rubrics_list
    if question is not None:
        GLOBAL_REWARD_STATE.question = question
    if policy_llm is not None:
        GLOBAL_REWARD_STATE.policy_llm = policy_llm
    
    if similarity_weight is not None:
        GLOBAL_REWARD_STATE.similarity_weight = similarity_weight
    if discrimination_weight is not None:
        GLOBAL_REWARD_STATE.discrimination_weight = discrimination_weight
    if diversity_weight is not None:
        GLOBAL_REWARD_STATE.diversity_weight = diversity_weight
    if reflect_weight is not None:
        GLOBAL_REWARD_STATE.reflect_weight = reflect_weight
    
    if enable_similarity is not None:
        GLOBAL_REWARD_STATE.enable_similarity = enable_similarity
    if enable_discrimination is not None:
        GLOBAL_REWARD_STATE.enable_discrimination = enable_discrimination
    if enable_diversity is not None:
        GLOBAL_REWARD_STATE.enable_diversity = enable_diversity
    if enable_reflect is not None:
        GLOBAL_REWARD_STATE.enable_reflect = enable_reflect

    if reflect_answers_per_baseline is not None:
        GLOBAL_REWARD_STATE.reflect_answers_per_baseline = reflect_answers_per_baseline
    if similarity_penalty_weight is not None:
        GLOBAL_REWARD_STATE.similarity_penalty_weight = similarity_penalty_weight
    if similarity_penalty_threshold is not None:
        GLOBAL_REWARD_STATE.similarity_penalty_threshold = similarity_penalty_threshold
    if max_concurrent_api_calls is not None:
        GLOBAL_REWARD_STATE.max_concurrent_api_calls = max_concurrent_api_calls

    print(f"[Reward Mode] switched to: {mode}")
    if mode == "rubrics_generator":
        print(
            "[Reward Config] weights: "
            f"sim={GLOBAL_REWARD_STATE.similarity_weight}(on={GLOBAL_REWARD_STATE.enable_similarity}), "
            f"disc={GLOBAL_REWARD_STATE.discrimination_weight}(on={GLOBAL_REWARD_STATE.enable_discrimination}), "
            f"div={GLOBAL_REWARD_STATE.diversity_weight}(on={GLOBAL_REWARD_STATE.enable_diversity}), "
            f"refl={GLOBAL_REWARD_STATE.reflect_weight}(on={GLOBAL_REWARD_STATE.enable_reflect})"
        )


def _build_reflect_answers_for_rubric(rubrics_idx: int) -> Dict[int, str] | None:
    """Convert {baseline_idx: {rubrics_idx: ans}} → {baseline_idx: ans}."""
    pool = GLOBAL_REWARD_STATE.reflect_answers_per_baseline
    if not pool:
        return None
    out: Dict[int, str] = {}
    for baseline_idx, rubrics_dict in pool.items():
        if isinstance(rubrics_dict, dict) and rubrics_idx in rubrics_dict:
            out[baseline_idx] = rubrics_dict[rubrics_idx]
    return out


def compute_adversarial_reward(data, **kwargs) -> torch.Tensor:
    """DataProto-style reward entry; returns a 1-D tensor of rewards."""
    global GLOBAL_REWARD_STATE

    evaluator = initialize_llm_evaluator()
    if VERL_AVAILABLE and hasattr(data, "batch"):
        responses = data.batch.get("responses", [])
        prompts = data.batch.get("prompts", [])
    elif isinstance(data, dict):
        responses = data.get("responses", [])
        prompts = data.get("prompts", [])
    else:
        raise TypeError(f"Unsupported data type: {type(data)}")
    
    print(f"[Reward] mode={GLOBAL_REWARD_STATE.mode}, responses={len(responses)}, prompts={len(prompts)}")
    
    rewards: List[float] = []
    
    if GLOBAL_REWARD_STATE.mode == "policy_llm":
        from reward_calculator import batch_calculate_policy_llm_rewards

        try:
            rewards, _ = batch_calculate_policy_llm_rewards(
                answers=responses,
                rubrics_list=GLOBAL_REWARD_STATE.rubrics_list,
                llm_evaluator=evaluator,
                question=GLOBAL_REWARD_STATE.question,
                max_concurrent_api_calls=GLOBAL_REWARD_STATE.max_concurrent_api_calls,
            )
        except Exception as e:  # noqa: BLE001
            print(f"[Reward] batch policy-llm reward failed, falling back per-sample: {e}")
            rewards = [0.0] * len(responses)

            def compute_single(idx: int, response: str) -> tuple[int, float]:
                r, _ = calculate_policy_llm_reward(
                    answer=response,
                    rubrics_list=GLOBAL_REWARD_STATE.rubrics_list,
                    llm_evaluator=evaluator,
                    question=GLOBAL_REWARD_STATE.question,
                )
                return idx, float(r)

            mc = max(1, min(int(GLOBAL_REWARD_STATE.max_concurrent_api_calls), 64))
            with ThreadPoolExecutor(max_workers=min(len(responses), mc)) as ex:
                fut2idx = {ex.submit(compute_single, i, r): i for i, r in enumerate(responses)}
                for fut in as_completed(fut2idx):
                    try:
                        idx, val = fut.result()
                        rewards[idx] = val
                    except Exception as e2:  # noqa: BLE001
                        idx = fut2idx[fut]
                        print(f"[Reward] response {idx} fallback failed: {e2}")
                        rewards[idx] = 0.0
            
    elif GLOBAL_REWARD_STATE.mode == "rubrics_generator":
        rewards = [0.0] * len(responses)

        def compute_single(idx: int, generated_rubrics: str) -> tuple[int, float, Dict[str, float]]:
            details = calculate_rubrics_generator_reward(
                        generated_rubrics=generated_rubrics,
                        ground_truth_rubrics=GLOBAL_REWARD_STATE.ground_truth_rubrics,
                        policy_answers=GLOBAL_REWARD_STATE.policy_answers,
                        llm_evaluator=evaluator,
                        question=GLOBAL_REWARD_STATE.question,
                        all_rubrics_list=GLOBAL_REWARD_STATE.all_rubrics_list,
                        policy_llm=GLOBAL_REWARD_STATE.policy_llm,
                        similarity_weight=GLOBAL_REWARD_STATE.similarity_weight,
                        discrimination_weight=GLOBAL_REWARD_STATE.discrimination_weight,
                        diversity_weight=GLOBAL_REWARD_STATE.diversity_weight,
                    reflect_weight=GLOBAL_REWARD_STATE.reflect_weight,
                    enable_similarity=GLOBAL_REWARD_STATE.enable_similarity,
                    enable_discrimination=GLOBAL_REWARD_STATE.enable_discrimination,
                    enable_diversity=GLOBAL_REWARD_STATE.enable_diversity,
                    enable_reflect=GLOBAL_REWARD_STATE.enable_reflect,
                similarity_penalty_weight=GLOBAL_REWARD_STATE.similarity_penalty_weight,
                similarity_penalty_threshold=GLOBAL_REWARD_STATE.similarity_penalty_threshold,
                reflect_answers=_build_reflect_answers_for_rubric(idx),
                    return_details=True,
            )
            return idx, float(details.get("total_reward", 0.0) if isinstance(details, dict) else details), details

        with ThreadPoolExecutor(max_workers=min(len(responses), 4)) as ex:
            fut2idx = {ex.submit(compute_single, i, r): i for i, r in enumerate(responses)}
            for fut in as_completed(fut2idx):
                try:
                    idx, val, details = fut.result()
                    rewards[idx] = val
                    if isinstance(details, dict):
                        GLOBAL_REWARD_STATE.sub_rewards_history.append(details)
                except Exception as e:  # noqa: BLE001
                    idx = fut2idx[fut]
                    print(f"[Reward] rubrics_generator sample {idx} failed: {e}")
                    rewards[idx] = 0.0

    else:
        raise ValueError(f"Unknown reward mode: {GLOBAL_REWARD_STATE.mode}")

    rewards_tensor = torch.tensor(rewards, dtype=torch.float32)
    print(
        f"[Reward] done: range=[{rewards_tensor.min().item():.4f}, "
        f"{rewards_tensor.max().item():.4f}], mean={rewards_tensor.mean().item():.4f}"
    )
    return rewards_tensor


def adversarial_reward_fn_batch(
    data_sources: List[str] | None = None,
    solution_strs: List[str] | None = None,
    ground_truths: List[str] | None = None,
    extra_infos: List[Dict[str, Any]] | None = None,
    **kwargs,
) -> List[Dict[str, Any]]:
    """Batch entry compatible with verl's ``BatchRewardManager``."""
    import numpy as np
    
    solution_strs = solution_strs if solution_strs is not None else []
    data_sources = data_sources if data_sources is not None else []
    ground_truths = ground_truths if ground_truths is not None else []
    extra_infos = extra_infos if extra_infos is not None else []

    if isinstance(solution_strs, np.ndarray):
        solution_strs = solution_strs.tolist()
    if isinstance(data_sources, np.ndarray):
        data_sources = data_sources.tolist()
    if isinstance(ground_truths, np.ndarray):
        ground_truths = ground_truths.tolist()
    if isinstance(extra_infos, np.ndarray):
        extra_infos = extra_infos.tolist()
    
    batch_size = len(solution_strs)
    if batch_size == 0:
        return []
    
    data_sources = list(data_sources) if len(data_sources) > 0 else [None] * batch_size
    ground_truths = list(ground_truths) if len(ground_truths) > 0 else [None] * batch_size
    extra_infos = list(extra_infos) if len(extra_infos) > 0 else [{}] * batch_size
    
    while len(data_sources) < batch_size:
        data_sources.append(None)
    while len(ground_truths) < batch_size:
        ground_truths.append(None)
    while len(extra_infos) < batch_size:
        extra_infos.append({})
    
    results: List[Dict[str, Any]] = []
    for i in range(batch_size):
        try:
            score = adversarial_reward_fn(
                data_source=data_sources[i],
                solution_str=solution_strs[i],
                ground_truth=ground_truths[i],
                extra_info=extra_infos[i],
                **kwargs,
            )
            if isinstance(score, dict):
                results.append(score)
            else:
                results.append({"score": float(score) if score is not None else 0.0})
        except Exception as e:  # noqa: BLE001
            print(f"[Batch Reward] sample {i} failed: {e}")
            results.append({"score": 0.0})
    
        while len(results) < batch_size:
            results.append({"score": 0.0})
    return results[:batch_size]


def adversarial_reward_fn(
    data_source: str | None = None,
    solution_str: str | None = None,
    ground_truth: str | None = None,
    extra_info: Dict[str, Any] | None = None,
    **kwargs,
) -> Any:
    """Single-sample entry compatible with verl's ``NaiveRewardManager``."""
    global GLOBAL_REWARD_STATE
    evaluator = initialize_llm_evaluator()
    sample_idx = extra_info.get("sample_idx", 0) if extra_info else 0

    mode = None
    rubrics_list = None
    question = None
    policy_answers = None
    ground_truth_rubrics = None
    all_rubrics_list = None
    
    if extra_info:
        mode = extra_info.get("adversarial_mode")
        rubrics_list = extra_info.get("adversarial_rubrics_list")
        question = extra_info.get(
            "adversarial_conversation_text", extra_info.get("adversarial_question")
        )
        policy_answers = extra_info.get("adversarial_policy_answers")
        ground_truth_rubrics = extra_info.get("adversarial_ground_truth_rubrics")
        all_rubrics_list = extra_info.get("adversarial_all_rubrics_list")
    
    if mode is None:
        mode = GLOBAL_REWARD_STATE.mode
    if rubrics_list is None:
        rubrics_list = GLOBAL_REWARD_STATE.rubrics_list
    if question is None:
        question = GLOBAL_REWARD_STATE.question
    if policy_answers is None:
        policy_answers = GLOBAL_REWARD_STATE.policy_answers
    if ground_truth_rubrics is None:
        ground_truth_rubrics = GLOBAL_REWARD_STATE.ground_truth_rubrics
    if all_rubrics_list is None:
        all_rubrics_list = GLOBAL_REWARD_STATE.all_rubrics_list
    
    if mode == "policy_llm":
        if not solution_str:
            return 0.0
        if not rubrics_list:
            print("[Reward Warning] rubrics_list is empty for policy_llm mode")
            return 0.0
        
        if extra_info and "cached_reward" in extra_info:
            return float(extra_info["cached_reward"])

            rubrics_reward, _ = calculate_policy_llm_reward(
                answer=solution_str,
                rubrics_list=rubrics_list,
                llm_evaluator=evaluator,
            question=question,
        )
        return float(rubrics_reward)
    
    elif mode == "rubrics_generator":
        if not solution_str:
            return 0.0

        # Hard format penalty: if the rubric output is not parseable,
        # downstream rewards collapse to 0, so we explicitly signal -1.0.
        if not parse_rubrics(solution_str):
            print(
                f"[Reward Debug] rubrics parse failed → format penalty -1.0. "
                f"preview: {solution_str[:120]}"
            )
            return -1.0
        
        cached_similarity = cached_discrimination = cached_diversity = cached_reflect = None
        if extra_info and "cached_reward_details" in extra_info:
            cached_details_tmp = extra_info.get("cached_reward_details", {})
            if isinstance(cached_details_tmp, dict):
                cached_similarity = cached_details_tmp.get("similarity_reward")
                cached_discrimination = cached_details_tmp.get("discrimination_reward")
                cached_diversity = cached_details_tmp.get("diversity_reward")
                cached_reflect = cached_details_tmp.get("reflect_reward")
                if "total_reward" in cached_details_tmp:
                    GLOBAL_REWARD_STATE.sub_rewards_history.append(cached_details_tmp)
                    return {
                        "score": cached_details_tmp.get("total_reward", 0.0),
                        "similarity_reward": cached_details_tmp.get("similarity_reward", 0.0),
                        "discrimination_reward": cached_details_tmp.get("discrimination_reward", 0.0),
                        "diversity_reward": cached_details_tmp.get("diversity_reward", 0.0),
                        "reflect_reward": cached_details_tmp.get("reflect_reward", 0.0),
                        "similarity_penalty": cached_details_tmp.get("similarity_penalty", 0.0),
                    }

        # Read weights/toggles from extra_info first (avoids stale global state under Ray).
        def _ei(key, default):
            return extra_info.get(key, default) if extra_info else default

        similarity_weight = _ei("adversarial_similarity_weight", GLOBAL_REWARD_STATE.similarity_weight)
        discrimination_weight = _ei("adversarial_discrimination_weight", GLOBAL_REWARD_STATE.discrimination_weight)
        diversity_weight = _ei("adversarial_diversity_weight", GLOBAL_REWARD_STATE.diversity_weight)
        reflect_weight = _ei("adversarial_reflect_weight", GLOBAL_REWARD_STATE.reflect_weight)
        similarity_penalty_weight = _ei(
            "adversarial_similarity_penalty_weight", GLOBAL_REWARD_STATE.similarity_penalty_weight
        )
        similarity_penalty_threshold = _ei(
            "adversarial_similarity_penalty_threshold", GLOBAL_REWARD_STATE.similarity_penalty_threshold
        )
        enable_similarity = _ei("adversarial_enable_similarity", GLOBAL_REWARD_STATE.enable_similarity)
        enable_discrimination = _ei("adversarial_enable_discrimination", GLOBAL_REWARD_STATE.enable_discrimination)
        enable_diversity = _ei("adversarial_enable_diversity", GLOBAL_REWARD_STATE.enable_diversity)
        enable_reflect = _ei("adversarial_enable_reflect", GLOBAL_REWARD_STATE.enable_reflect)

        reflect_answers_per_baseline = _ei("adversarial_reflect_answers_per_baseline", None)
        reflect_answers_for_rubric: Dict[int, str] | None = None
        if isinstance(reflect_answers_per_baseline, dict):
            reflect_answers_for_rubric = {}
            for baseline_idx, rubrics_dict in reflect_answers_per_baseline.items():
                if isinstance(rubrics_dict, dict) and sample_idx in rubrics_dict:
                    reflect_answers_for_rubric[baseline_idx] = rubrics_dict[sample_idx]
        
        use_cached_similarity = cached_similarity is not None
        use_cached_discrimination = cached_discrimination is not None
        use_cached_diversity = cached_diversity is not None
        use_cached_reflect = cached_reflect is not None

        reward_details = calculate_rubrics_generator_reward(
            generated_rubrics=solution_str,
            ground_truth_rubrics=ground_truth_rubrics or "",
            policy_answers=policy_answers or [],
            llm_evaluator=evaluator,
            question=question or "",
            all_rubrics_list=all_rubrics_list or [],
            policy_llm=GLOBAL_REWARD_STATE.policy_llm,
            similarity_weight=similarity_weight,
            discrimination_weight=discrimination_weight,
            diversity_weight=diversity_weight,
            reflect_weight=reflect_weight,
            similarity_penalty_weight=similarity_penalty_weight,
            similarity_penalty_threshold=similarity_penalty_threshold,
            enable_similarity=enable_similarity and not use_cached_similarity,
            enable_discrimination=enable_discrimination and not use_cached_discrimination,
            enable_diversity=enable_diversity and not use_cached_diversity,
            enable_reflect=enable_reflect and not use_cached_reflect,
            skip_similarity_reason="cached_reward_details" if use_cached_similarity else None,
            skip_discrimination_reason="cached_reward_details" if use_cached_discrimination else None,
            skip_diversity_reason="cached_reward_details" if use_cached_diversity else None,
            skip_reflect_reason="cached_reward_details" if use_cached_reflect else None,
            return_details=True,
            reflect_answers=reflect_answers_for_rubric,
        )

        if use_cached_similarity:
            reward_details["similarity_reward"] = cached_similarity
        if use_cached_discrimination:
            reward_details["discrimination_reward"] = cached_discrimination
        if use_cached_diversity:
            reward_details["diversity_reward"] = cached_diversity
        if use_cached_reflect:
            reward_details["reflect_reward"] = cached_reflect

        reward_details["total_reward"] = (
            similarity_weight * reward_details.get("similarity_reward", 0.0)
            + discrimination_weight * reward_details.get("discrimination_reward", 0.0)
            + diversity_weight * reward_details.get("diversity_reward", 0.0)
            + reflect_weight * reward_details.get("reflect_reward", 0.0)
            - similarity_penalty_weight * reward_details.get("similarity_penalty", 0.0)
        )

        GLOBAL_REWARD_STATE.sub_rewards_history.append(reward_details)
        return {
            "score": reward_details["total_reward"],
            "similarity_reward": reward_details.get("similarity_reward", 0.0),
            "discrimination_reward": reward_details.get("discrimination_reward", 0.0),
            "diversity_reward": reward_details.get("diversity_reward", 0.0),
            "reflect_reward": reward_details.get("reflect_reward", 0.0),
            "similarity_penalty": reward_details.get("similarity_penalty", 0.0),
        }
    
    else:
        raise ValueError(f"Unknown reward mode: {mode}")


def adversarial_reward_fn_dataproto(data, **kwargs) -> torch.Tensor:
    """DataProto-style wrapper for backward compatibility."""
    return compute_adversarial_reward(data, **kwargs)
