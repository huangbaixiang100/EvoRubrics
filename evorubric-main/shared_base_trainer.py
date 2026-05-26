"""Shared-base dual-LoRA adversarial trainer for EvoRubrics."""
import os
import json
import socket
import sys
import time
import uuid
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import ray
import torch
from omegaconf import OmegaConf
from tqdm import tqdm
from torch.utils.data import RandomSampler

package_root = Path(__file__).resolve().parent
project_root = package_root.parent
sys.path.insert(0, str(project_root / "third_party" / "verl"))
sys.path.insert(0, str(project_root / "rubric_guidance"))
sys.path.insert(0, str(package_root))
sys.path.insert(0, str(project_root))

try:
    from verl import DataProto
except ImportError:
    from verl.protocol import DataProto

from verl.trainer.ppo.ray_trainer import (
    ResourcePoolManager,
    compute_advantage,
    compute_response_mask,
    Role,
)
from verl.trainer.ppo.core_algos import AdvantageEstimator, agg_loss
from verl.trainer.ppo.reward import compute_reward
from verl.trainer.ppo.utils import need_critic, need_reference_policy, need_reward_model
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.utils.config import validate_config
from verl.utils.fs import copy_to_local
from verl.utils import hf_tokenizer, hf_processor
from verl.utils.dataset.rl_dataset import collate_fn
from verl.utils.metric import reduce_metrics
from verl.utils.debug import marked_timer
from verl.utils.tracking import Tracking
from verl.single_controller.ray import RayWorkerGroup
from adversarial_dataset import AdversarialDataset
from dual_lora_worker import (
    DualLoRAActorRolloutRefWorker,
    POLICY_LLM_ADAPTER,
    RUBRICS_GEN_ADAPTER,
)
from models import (
    POLICY_LLM_SYSTEM_PROMPT,
    RUBRICS_GENERATOR_SYSTEM_PROMPT,
    UNIVERSAL_RUBRICS_GENERATOR_SYSTEM_PROMPT,
    REFLECT_SYSTEM_PROMPT,
    REFLECT_PROMPT_TEMPLATE,
    RUBRICS_EXEMPLAR_TEMPLATE,
    format_answer_aware_condition,
)
from llm_evaluator import LLMEvaluator
from reward_calculator import (
    calculate_rubrics_generator_reward,
    parse_rubrics,
    normalize_score_to_minus1_1,
    batch_calculate_diversity_rewards,
    batch_calculate_intra_rubric_similarity_penalties,
    filter_similar_rubric_items_by_semantic_similarity,
    score_answer_with_rubrics,
    batch_calculate_similarity_rewards,
)
from rubrics_vllm_utils import (
    RubricsVLLMConfig,
    RubricsVLLMRuntime,
)


class SharedBaseAdversarialTrainer:
    """English docstring."""
    def __init__(self):
        self.role_worker_mapping = {}
        self.mapping = {}
        self.step_count = 0
        self.global_steps = 0
        
        self.actor_rollout_wg = None
        self.tokenizer = None
        self.processor = None
        
        self.policy_dataset = None
        self.rubrics_dataset = None
        
        self.config = None
        self._sentence_transformer_warmed = False
        
        # 
        self.previous_best_responses = None
        self._pending_best_responses = None
        self._policy_accum_batches: List[DataProto] = []
        self._rubrics_accum_batches: List[DataProto] = []

    def _exemplar_trick_enabled(self, config) -> bool:
        adv = getattr(config, "adversarial", None) or {}
        if hasattr(adv, "get"):
            return bool(adv.get("enable_exemplar_trick", True))
        return True
    
    def add_dual_lora_worker(self, config):
        """English docstring."""
        ray_worker_group_cls = RayWorkerGroup
        self.role_worker_mapping[Role.ActorRollout] = ray.remote(DualLoRAActorRolloutRefWorker)
        return DualLoRAActorRolloutRefWorker, ray_worker_group_cls
    
    def init_resource_pool_mgr(self, config, pool_name="shared_pool"):
        """English docstring."""
        gpus_per_node = config.trainer.n_gpus_per_node
        
        resource_pool_spec = {
            pool_name: [gpus_per_node] * config.trainer.nnodes,
        }
        
        mapping = {}
        mapping[Role.ActorRollout] = pool_name
        
        if config.algorithm.use_kl_in_reward or config.actor_rollout_ref.actor.use_kl_loss:
            mapping[Role.RefPolicy] = pool_name
        
        resource_pool_manager = ResourcePoolManager(
            resource_pool_spec=resource_pool_spec,
            mapping=mapping
        )
        return resource_pool_manager
    
    def run(self, config):
        """English docstring."""
        print(f"=" * 80)
        print("[message omitted]")
        print(f"Hostname: {socket.gethostname()}, PID: {os.getpid()}")
        print(f"=" * 80)
        
        from pprint import pprint
        pprint(OmegaConf.to_container(config, resolve=True))
        OmegaConf.resolve(config)
        
        self.config = config
        
        print("[message omitted]")
        worker_cls, ray_worker_group_cls = self.add_dual_lora_worker(config)
        
        print("[message omitted]")
        total_gpus = config.trainer.n_gpus_per_node * config.trainer.nnodes
        print("[message omitted]")
        
        resource_pool_manager = self.init_resource_pool_mgr(config, pool_name="shared_pool")
        resource_pool_manager.create_resource_pool()
        
        print("[message omitted]")
        model_path = config.get("policy_model", {}).get("path") or config.actor_rollout_ref.model.path
        print("[message omitted]")
        
        local_path = copy_to_local(
            model_path,
            use_shm=config.actor_rollout_ref.model.get("use_shm", False)
        )
        
        trust_remote_code = config.data.get("trust_remote_code", False)
        self.tokenizer = hf_tokenizer(
            local_path, 
            trust_remote_code=trust_remote_code,
            use_fast=False,  
            padding_side='left',  
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        self.processor = hf_processor(local_path, trust_remote_code=trust_remote_code, use_fast=True)
        
        print("[message omitted]"
              f"trust_remote_code={trust_remote_code}")
        
        print("[message omitted]")
        from verl.trainer.ppo.reward import load_reward_manager
        
        reward_fn = load_reward_manager(
            config, self.tokenizer, num_examine=0, **config.reward_model.get("reward_kwargs", {})
        )
        
        print("[message omitted]")
        
        policy_dataset = AdversarialDataset(
            data_files=config.data.train_files,
            tokenizer=self.tokenizer,
            processor=self.processor,
            config=config.data,
            max_samples=config.data.get("train_max_samples", -1),
            mode="policy_llm"
        )
        
        rubrics_dataset = AdversarialDataset(
            data_files=config.data.train_files,
            tokenizer=self.tokenizer,
            processor=self.processor,
            config=config.data,
            max_samples=config.data.get("train_max_samples", -1),
            mode="rubrics_generator"
        )
        
        self.policy_dataset = policy_dataset
        self.rubrics_dataset = rubrics_dataset
        
        print("[message omitted]")
        worker_config = OmegaConf.create(OmegaConf.to_container(config, resolve=True))
        worker_config.actor_rollout_ref.model.path = local_path
        
        from verl.single_controller.ray.base import RayClassWithInitArgs
        
        resource_pool = resource_pool_manager.get_resource_pool(Role.ActorRollout)
        
        worker_cls_with_args = RayClassWithInitArgs(
            cls=self.role_worker_mapping[Role.ActorRollout],
            config=worker_config.actor_rollout_ref,
            role="actor_rollout_ref",
        )
        
        self.actor_rollout_wg = ray_worker_group_cls(
            resource_pool=resource_pool,
            ray_cls_with_init=worker_cls_with_args,
            name_prefix="dual_lora_worker",
        )
        
        self.actor_rollout_wg.init_model()
        
        self.reward_fn = reward_fn
        
        print("[message omitted]")
        self._adversarial_training_loop(config)
        
        print("\n" + "=" * 80)
        print("[message omitted]")
        print("=" * 80)
    
    def _safe_decode_token_ids(self, token_ids, sequence_idx=0):
        """English docstring."""
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.cpu().tolist()
        if not token_ids:
            return ""
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)
    
    def _adversarial_training_loop(self, config):
        """English docstring."""
        logger = Tracking(
            project_name=config.trainer.project_name,
            experiment_name=config.trainer.experiment_name,
            default_backend=config.trainer.logger,
            config=OmegaConf.to_container(config, resolve=True),
        )
        
        self.swanlab_run = None
        swanlab_config = config.trainer.get("swanlab", {})
        if swanlab_config.get("enabled", False):
            try:
                import swanlab
                api_key = swanlab_config.get("api_key")
                if api_key:
                    swanlab.login(api_key=api_key)
                
                self.swanlab_run = swanlab.init(
                    project=config.trainer.project_name,
                    experiment_name=config.trainer.experiment_name + "_shared_base",
                    workspace=swanlab_config.get("workspace"),
                    config=OmegaConf.to_container(config, resolve=True),
                )
                print("[message omitted]")
            except ImportError:
                print("[message omitted]")
            except Exception as e:
                print("[message omitted]")
        
        resume_step = config.trainer.get("resume_step", 0)
        self.global_steps = resume_step
        if resume_step > 0:
            print("[message omitted]")
            
            policy_lora_path = config.actor_rollout_ref.model.get("policy_lora_path", None)
            rubrics_lora_path = config.actor_rollout_ref.model.get("rubrics_lora_path", None)
            
            if policy_lora_path:
                policy_step_dir = os.path.dirname(policy_lora_path)
                print("[message omitted]")
                self.actor_rollout_wg.load_optimizer_checkpoint(policy_step_dir, POLICY_LLM_ADAPTER)
            
            if rubrics_lora_path:
                rubrics_step_dir = os.path.dirname(rubrics_lora_path)
                print("[message omitted]")
                self.actor_rollout_wg.load_optimizer_checkpoint(rubrics_step_dir, RUBRICS_GEN_ADAPTER)
        
        num_rubrics = config.data.get("num_rubrics", 4)
        num_answers = config.data.get("num_answers", 4)
        save_lora_freq = config.trainer.get("save_lora_freq", 2)
        training_flow = str(config.adversarial.get("training_flow", "unified")).strip().lower()
        print(
            "[message omitted]"
            ""
        )
        
        exp_log_dir = os.path.join(config.trainer.default_local_dir, "training_logs")
        os.makedirs(exp_log_dir, exist_ok=True)
        log_file = os.path.join(exp_log_dir, f"shared_base_training_{time.strftime('%Y%m%d_%H%M%S')}.log")
        self.log_file = open(log_file, "w", encoding="utf-8")
        print("[message omitted]")
        
        self.prompt_output_dir = os.path.join(exp_log_dir, "prompts_and_outputs")
        os.makedirs(self.prompt_output_dir, exist_ok=True)
        print("[message omitted]")
        
        from torch.utils.data import DataLoader
        
        policy_dataloader = DataLoader(
            self.policy_dataset,
            batch_size=config.data.train_batch_size,
            shuffle=config.data.shuffle,
            collate_fn=collate_fn,
        )
        
        num_batches_per_epoch = len(policy_dataloader)
        total_batches = config.trainer.total_epochs * num_batches_per_epoch
        progress_bar = tqdm(
            total=total_batches,
            initial=self.global_steps,
            desc="Shared Base Adversarial Training"
        )
        
        skip_epochs = resume_step // num_batches_per_epoch if num_batches_per_epoch > 0 else 0
        skip_batches_in_current_epoch = resume_step % num_batches_per_epoch if num_batches_per_epoch > 0 else 0
        
        # ==========================================================
        # ==========================================================
        eval_config = config.trainer.get("eval", {})
        eval_enabled = eval_config.get("enabled", False)
        eval_before_train = eval_config.get("eval_before_train", False)
        eval_freq = eval_config.get("eval_freq", 100)
        
        if eval_enabled:
            print("[message omitted]")
            print("[message omitted]")
            print("[message omitted]")
            print("[message omitted]")
            print("[message omitted]")
            print("[message omitted]")
        
        if eval_enabled and eval_before_train:
            print("[message omitted]")
            self._run_evaluation(config, step=resume_step)
        
        for epoch in range(config.trainer.total_epochs):
            if epoch < skip_epochs:
                print("[message omitted]")
                continue
            
            print(f"\n{'='*80}")
            print(f"Epoch {epoch+1}/{config.trainer.total_epochs}")
            print(f"{'='*80}")
            
            for batch_idx, batch_dict in enumerate(policy_dataloader):
                if epoch == skip_epochs and batch_idx < skip_batches_in_current_epoch:
                    if batch_idx == 0:
                        print("[message omitted]")
                    continue
                
                metrics = {}
                timing_raw = {}
                
                original_batch_dict = {k: v.clone() if isinstance(v, torch.Tensor) else 
                                       (v.copy() if isinstance(v, np.ndarray) else v) 
                                       for k, v in batch_dict.items()}
                
                batch = self._create_dataproto(batch_dict)
                
                questions = batch.non_tensor_batch.get("question", [""])
                ground_truth_rubrics_list = batch.non_tensor_batch.get("ground_truth_rubrics", [""])
                golden_answer_list = batch.non_tensor_batch.get("golden_answer", [""] * len(questions))
                batch_size = len(questions)
                
                print("[message omitted]")
                print("[message omitted]")
                if batch_size > 1:
                    print("[message omitted]")
                
                self.log_file.write(f"\n{'='*80}\n")
                self.log_file.write("")
                self.log_file.write(f"{'='*80}\n")
                for q_idx, question in enumerate(questions):
                    self.log_file.write("")
                self.log_file.flush()
                
                try:
                    for i in range(torch.cuda.device_count()):
                        torch.cuda.reset_peak_memory_stats(i)
                except Exception as e:
                    print("[message omitted]")
                
                iteration_count = int(config.adversarial.get("num_iterations_per_query", 1))
                for iteration_idx in range(iteration_count):
                    if training_flow == "async_two_phase":
                        with marked_timer(f"async_policy_phase_{iteration_idx}", timing_raw):
                            policy_metrics = self._train_policy_llm_step(
                                batch=batch,
                                questions=questions,
                                ground_truth_rubrics_list=ground_truth_rubrics_list,
                                num_rubrics=num_rubrics,
                                num_answers=num_answers,
                                config=config,
                            )
                        for k, v in policy_metrics.items():
                            metrics[f"iter{iteration_idx}/policy_phase/{k}"] = v

                        with marked_timer(f"async_rubrics_phase_{iteration_idx}", timing_raw):
                            rubrics_metrics = self._train_rubrics_generator_step_batch(
                                batch=batch,
                                questions=questions,
                                ground_truth_rubrics_list=ground_truth_rubrics_list,
                                num_rubrics=num_rubrics,
                                num_answers=num_answers,
                                config=config,
                                golden_answer_list=golden_answer_list,
                            )
                        for k, v in rubrics_metrics.items():
                            metrics[f"iter{iteration_idx}/rubrics_phase/{k}"] = v
                    else:
                        with marked_timer(f"unified_iter_{iteration_idx}", timing_raw):
                            unified_metrics = self._train_unified_iteration(
                                batch=batch,
                                questions=questions,
                                num_rubrics=num_rubrics,
                                num_answers=num_answers,
                                config=config,
                                iteration_idx=iteration_idx,
                            )
                        for k, v in unified_metrics.items():
                            metrics[f"iter{iteration_idx}/{k}"] = v
                
                if self._exemplar_trick_enabled(config) and self._pending_best_responses is not None:
                    self.previous_best_responses = self._pending_best_responses
                    self._pending_best_responses = None
                    print("[message omitted]")
                
                metrics.update(timing_raw)
                logger.log(data=metrics, step=self.global_steps)
                
                self._log_to_swanlab(metrics)
                
                if self.global_steps % save_lora_freq == 0 and self.global_steps > 0:
                    self._save_checkpoints(config, epoch)
                
                self.global_steps += 1
                progress_bar.update(1)
                
                if eval_enabled and eval_freq > 0 and self.global_steps % eval_freq == 0:
                    print("[message omitted]")
                    self._run_evaluation(config, step=self.global_steps)
            
            print("[message omitted]")
            self._save_checkpoints(config, epoch)

        flushed_any = self._flush_pending_accum_updates()
        if flushed_any:
            print("[message omitted]")
            self._save_checkpoints(config, config.trainer.total_epochs - 1)
        
        progress_bar.close()
        if hasattr(self, 'log_file') and self.log_file:
            self.log_file.close()
        
        if self.swanlab_run is not None:
            try:
                import swanlab
                swanlab.finish()
                print("[message omitted]")
            except Exception as e:
                print("[message omitted]")

    def _build_prompt_batch_from_strings(self, prompts: List[str], config) -> DataProto:
        encoded = self.tokenizer(prompts, padding=True, truncation=False, return_tensors="pt")
        attention_mask = encoded["attention_mask"]
        position_ids = attention_mask.long().cumsum(dim=1) - 1
        position_ids = position_ids.clamp(min=0)
        return DataProto.from_dict(
            tensors={
                "input_ids": encoded["input_ids"],
                "attention_mask": encoded["attention_mask"],
                "position_ids": position_ids,
            },
            non_tensors={"prompt": np.array(prompts, dtype=object)},
            meta_info={
                "n": 1,
                "do_sample": True,
                "temperature": float(config.adversarial.get("generation_temperature", 1.0)),
                "top_p": 0.95,
                "top_k": 50,
            },
        )

    def _build_token_level_reward_tensor(self, full_batch: DataProto, sample_rewards: List[float]) -> torch.Tensor:
        responses = full_batch.batch["responses"]
        reward_tensor = torch.zeros(
            responses.shape, dtype=torch.float32, device=responses.device
        )
        pad_token_id = getattr(self.tokenizer, "pad_token_id", None)
        if pad_token_id is not None:
            response_mask = (responses != pad_token_id).long()
        else:
            response_mask = torch.ones_like(responses, dtype=torch.long)

        for i in range(min(len(sample_rewards), reward_tensor.size(0))):
            valid_len = int(response_mask[i].sum().item())
            if valid_len > 0:
                reward_tensor[i, valid_len - 1] = float(sample_rewards[i])
        return reward_tensor

    def _flush_pending_accum_updates(self) -> bool:
        """English docstring."""
        flushed_any = False
        if self._policy_accum_batches:
            pending = len(self._policy_accum_batches)
            print("[message omitted]")
            self.actor_rollout_wg.set_adapter(POLICY_LLM_ADAPTER)
            merged_policy_batch = self._merge_training_batches(
                self._policy_accum_batches,
                uid_prefix=f"policy_accum_flush_step{self.global_steps}",
            )
            self.actor_rollout_wg.update_actor(merged_policy_batch)
            self._policy_accum_batches = []
            flushed_any = True

        if self._rubrics_accum_batches:
            pending = len(self._rubrics_accum_batches)
            print("[message omitted]")
            self.actor_rollout_wg.set_adapter(RUBRICS_GEN_ADAPTER)
            merged_rubrics_batch = self._merge_training_batches(
                self._rubrics_accum_batches,
                uid_prefix=f"rubrics_accum_flush_step{self.global_steps}",
            )
            self.actor_rollout_wg.update_actor(merged_rubrics_batch)
            self._rubrics_accum_batches = []
            flushed_any = True
        return flushed_any

    def _pick_ref_answers_for_aware(self, q_answers: List[str], num_refs: int) -> List[str]:
        """English docstring."""
        import random

        if not q_answers or num_refs <= 0:
            return []
        if len(q_answers) >= num_refs:
            return random.sample(q_answers, num_refs)
        if len(q_answers) == 1:
            return [q_answers[0]] * num_refs
        return random.choices(q_answers, k=num_refs)

    def _build_answer_aware_rubrics_prompts(
        self,
        questions: List[str],
        answers_per_query: List[List[str]],
        num_rubrics: int,
        prompt_messages_json_list: Optional[List[str]] = None,
        instruction_list: Optional[List[str]] = None,
    ) -> List[str]:
        prompts: List[str] = []
        import copy

        answer_aware_num_refs = int(self.config.adversarial.get("answer_aware_num_refs", 2))
        dataset_mode = self.config.data.get("dataset_mode", "golden_rubrics")
        if dataset_mode == "open_rubrics":
            system_prompt = UNIVERSAL_RUBRICS_GENERATOR_SYSTEM_PROMPT
        else:
            system_prompt = UNIVERSAL_RUBRICS_GENERATOR_SYSTEM_PROMPT

        for q_idx, question in enumerate(questions):
            q_answers = answers_per_query[q_idx] if q_idx < len(answers_per_query) else []
            instruction = ""
            if instruction_list is not None and q_idx < len(instruction_list):
                instruction = instruction_list[q_idx] or ""

            raw_messages = []
            if prompt_messages_json_list is not None and q_idx < len(prompt_messages_json_list):
                try:
                    candidate = json.loads(prompt_messages_json_list[q_idx])
                    if isinstance(candidate, list):
                        raw_messages = candidate
                except Exception:
                    raw_messages = []

            base_messages = [{"role": "system", "content": system_prompt}]
            if instruction:
                base_messages.append({"role": "system", "content": instruction})

            if raw_messages:
                for msg in raw_messages:
                    role = msg.get("role", "")
                    content = msg.get("content", "")
                    if role not in ("user", "assistant", "system") or not isinstance(content, str):
                        continue
                    base_messages.append({"role": role, "content": content})
            else:
                base_messages.append({"role": "user", "content": question})

            for _ in range(num_rubrics):
                messages = copy.deepcopy(base_messages)
                if answer_aware_num_refs > 0:
                    refs = self._pick_ref_answers_for_aware(q_answers, answer_aware_num_refs)
                    aware = format_answer_aware_condition(refs) if refs else ""
                    if aware:
                        if raw_messages:
                            for i in range(len(messages) - 1, -1, -1):
                                if messages[i].get("role") == "user":
                                    messages[i]["content"] = f"{messages[i]['content']}\n\n{aware}"
                                    break
                            else:
                                messages.append({"role": "user", "content": aware})
                        else:
                            messages[-1]["content"] = f"{messages[-1]['content']}\n\n{aware}"
                prompt = self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
                )
                prompts.append(prompt)
        return prompts

    def _append_answer_aware_to_last_user(self, messages: List[Dict[str, str]], q_answers: List[str]) -> None:
        """English docstring."""
        answer_aware_num_refs = int(self.config.adversarial.get("answer_aware_num_refs", 2))
        if answer_aware_num_refs <= 0 or not q_answers:
            return
        refs = self._pick_ref_answers_for_aware(q_answers, answer_aware_num_refs)
        if not refs:
            return
        aware = format_answer_aware_condition(refs)
        for i in range(len(messages) - 1, -1, -1):
            if messages[i].get("role") == "user":
                messages[i]["content"] = f"{messages[i]['content']}\n\n{aware}"
                return
        messages.append({"role": "user", "content": aware})

    @staticmethod
    def _format_conversation_text(
        prompt_messages_json_list: Optional[List[str]],
        q_idx: int,
        questions: List[str],
    ) -> str:
        """English docstring."""
        if prompt_messages_json_list is not None and q_idx < len(prompt_messages_json_list):
            try:
                messages = json.loads(prompt_messages_json_list[q_idx])
                if isinstance(messages, list) and len(messages) > 1:
                    parts = []
                    for msg in messages:
                        role = msg.get("role", "")
                        content = msg.get("content", "")
                        if role == "system" or not isinstance(content, str):
                            continue
                        label = "User" if role == "user" else "Assistant"
                        parts.append(f"[{label}]: {content}")
                    if parts:
                        return "\n\n".join(parts)
            except Exception:
                pass
        return questions[q_idx] if q_idx < len(questions) else ""

    def _train_unified_iteration(
        self,
        batch: DataProto,
        questions: List[str],
        num_rubrics: int,
        num_answers: int,
        config,
        iteration_idx: int,
    ) -> Dict[str, float]:
        metrics: Dict[str, float] = {}
        batch_size = len(questions)
        max_retries = int(config.adversarial.get("max_rubrics_parse_retries", 3))
        adv_cfg = config.adversarial
        rubrics_update_freq = max(1, int(adv_cfg.get("rubrics_update_freq", 1)))
        should_update_rubrics = (self.global_steps % rubrics_update_freq) == 0
        answer_aware_num_refs = int(adv_cfg.get("answer_aware_num_refs", 2))
        answer_aware_enabled = answer_aware_num_refs > 0

        ans_gen_batch = None
        ans_gen_out = None
        answers_per_query: List[List[str]] = []
        prompt_messages_json_list = None
        instruction_list = None
        if batch.non_tensor_batch is not None:
            if "prompt_messages_json" in batch.non_tensor_batch:
                prompt_messages_json_list = list(batch.non_tensor_batch["prompt_messages_json"])
            if "instruction" in batch.non_tensor_batch:
                instruction_list = list(batch.non_tensor_batch["instruction"])

        conversation_texts = [
            self._format_conversation_text(prompt_messages_json_list, q_idx, questions)
            for q_idx in range(batch_size)
        ]

        # Step A/B:
        if answer_aware_enabled:
            self.actor_rollout_wg.set_adapter(POLICY_LLM_ADAPTER)
            ans_gen_batch = self._prepare_answer_generation_batch(
                batch, num_answers, config, use_diverse_sampling=False
            )
            ans_gen_out = self.actor_rollout_wg.generate_sequences(ans_gen_batch)
            response_ids = ans_gen_out.batch["responses"]
            total_expected_answers = batch_size * num_answers
            all_answers_flat = [
                self._safe_decode_token_ids(response_ids[i], sequence_idx=i)
                for i in range(min(total_expected_answers, response_ids.size(0)))
            ]
            for q_idx in range(batch_size):
                start = q_idx * num_answers
                answers_per_query.append(all_answers_flat[start : start + num_answers])

            self.actor_rollout_wg.set_adapter(RUBRICS_GEN_ADAPTER)
            rubrics_prompts = self._build_answer_aware_rubrics_prompts(
                questions,
                answers_per_query,
                num_rubrics,
                prompt_messages_json_list=prompt_messages_json_list,
                instruction_list=instruction_list,
            )
        else:
            self.actor_rollout_wg.set_adapter(RUBRICS_GEN_ADAPTER)
            empty_answers = [[] for _ in range(batch_size)]
            rubrics_prompts = self._build_answer_aware_rubrics_prompts(
                questions,
                empty_answers,
                num_rubrics,
                prompt_messages_json_list=prompt_messages_json_list,
                instruction_list=instruction_list,
            )

        total_expected_rubrics = batch_size * num_rubrics
        overlong_rubrics_indices = set()
        rollout_cfg = config.actor_rollout_ref.get("rollout", {})
        max_model_len_cfg = rollout_cfg.get("max_model_len", None)
        try:
            max_model_len = int(max_model_len_cfg) if max_model_len_cfg is not None else None
        except (TypeError, ValueError):
            max_model_len = None

        if max_model_len is not None and max_model_len > 0 and rubrics_prompts:
            for idx, prompt in enumerate(rubrics_prompts):
                encoded_no_pad = self.tokenizer(
                    prompt, padding=False, truncation=False, return_tensors="pt"
                )
                prompt_len = int(encoded_no_pad["input_ids"].shape[-1])
                if prompt_len > max_model_len:
                    overlong_rubrics_indices.add(idx)

            if overlong_rubrics_indices:
                for idx in overlong_rubrics_indices:
                    rubrics_prompts[idx] = "Skip overlong prompt."
                print(
                    "[message omitted]"
                    ""
                )

        rubrics_batch = self._build_prompt_batch_from_strings(rubrics_prompts, config)
        rubrics_out = self.actor_rollout_wg.generate_sequences(rubrics_batch)
        rubrics_ids = rubrics_out.batch["responses"]
        all_rubrics_flat = [
            self._safe_decode_token_ids(rubrics_ids[i], sequence_idx=i)
            for i in range(min(total_expected_rubrics, rubrics_ids.size(0)))
        ]
        parse_failures = np.zeros((batch_size, num_rubrics), dtype=bool)

        for attempt in range(1, max_retries + 1):
            failed_global_indices = []
            for idx in range(total_expected_rubrics):
                q_idx = idx // num_rubrics
                r_idx = idx % num_rubrics
                if idx in overlong_rubrics_indices:
                    parse_failures[q_idx, r_idx] = True
                    continue
                if idx >= len(all_rubrics_flat):
                    parse_failures[q_idx, r_idx] = True
                    failed_global_indices.append(idx)
                    continue
                if parse_rubrics(all_rubrics_flat[idx]):
                    parse_failures[q_idx, r_idx] = False
                else:
                    parse_failures[q_idx, r_idx] = True
                    failed_global_indices.append(idx)
            if not failed_global_indices or attempt >= max_retries:
                break
            retry_questions = [questions[i // num_rubrics] for i in failed_global_indices]
            retry_answers = (
                [answers_per_query[i // num_rubrics] for i in failed_global_indices]
                if answer_aware_enabled
                else [[] for _ in failed_global_indices]
            )
            retry_prompt_messages_json_list = (
                [prompt_messages_json_list[i // num_rubrics] for i in failed_global_indices]
                if prompt_messages_json_list is not None
                else None
            )
            retry_instruction_list = (
                [instruction_list[i // num_rubrics] for i in failed_global_indices]
                if instruction_list is not None
                else None
            )
            retry_prompts = self._build_answer_aware_rubrics_prompts(
                retry_questions,
                retry_answers,
                1,
                prompt_messages_json_list=retry_prompt_messages_json_list,
                instruction_list=retry_instruction_list,
            )
            retry_batch = self._build_prompt_batch_from_strings(retry_prompts, config)
            retry_out = self.actor_rollout_wg.generate_sequences(retry_batch)
            retry_ids = retry_out.batch["responses"]
            for local_idx, global_idx in enumerate(failed_global_indices):
                if local_idx < retry_ids.size(0):
                    all_rubrics_flat[global_idx] = self._safe_decode_token_ids(
                        retry_ids[local_idx], sequence_idx=local_idx
                    )

        rubrics_per_query = []
        for q_idx in range(batch_size):
            start = q_idx * num_rubrics
            rubrics_per_query.append(all_rubrics_flat[start : start + num_rubrics])

        enable_rubrics_similarity_filter = bool(adv_cfg.get("enable_rubrics_similarity_filter", True))
        rubrics_similarity_filter_threshold = float(adv_cfg.get("rubrics_similarity_filter_threshold", 0.75))
        if enable_rubrics_similarity_filter:
            total_items_before_filter = 0
            total_items_after_filter = 0
            for q_idx in range(batch_size):
                for r_idx in range(num_rubrics):
                    if parse_failures[q_idx, r_idx]:
                        continue
                    if r_idx >= len(rubrics_per_query[q_idx]):
                        continue
                    original_rubrics_text = rubrics_per_query[q_idx][r_idx]
                    filtered_rubrics_text, before_cnt, after_cnt = (
                        filter_similar_rubric_items_by_semantic_similarity(
                            original_rubrics_text,
                            threshold=rubrics_similarity_filter_threshold,
                        )
                    )
                    total_items_before_filter += int(before_cnt)
                    total_items_after_filter += int(after_cnt)
                    rubrics_per_query[q_idx][r_idx] = filtered_rubrics_text
                    print(
                        ""
                        ""
                    )
            metrics["rubrics_items_before_filter"] = float(total_items_before_filter)
            metrics["rubrics_items_after_filter"] = float(total_items_after_filter)
            metrics["rubrics_items_filtered"] = float(
                max(0, total_items_before_filter - total_items_after_filter)
            )

        if not answer_aware_enabled:
            self.actor_rollout_wg.set_adapter(POLICY_LLM_ADAPTER)
            ans_gen_batch = self._prepare_answer_generation_batch(
                batch, num_answers, config, use_diverse_sampling=False
            )
            ans_gen_out = self.actor_rollout_wg.generate_sequences(ans_gen_batch)
            response_ids = ans_gen_out.batch["responses"]
            total_expected_answers = batch_size * num_answers
            all_answers_flat = [
                self._safe_decode_token_ids(response_ids[i], sequence_idx=i)
                for i in range(min(total_expected_answers, response_ids.size(0)))
            ]
            for q_idx in range(batch_size):
                start = q_idx * num_answers
                answers_per_query.append(all_answers_flat[start : start + num_answers])

        evaluator = LLMEvaluator()
        max_concurrent_api_calls = max(1, int(adv_cfg.get("max_concurrent_api_calls", 64)))
        score_matrices: List[np.ndarray] = [
            np.full((num_rubrics, num_answers), -1.0, dtype=float) for _ in range(batch_size)
        ]
        parsed_rubrics_cache: List[List[Optional[List[Any]]]] = [
            [None for _ in range(num_rubrics)] for _ in range(batch_size)
        ]
        score_norm_sums: List[List[Tuple[float, float]]] = [
            [(0.0, 0.0) for _ in range(num_rubrics)] for _ in range(batch_size)
        ]

        eval_tasks = []
        for q_idx in range(batch_size):
            for r_idx, rubrics_text in enumerate(rubrics_per_query[q_idx]):
                if parse_failures[q_idx, r_idx]:
                    continue
                rubrics_items = parse_rubrics(rubrics_text)
                if not rubrics_items:
                    parse_failures[q_idx, r_idx] = True
                    continue
                parsed_rubrics_cache[q_idx][r_idx] = rubrics_items
                rubric_desc = [r.description for r in rubrics_items]
                rubric_points = [r.points for r in rubrics_items]
                positive_sum = sum(p for p in rubric_points if p > 0)
                negative_sum = sum(p for p in rubric_points if p < 0)
                score_norm_sums[q_idx][r_idx] = (positive_sum, negative_sum)
                eval_tasks.append((q_idx, r_idx, rubric_desc, rubric_points, positive_sum, negative_sum))

        reflect_prompts = []
        reflect_mapping = []
        for q_idx in range(batch_size):
            for r_idx in range(num_rubrics):
                if parse_failures[q_idx, r_idx]:
                    continue
                for baseline_idx in range(num_answers):
                    prompt = REFLECT_PROMPT_TEMPLATE.format(
                        question=conversation_texts[q_idx],
                        baseline_answer=answers_per_query[q_idx][baseline_idx],
                        rubrics=rubrics_per_query[q_idx][r_idx],
                    )
                    reflect_prompts.append(prompt)
                    reflect_mapping.append((q_idx, r_idx, baseline_idx))

        reflect_enabled = bool(adv_cfg.get("enable_reflect", True))
        reflect_enabled_this_step = reflect_enabled and should_update_rubrics
        reflect_batch = None
        reflect_ids = None

        if reflect_prompts and reflect_enabled_this_step:
            if answer_aware_enabled:
                self.actor_rollout_wg.set_adapter(POLICY_LLM_ADAPTER)
            reflect_batch = self._prepare_multiple_questions_batch(
                reflect_prompts,
                config,
                use_diverse_sampling=False,
                add_system_prompt=True,
                system_prompt=REFLECT_SYSTEM_PROMPT,
                max_prompt_length_override=3072,
                temperature_override=float(adv_cfg.get("reflect_generation_temperature", 0.0)),
            )

        def _run_mxm_scoring() -> None:
            if not eval_tasks:
                return
            with ThreadPoolExecutor(max_workers=min(max_concurrent_api_calls, len(eval_tasks))) as executor:
                future_to_task = {
                    executor.submit(
                        evaluator.batch_evaluate_multiple_answers,
                        answers=answers_per_query[q_idx],
                        rubric_descriptions=rubric_desc,
                        question=conversation_texts[q_idx],
                        rubric_points=rubric_points,
                    ): (q_idx, r_idx, positive_sum, negative_sum)
                    for q_idx, r_idx, rubric_desc, rubric_points, positive_sum, negative_sum in eval_tasks
                }
                for future in as_completed(future_to_task):
                    q_idx, r_idx, positive_sum, negative_sum = future_to_task[future]
                    try:
                        scored = future.result()
                        for a_idx in range(min(num_answers, len(scored))):
                            total_score = scored[a_idx].get("total_score", 0.0)
                            score_matrices[q_idx][r_idx, a_idx] = normalize_score_to_minus1_1(
                                total_score, positive_sum, negative_sum
                            )
                    except Exception as e:
                        print(f"[Unified] MxM scoring failed q={q_idx}, r={r_idx}: {e}")

        parallel_mxm_reflect = bool(adv_cfg.get("parallel_mxm_reflect", True))
        if eval_tasks and reflect_batch is not None and parallel_mxm_reflect:
            with ThreadPoolExecutor(max_workers=2) as executor:
                mxm_future = executor.submit(_run_mxm_scoring)
                reflect_future = executor.submit(self.actor_rollout_wg.generate_sequences, reflect_batch)
                mxm_future.result()
                try:
                    reflect_out = reflect_future.result()
                    reflect_ids = reflect_out.batch["responses"]
                except Exception as e:
                    print(f"[Unified] Reflect generation failed: {e}")
        elif eval_tasks:
            _run_mxm_scoring()
        elif reflect_batch is not None:
            try:
                reflect_out = self.actor_rollout_wg.generate_sequences(reflect_batch)
                reflect_ids = reflect_out.batch["responses"]
            except Exception as e:
                print(f"[Unified] Reflect generation failed: {e}")
        metrics["mxm_api_calls"] = float(len(eval_tasks))

        # Step D: Policy rewards from column-wise mean
        policy_rewards: List[float] = []
        has_any_valid_policy_signal = False
        no_valid_rubrics_query_count = 0
        policy_train_q_indices: List[int] = []
        for q_idx in range(batch_size):
            valid_rows_mask = ~parse_failures[q_idx]
            if np.any(valid_rows_mask):
                has_any_valid_policy_signal = True
                col_means = np.mean(score_matrices[q_idx][valid_rows_mask, :], axis=0)
                policy_rewards.extend(col_means.tolist())
                policy_train_q_indices.append(q_idx)
            else:
                no_valid_rubrics_query_count += 1
        metrics["policy_reward_mean"] = float(np.mean(policy_rewards)) if policy_rewards else 0.0
        metrics["policy_no_valid_rubrics_query_count"] = float(no_valid_rubrics_query_count)

        rubrics_rewards: List[float] = []
        reflect_improvements = {(q, r): 0.0 for q in range(batch_size) for r in range(num_rubrics)}
        reflect_improvement_values = {(q, r): [] for q in range(batch_size) for r in range(num_rubrics)}
        reflect_improvement_triplet = {}
        reflect_outputs_by_mapping: List[str] = []
        metrics["reflect_api_calls"] = 0.0

        reflect_use_golden_rubrics = bool(adv_cfg.get("reflect_use_golden_rubrics", False))
        ground_truth_rubrics_list_local: List[str] = []
        if reflect_use_golden_rubrics and batch.non_tensor_batch is not None:
            ground_truth_rubrics_list_local = [
                str(x) if x is not None else ""
                for x in batch.non_tensor_batch.get("ground_truth_rubrics", [""] * batch_size)
            ]

        golden_rubrics_items_per_query: List[Optional[List[Any]]] = [None] * batch_size
        golden_score_norm_sums_per_query: List[Tuple[float, float]] = [(0.0, 0.0)] * batch_size
        golden_baseline_scores: List[List[float]] = [
            [0.0 for _ in range(num_answers)] for _ in range(batch_size)
        ]
        if (
            should_update_rubrics
            and reflect_use_golden_rubrics
            and reflect_enabled_this_step
        ):
            for q_idx in range(batch_size):
                gt_text = (
                    ground_truth_rubrics_list_local[q_idx]
                    if q_idx < len(ground_truth_rubrics_list_local)
                    else ""
                )
                if not gt_text or not gt_text.strip():
                    continue
                gitems = parse_rubrics(gt_text)
                if not gitems:
                    continue
                golden_rubrics_items_per_query[q_idx] = gitems
                ps = sum(r.points for r in gitems if r.points > 0)
                ns = sum(r.points for r in gitems if r.points < 0)
                golden_score_norm_sums_per_query[q_idx] = (float(ps), float(ns))

            baseline_scoring_tasks = []
            for q_idx in range(batch_size):
                gitems = golden_rubrics_items_per_query[q_idx]
                if gitems is None:
                    continue
                for b_idx in range(num_answers):
                    if b_idx >= len(answers_per_query[q_idx]):
                        continue
                    baseline_scoring_tasks.append(
                        (q_idx, b_idx, answers_per_query[q_idx][b_idx], gitems)
                    )

            if baseline_scoring_tasks:
                with ThreadPoolExecutor(
                    max_workers=min(max_concurrent_api_calls, len(baseline_scoring_tasks))
                ) as executor:
                    baseline_future_to_task = {
                        executor.submit(
                            score_answer_with_rubrics,
                            answer=ans_text,
                            rubrics=gitems,
                            llm_evaluator=evaluator,
                            question=conversation_texts[q_idx],
                        ): (q_idx, b_idx)
                        for q_idx, b_idx, ans_text, gitems in baseline_scoring_tasks
                    }
                    for future in as_completed(baseline_future_to_task):
                        q_idx, b_idx = baseline_future_to_task[future]
                        try:
                            total_score, _ = future.result()
                        except Exception:
                            total_score = 0.0
                        ps, ns = golden_score_norm_sums_per_query[q_idx]
                        golden_baseline_scores[q_idx][b_idx] = normalize_score_to_minus1_1(
                            total_score, ps, ns
                        )
                metrics["reflect_golden_baseline_api_calls"] = float(len(baseline_scoring_tasks))
            else:
                metrics["reflect_golden_baseline_api_calls"] = 0.0
        else:
            metrics["reflect_golden_baseline_api_calls"] = 0.0

        enable_similarity = bool(adv_cfg.get("enable_similarity", False))
        similarity_weight = float(adv_cfg.get("similarity_weight", 0.0))
        use_similarity = enable_similarity and similarity_weight > 0.0 and should_update_rubrics
        similarity_per_query: List[List[float]] = [
            [0.0 for _ in range(num_rubrics)] for _ in range(batch_size)
        ]
        sim_api_calls = 0
        similarity_scoring_completed = False

        def _run_similarity_scoring() -> None:
            nonlocal sim_api_calls
            if not use_similarity:
                return

            gt_rubrics_list_for_sim: List[str] = []
            if batch.non_tensor_batch is not None:
                gt_rubrics_list_for_sim = [
                    str(x) if x is not None else ""
                    for x in batch.non_tensor_batch.get(
                        "ground_truth_rubrics", [""] * batch_size
                    )
                ]

            sim_tasks = []
            for q_idx in range(batch_size):
                gt_text = (
                    gt_rubrics_list_for_sim[q_idx]
                    if q_idx < len(gt_rubrics_list_for_sim)
                    else ""
                )
                if not gt_text or not gt_text.strip():
                    continue
                rubrics_texts_for_q = [
                    rubrics_per_query[q_idx][r_idx]
                    if not parse_failures[q_idx, r_idx]
                    else ""
                    for r_idx in range(num_rubrics)
                ]
                sim_tasks.append((q_idx, gt_text, rubrics_texts_for_q, conversation_texts[q_idx]))

            if not sim_tasks:
                return

            with ThreadPoolExecutor(
                max_workers=min(max_concurrent_api_calls, len(sim_tasks))
            ) as executor:
                future_to_q_idx = {
                    executor.submit(
                        batch_calculate_similarity_rewards,
                        rubrics_list=rubrics_texts_for_q,
                        ground_truth_rubrics=gt_text,
                        llm_evaluator=evaluator,
                        question=question_text,
                    ): q_idx
                    for q_idx, gt_text, rubrics_texts_for_q, question_text in sim_tasks
                }
                for future in as_completed(future_to_q_idx):
                    q_idx = future_to_q_idx[future]
                    try:
                        sim_scores = future.result()
                    except Exception as e:
                        print(f"[Unified] similarity reward calc failed for q={q_idx}: {e}")
                        sim_scores = [0.0] * num_rubrics
                    for r_idx in range(num_rubrics):
                        if r_idx < len(sim_scores):
                            similarity_per_query[q_idx][r_idx] = float(sim_scores[r_idx])

            sim_api_calls = len(sim_tasks)

        if should_update_rubrics:
            if reflect_prompts and reflect_enabled_this_step:
                if reflect_ids is None and reflect_batch is not None:
                    try:
                        reflect_out = self.actor_rollout_wg.generate_sequences(reflect_batch)
                        reflect_ids = reflect_out.batch["responses"]
                    except Exception as e:
                        print(f"[Unified] Reflect generation fallback failed: {e}")
                        reflect_ids = None

                if reflect_ids is not None:
                    reflect_eval_tasks = []
                    for i, (q_idx, r_idx, baseline_idx) in enumerate(reflect_mapping):
                        if i >= reflect_ids.size(0):
                            reflect_outputs_by_mapping.append("")
                            continue
                        revised = self._safe_decode_token_ids(reflect_ids[i], sequence_idx=i)
                        reflect_outputs_by_mapping.append(revised)

                        if reflect_use_golden_rubrics:
                            rubrics_items = golden_rubrics_items_per_query[q_idx]
                            if not rubrics_items:
                                continue
                            positive_sum, negative_sum = golden_score_norm_sums_per_query[q_idx]
                        else:
                            rubrics_items = parsed_rubrics_cache[q_idx][r_idx]
                            if rubrics_items is None:
                                rubrics_items = parse_rubrics(rubrics_per_query[q_idx][r_idx])
                                parsed_rubrics_cache[q_idx][r_idx] = rubrics_items
                            if not rubrics_items:
                                continue
                            positive_sum, negative_sum = score_norm_sums[q_idx][r_idx]

                        reflect_eval_tasks.append(
                            (q_idx, r_idx, baseline_idx, revised, rubrics_items, positive_sum, negative_sum)
                        )

                    if reflect_eval_tasks:
                        def _run_reflect_eval(tasks: List[Tuple[int, int, int, str, List[Any], float, float]]) -> None:
                            with ThreadPoolExecutor(
                                max_workers=min(max_concurrent_api_calls, len(tasks))
                            ) as executor:
                                future_to_task = {
                                    executor.submit(
                                        score_answer_with_rubrics,
                                        answer=revised,
                                        rubrics=rubrics_items,
                                        llm_evaluator=evaluator,
                                        question=conversation_texts[q_idx],
                                    ): (q_idx, r_idx, baseline_idx, positive_sum, negative_sum)
                                    for q_idx, r_idx, baseline_idx, revised, rubrics_items, positive_sum, negative_sum in tasks
                                }
                                for future in as_completed(future_to_task):
                                    q_idx, r_idx, baseline_idx, positive_sum, negative_sum = future_to_task[future]
                                    try:
                                        total_score, _ = future.result()
                                    except Exception:
                                        total_score, _ = (0.0, {})
                                    revised_score = normalize_score_to_minus1_1(
                                        total_score, positive_sum, negative_sum
                                    )
                                    if reflect_use_golden_rubrics:
                                        baseline_score = golden_baseline_scores[q_idx][baseline_idx]
                                    else:
                                        baseline_score = score_matrices[q_idx][r_idx, baseline_idx]
                                    improvement = float(revised_score - baseline_score)
                                    reflect_improvement_values[(q_idx, r_idx)].append(improvement)
                                    reflect_improvement_triplet[(q_idx, r_idx, baseline_idx)] = improvement
                            metrics["reflect_api_calls"] = float(len(tasks))

                        if use_similarity:
                            with ThreadPoolExecutor(max_workers=2) as outer_executor:
                                reflect_future = outer_executor.submit(_run_reflect_eval, reflect_eval_tasks)
                                similarity_future = outer_executor.submit(_run_similarity_scoring)
                                reflect_future.result()
                                similarity_future.result()
                            similarity_scoring_completed = True
                        else:
                            _run_reflect_eval(reflect_eval_tasks)

            for q_idx in range(batch_size):
                for r_idx in range(num_rubrics):
                    values = reflect_improvement_values.get((q_idx, r_idx), [])
                    reflect_improvements[(q_idx, r_idx)] = float(np.mean(values)) if values else 0.0

            if use_similarity and not similarity_scoring_completed:
                _run_similarity_scoring()
                similarity_scoring_completed = True

            disc_vals = []
            div_vals = []
            reflect_vals = []
            similarity_vals: List[float] = []
            intra_pen_vals = []
            intra_sim_vals = []
            enable_diversity = bool(adv_cfg.get("enable_diversity", True))
            diversity_weight = float(adv_cfg.get("diversity_weight", 0.25))
            disc_weight = float(adv_cfg.get("discrimination_weight", 0.25))
            reflect_weight = float(adv_cfg.get("reflect_weight", 0.25))
            similarity_penalty_weight = float(adv_cfg.get("similarity_penalty_weight", 0.0))
            similarity_penalty_threshold = float(adv_cfg.get("similarity_penalty_threshold", 0.8))
            use_diversity = enable_diversity and diversity_weight > 0.0

            metrics["similarity_reward_api_calls"] = float(sim_api_calls) if use_similarity else 0.0

            for q_idx in range(batch_size):
                if use_diversity:
                    diversity_per_rubric = batch_calculate_diversity_rewards(rubrics_per_query[q_idx])
                else:
                    diversity_per_rubric = [0.0] * num_rubrics
                if similarity_penalty_weight > 0.0:
                    intra_pen_per_rubric = batch_calculate_intra_rubric_similarity_penalties(
                        rubrics_per_query[q_idx], similarity_penalty_threshold
                    )
                else:
                    intra_pen_per_rubric = [(0.0, 0.0)] * num_rubrics
                for r_idx in range(num_rubrics):
                    if parse_failures[q_idx, r_idx]:
                        rubrics_rewards.append(-1.0)
                        continue
                    disc = float(np.std(score_matrices[q_idx][r_idx, :]))
                    div = float(diversity_per_rubric[r_idx]) if r_idx < len(diversity_per_rubric) else 0.0
                    refl = float(reflect_improvements.get((q_idx, r_idx), 0.0))
                    pen_r, avg_sim_r = (
                        intra_pen_per_rubric[r_idx]
                        if r_idx < len(intra_pen_per_rubric)
                        else (0.0, 0.0)
                    )
                    intra_pen_vals.append(pen_r)
                    intra_sim_vals.append(avg_sim_r)
                    sim_val = (
                        float(similarity_per_query[q_idx][r_idx])
                        if use_similarity
                        else 0.0
                    )
                    total = (
                        disc_weight * disc
                        + diversity_weight * div
                        + reflect_weight * refl
                        + similarity_weight * sim_val
                        - similarity_penalty_weight * pen_r
                    )
                    rubrics_rewards.append(total)
                    disc_vals.append(disc)
                    div_vals.append(div)
                    reflect_vals.append(refl)
                    similarity_vals.append(sim_val)
            metrics["rubrics_reward_mean"] = float(np.mean(rubrics_rewards)) if rubrics_rewards else 0.0
            metrics["discrimination_reward_mean"] = float(np.mean(disc_vals)) if disc_vals else 0.0
            metrics["diversity_reward_mean"] = float(np.mean(div_vals)) if div_vals else 0.0
            metrics["reflect_reward_mean"] = float(np.mean(reflect_vals)) if reflect_vals else 0.0
            metrics["similarity_reward_mean"] = (
                float(np.mean(similarity_vals)) if similarity_vals else 0.0
            )
            metrics["similarity_penalty_mean"] = float(np.mean(intra_pen_vals)) if intra_pen_vals else 0.0
            metrics["intra_rubric_avg_cos_sim_mean"] = float(np.mean(intra_sim_vals)) if intra_sim_vals else 0.0
        else:
            rubrics_rewards = [0.0] * (batch_size * num_rubrics)
            metrics["rubrics_reward_mean"] = 0.0
            metrics["discrimination_reward_mean"] = 0.0
            metrics["diversity_reward_mean"] = 0.0
            metrics["reflect_reward_mean"] = 0.0
            metrics["similarity_reward_mean"] = 0.0
            metrics["similarity_penalty_mean"] = 0.0
            metrics["intra_rubric_avg_cos_sim_mean"] = 0.0
            print(
                "[message omitted]"
                f"rubrics_update_freq={rubrics_update_freq}"
            )

        # Step E.5: save unified step prompts/outputs for debugging and analysis
        try:
            if not hasattr(self, "prompt_output_dir") or self.prompt_output_dir is None:
                exp_log_dir = os.path.join(self.config.trainer.default_local_dir, "training_logs")
                self.prompt_output_dir = os.path.join(exp_log_dir, "prompts_and_outputs")
            os.makedirs(self.prompt_output_dir, exist_ok=True)

            policy_inputs = []
            if ans_gen_batch.non_tensor_batch is not None and "prompt" in ans_gen_batch.non_tensor_batch:
                raw_policy_prompts = list(ans_gen_batch.non_tensor_batch["prompt"])
                policy_inputs = [str(x) for x in raw_policy_prompts[:batch_size]]

            rubrics_inputs = []
            for q_idx in range(batch_size):
                for r_idx in range(num_rubrics):
                    flat_idx = q_idx * num_rubrics + r_idx
                    prompt_text = rubrics_prompts[flat_idx] if flat_idx < len(rubrics_prompts) else ""
                    row = {
                        "q_idx": q_idx,
                        "r_idx": r_idx,
                        "prompt": prompt_text,
                        "output": rubrics_per_query[q_idx][r_idx] if r_idx < len(rubrics_per_query[q_idx]) else "",
                        "parse_failed": bool(parse_failures[q_idx, r_idx]),
                    }
                    rubrics_inputs.append(row)

            reflect_records = []
            for i, (q_idx, r_idx, baseline_idx) in enumerate(reflect_mapping):
                record_improvement = float(
                    reflect_improvement_triplet.get(
                        (q_idx, r_idx, baseline_idx),
                        reflect_improvements.get((q_idx, r_idx), 0.0),
                    )
                )
                reflect_records.append(
                    {
                        "q_idx": q_idx,
                        "r_idx": r_idx,
                        "baseline_answer_idx": baseline_idx,
                        "prompt": (
                            str(reflect_batch.non_tensor_batch["prompt"][i])
                            if reflect_batch is not None
                            and reflect_batch.non_tensor_batch is not None
                            and "prompt" in reflect_batch.non_tensor_batch
                            and i < len(reflect_batch.non_tensor_batch["prompt"])
                            else (reflect_prompts[i] if i < len(reflect_prompts) else "")
                        ),
                        "answer": reflect_outputs_by_mapping[i] if i < len(reflect_outputs_by_mapping) else "",
                        "improvement": record_improvement,
                    }
                )

            unified_log_data = {
                "step": int(self.global_steps),
                "iteration_idx": int(iteration_idx),
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "policy": {
                    "inputs": policy_inputs,
                    "answers_per_query": answers_per_query,
                },
                "rubrics_generator": {
                    "inputs_outputs": rubrics_inputs,
                },
                "reflect": {
                    "prompts_answers": reflect_records,
                },
            }

            unified_filename = f"step_{self.global_steps:06d}_iter_{iteration_idx:02d}_unified.json"
            unified_filepath = os.path.join(self.prompt_output_dir, unified_filename)
            with open(unified_filepath, "w", encoding="utf-8") as f:
                json.dump(unified_log_data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print("[message omitted]")

        # Step F: Policy update
        policy_update_freq = max(1, int(adv_cfg.get("policy_update_freq", 1)))
        should_update_policy = (self.global_steps % policy_update_freq) == 0
        policy_grad_accum_steps = max(1, int(adv_cfg.get("policy_grad_accum_steps", 1)))
        rubrics_for_policy = [rubrics_per_query[i][0] if rubrics_per_query[i] else "" for i in range(batch_size)]
        if should_update_policy and has_any_valid_policy_signal:
            self.actor_rollout_wg.set_adapter(POLICY_LLM_ADAPTER)
            policy_full_batch = self._prepare_full_batch_for_training_batch(
                ans_gen_batch,
                ans_gen_out,
                num_answers,
                batch_size,
                "policy_llm",
                questions,
                rubrics_for_policy,
                conversation_texts=conversation_texts,
                include_question_indices=policy_train_q_indices,
            )
            policy_reward_tensor = self._build_token_level_reward_tensor(policy_full_batch, policy_rewards)
            policy_full_batch.batch["token_level_scores"] = policy_reward_tensor
            policy_full_batch.batch["token_level_rewards"] = policy_reward_tensor
            old_log_prob = self.actor_rollout_wg.compute_log_prob(policy_full_batch)
            policy_full_batch = policy_full_batch.union(old_log_prob)
            if config.actor_rollout_ref.actor.use_kl_loss:
                ref_log_prob = self.actor_rollout_wg.compute_ref_log_prob(policy_full_batch)
                policy_full_batch = policy_full_batch.union(ref_log_prob)
            policy_full_batch = compute_advantage(
                policy_full_batch,
                adv_estimator=config.algorithm.adv_estimator,
                gamma=config.algorithm.gamma,
                lam=config.algorithm.lam,
                num_repeat=num_answers * batch_size,
                norm_adv_by_std_in_grpo=config.algorithm.get("norm_adv_by_std_in_grpo", True),
                config=config.algorithm,
            )
            self._policy_accum_batches.append(policy_full_batch)
            pending_policy_batches = len(self._policy_accum_batches)
            metrics["policy/accum_target_steps"] = float(policy_grad_accum_steps)
            metrics["policy/accum_pending_steps"] = float(pending_policy_batches)
            if pending_policy_batches >= policy_grad_accum_steps:
                merged_policy_batch = self._merge_training_batches(
                    self._policy_accum_batches,
                    uid_prefix=f"policy_accum_step{self.global_steps}",
                )
                self._policy_accum_batches = []
                policy_actor_output = self.actor_rollout_wg.update_actor(merged_policy_batch)
                policy_actor_metrics = reduce_metrics(policy_actor_output.meta_info["metrics"])
                for k, v in policy_actor_metrics.items():
                    metrics[f"policy/{k}"] = v
                metrics["policy/update_skipped_waiting_accum"] = 0.0
                metrics["policy/accum_pending_steps"] = 0.0
            else:
                metrics["policy/update_skipped_waiting_accum"] = 1.0
                print(
                    "[message omitted]"
                    ""
                )
            metrics["policy/update_skipped_no_valid_rubrics"] = 0.0
            metrics["policy/update_skipped_by_freq"] = 0.0
        elif not should_update_policy:
            metrics["policy/update_skipped_by_freq"] = 1.0
            metrics["policy/update_skipped_no_valid_rubrics"] = 0.0
            metrics["policy/update_skipped_waiting_accum"] = 0.0
            metrics["policy/accum_target_steps"] = float(policy_grad_accum_steps)
            metrics["policy/accum_pending_steps"] = float(len(self._policy_accum_batches))
            print(
                "[message omitted]"
                f"policy_update_freq={policy_update_freq}"
            )
        else:
            metrics["policy/update_skipped_by_freq"] = 0.0
            metrics["policy/update_skipped_no_valid_rubrics"] = 1.0
            metrics["policy/update_skipped_waiting_accum"] = 0.0
            metrics["policy/accum_target_steps"] = float(policy_grad_accum_steps)
            metrics["policy/accum_pending_steps"] = float(len(self._policy_accum_batches))
            print("[message omitted]")

        rubrics_grad_accum_steps = max(1, int(adv_cfg.get("rubrics_grad_accum_steps", 1)))
        if should_update_rubrics:
            self.actor_rollout_wg.set_adapter(RUBRICS_GEN_ADAPTER)
            rubrics_full_batch = self._prepare_full_batch_for_training_batch(
                rubrics_batch,
                rubrics_out,
                num_rubrics,
                batch_size,
                "rubrics_generator",
                questions,
                rubrics_for_policy,
                conversation_texts=conversation_texts,
            )
            rubrics_reward_tensor = self._build_token_level_reward_tensor(rubrics_full_batch, rubrics_rewards)
            rubrics_full_batch.batch["token_level_scores"] = rubrics_reward_tensor
            rubrics_full_batch.batch["token_level_rewards"] = rubrics_reward_tensor
            old_log_prob = self.actor_rollout_wg.compute_log_prob(rubrics_full_batch)
            rubrics_full_batch = rubrics_full_batch.union(old_log_prob)
            if config.actor_rollout_ref.actor.use_kl_loss:
                ref_log_prob = self.actor_rollout_wg.compute_ref_log_prob(rubrics_full_batch)
                rubrics_full_batch = rubrics_full_batch.union(ref_log_prob)
            rubrics_full_batch = compute_advantage(
                rubrics_full_batch,
                adv_estimator=config.algorithm.adv_estimator,
                gamma=config.algorithm.gamma,
                lam=config.algorithm.lam,
                num_repeat=num_rubrics * batch_size,
                norm_adv_by_std_in_grpo=config.algorithm.get("norm_adv_by_std_in_grpo", True),
                config=config.algorithm,
            )
            self._rubrics_accum_batches.append(rubrics_full_batch)
            pending_rubrics_batches = len(self._rubrics_accum_batches)
            metrics["rubrics/accum_target_steps"] = float(rubrics_grad_accum_steps)
            metrics["rubrics/accum_pending_steps"] = float(pending_rubrics_batches)
            if pending_rubrics_batches >= rubrics_grad_accum_steps:
                merged_rubrics_batch = self._merge_training_batches(
                    self._rubrics_accum_batches,
                    uid_prefix=f"rubrics_accum_step{self.global_steps}",
                )
                self._rubrics_accum_batches = []
                rubrics_actor_output = self.actor_rollout_wg.update_actor(merged_rubrics_batch)
                rubrics_actor_metrics = reduce_metrics(rubrics_actor_output.meta_info["metrics"])
                for k, v in rubrics_actor_metrics.items():
                    metrics[f"rubrics/{k}"] = v
                metrics["rubrics/update_skipped_waiting_accum"] = 0.0
                metrics["rubrics/accum_pending_steps"] = 0.0
            else:
                metrics["rubrics/update_skipped_waiting_accum"] = 1.0
                print(
                    "[message omitted]"
                    ""
                )
            metrics["rubrics/update_skipped_by_freq"] = 0.0
        else:
            metrics["rubrics/update_skipped_by_freq"] = 1.0
            metrics["rubrics/update_skipped_waiting_accum"] = 0.0
            metrics["rubrics/accum_target_steps"] = float(rubrics_grad_accum_steps)
            metrics["rubrics/accum_pending_steps"] = float(len(self._rubrics_accum_batches))
            print(
                "[message omitted]"
                f"rubrics_update_freq={rubrics_update_freq}"
            )

        metrics["parse_failure_count"] = float(parse_failures.sum())
        metrics["iteration_idx"] = float(iteration_idx)
        return metrics
    
    def _create_dataproto(self, batch_dict) -> DataProto:
        """English docstring."""
        tensors = {}
        non_tensors = {}
        
        for key, val in batch_dict.items():
            if isinstance(val, torch.Tensor):
                tensors[key] = val
            elif isinstance(val, np.ndarray):
                non_tensors[key] = val
            elif isinstance(val, (list, tuple)):
                non_tensors[key] = np.array(val, dtype=object)
            else:
                non_tensors[key] = np.array([val], dtype=object)
        
        batch = DataProto.from_dict(tensors=tensors, non_tensors=non_tensors)
        
        if batch.batch is not None and hasattr(batch.batch, 'batch_size') and len(batch.batch.batch_size) > 0:
            uid_len = batch.batch.batch_size[0]
        elif batch.non_tensor_batch is not None and len(batch.non_tensor_batch) > 0:
            first_key = next(iter(batch.non_tensor_batch.keys()))
            uid_len = len(batch.non_tensor_batch[first_key])
        else:
            uid_len = 1
        
        batch.non_tensor_batch["uid"] = np.array(
            [str(uuid.uuid4()) for _ in range(uid_len)], dtype=object
        )
        
        return batch
    
    def _save_prompts_and_outputs(
        self,
        step: int,
        model_type: str,  # "policy_llm" or "rubrics_generator"
        prompts: list,
        outputs: list,
        questions: list = None,
        metadata: dict = None
    ):
        """English docstring."""
        import json
        
        if not hasattr(self, 'prompt_output_dir') or self.prompt_output_dir is None:
            if hasattr(self, 'config') and self.config is not None:
                exp_log_dir = os.path.join(self.config.trainer.default_local_dir, "training_logs")
            else:
                exp_log_dir = os.path.join(project_root, "training_logs")
            self.prompt_output_dir = os.path.join(exp_log_dir, "prompts_and_outputs")
        
        try:
            os.makedirs(self.prompt_output_dir, exist_ok=True)
            if not os.path.isdir(self.prompt_output_dir):
                raise OSError("")
        except Exception as e:
            print("[message omitted]")
            fallback_dir = os.path.join(os.getcwd(), "training_logs", "prompts_and_outputs")
            try:
                os.makedirs(fallback_dir, exist_ok=True)
                self.prompt_output_dir = fallback_dir
                print("[message omitted]")
            except Exception as e2:
                print("[message omitted]")
                raise
        
        save_data = {
            "step": step,
            "model_type": model_type,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "data": []
        }
        
        if questions is not None:
            if isinstance(questions, np.ndarray):
                questions = questions.tolist()
            elif not isinstance(questions, list):
                try:
                    questions = list(questions)
                except (TypeError, ValueError):
                    questions = [str(questions)]
            batch_size = len(questions) if questions else 1
        else:
            batch_size = 1
            questions = []
        
        if len(outputs) > 0 and batch_size > 0:
            num_outputs_per_question = len(outputs) // batch_size
        else:
            num_outputs_per_question = 0
        
        expected_prompts = batch_size * num_outputs_per_question
        if len(prompts) < expected_prompts:
            if len(prompts) > 0:
                prompts.extend([prompts[-1]] * (expected_prompts - len(prompts)))
            else:
                prompts = [""] * expected_prompts
        
        for q_idx in range(batch_size):
            if questions is not None and len(questions) > 0 and q_idx < len(questions):
                question = questions[q_idx]
                if isinstance(question, np.str_) or (
                    isinstance(question, np.generic) and getattr(question.dtype, "kind", None) == "U"
                ):
                    question = str(question)
                elif isinstance(question, np.generic) and getattr(question.dtype, "kind", None) == "S":
                    question = question.item().decode("utf-8", errors="replace")
                elif not isinstance(question, str):
                    question = str(question)
            else:
                question = f"question_{q_idx}"
            
            start_idx = q_idx * num_outputs_per_question
            end_idx = start_idx + num_outputs_per_question
            
            question_prompts = prompts[start_idx:end_idx] if start_idx < len(prompts) else []
            question_outputs = outputs[start_idx:end_idx] if start_idx < len(outputs) else []
            
            min_len = min(len(question_prompts), len(question_outputs))
            if min_len > 0:
                question_prompts = question_prompts[:min_len]
                question_outputs = question_outputs[:min_len]
            
            question_data = {
                "question_idx": q_idx,
                "question": question,
                "prompts": question_prompts,
                "outputs": question_outputs,
                "num_samples": len(question_outputs)
            }
            
            if metadata:
                question_data["metadata"] = metadata
            
            save_data["data"].append(question_data)
        
        filename = f"step_{step:06d}_{model_type}.json"
        filepath = os.path.join(self.prompt_output_dir, filename)
        
        try:
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(save_data, f, ensure_ascii=False, indent=2)
            print("[message omitted]")
        except Exception as e:
            print("[message omitted]")
            import traceback
            traceback.print_exc()
    
    def _train_policy_llm_step(
        self,
        batch: DataProto,
        questions: list,
        ground_truth_rubrics_list: list,
        num_rubrics: int,
        num_answers: int,
        config
    ) -> Dict[str, float]:
        """English docstring."""
        import time
        metrics = {}
        batch_size = len(questions)

        adv_cfg = config.adversarial
        answer_aware_num_refs = int(adv_cfg.get("answer_aware_num_refs", 2))
        answer_aware_enabled = answer_aware_num_refs > 0

        prompt_messages_json_list = None
        instruction_list = None
        if batch.non_tensor_batch is not None:
            if "prompt_messages_json" in batch.non_tensor_batch:
                prompt_messages_json_list = list(batch.non_tensor_batch["prompt_messages_json"])
            if "instruction" in batch.non_tensor_batch:
                instruction_list = list(batch.non_tensor_batch["instruction"])

        conversation_texts = [
            self._format_conversation_text(prompt_messages_json_list, q_idx, questions)
            for q_idx in range(batch_size)
        ]

        gen_batch = None
        gen_batch_output = None

        if answer_aware_enabled:
            print("[message omitted]")
            t_start = time.time()
            self.actor_rollout_wg.set_adapter(POLICY_LLM_ADAPTER)
            t_adapter_switch_ans = time.time() - t_start

            print("[message omitted]")
            gen_batch = self._prepare_answer_generation_batch(batch, num_answers, config, use_diverse_sampling=False)
            t_start = time.time()
            gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)
            t_answers_gen = time.time() - t_start

            print("[message omitted]")
            print("[message omitted]")

            all_generated_answers = []
            if gen_batch_output.batch is not None and "responses" in gen_batch_output.batch:
                response_ids = gen_batch_output.batch["responses"]
                expected_total = batch_size * num_answers
                actual_num_responses = response_ids.size(0)
                if actual_num_responses < expected_total:
                    raise RuntimeError(
                        ""
                        ""
                        ""
                    )
                if actual_num_responses > expected_total:
                    print(
                        ""
                    )
                for i in range(min(expected_total, response_ids.size(0))):
                    all_generated_answers.append(self._safe_decode_token_ids(response_ids[i], sequence_idx=i))
            else:
                all_generated_answers = []

            answers_per_query = []
            for q_idx in range(batch_size):
                start = q_idx * num_answers
                answers_per_query.append(all_generated_answers[start : start + num_answers])

            print("[message omitted]")
            t_start = time.time()
            self.actor_rollout_wg.set_adapter(RUBRICS_GEN_ADAPTER)
            t_adapter_switch = time.time() - t_start

            rubrics_batch = self._prepare_rubrics_generation_batch_multi(
                questions,
                1,
                config,
                prompt_messages_json_list=prompt_messages_json_list,
                instruction_list=instruction_list,
                exemplar_data=self.previous_best_responses if self._exemplar_trick_enabled(config) else None,
                answers_per_query=answers_per_query,
            )

            if rubrics_batch.batch is not None and "input_ids" in rubrics_batch.batch:
                print(f"[1.1] rubrics_batch input_ids shape: {rubrics_batch.batch['input_ids'].shape}")
            else:
                print(f"[1.1] rubrics_batch.batch is None or missing input_ids")

            t_start = time.time()
            rubrics_output = self.actor_rollout_wg.generate_sequences(rubrics_batch)
            t_rubrics_gen = time.time() - t_start

            print("[message omitted]")
            print("[message omitted]")
            print("[message omitted]")

            if rubrics_output.batch is not None and "responses" in rubrics_output.batch:
                response_ids = rubrics_output.batch["responses"]
                actual_num_responses = response_ids.size(0)
                n_gpus = config.trainer.n_gpus_per_node
                expected_padded_size = ((batch_size + n_gpus - 1) // n_gpus) * n_gpus
            
                if actual_num_responses < batch_size:
                    raise RuntimeError(
                        ""
                        ""
                        ""
                    )
            
                if actual_num_responses != expected_padded_size:
                    print("[message omitted]")
            
                all_generated_rubrics = []
                for q_idx in range(batch_size):
                    if q_idx < response_ids.size(0):
                        response_text = self._safe_decode_token_ids(response_ids[q_idx], sequence_idx=q_idx)
                        all_generated_rubrics.append(response_text)
                    else:
                        all_generated_rubrics.append("")
            else:
                all_generated_rubrics = [""] * batch_size

            print("[message omitted]")
            del rubrics_batch, rubrics_output

            self.log_file.write("")
            for q_idx, r in enumerate(all_generated_rubrics):
                self.log_file.write("")
            self.log_file.flush()

        else:
            print("[message omitted]")
            t_start = time.time()
            self.actor_rollout_wg.set_adapter(RUBRICS_GEN_ADAPTER)
            t_adapter_switch = time.time() - t_start

            print("[message omitted]")
            rubrics_batch = self._prepare_rubrics_generation_batch_multi(
                questions,
                1,
                config,
                prompt_messages_json_list=prompt_messages_json_list,
                instruction_list=instruction_list,
                exemplar_data=self.previous_best_responses if self._exemplar_trick_enabled(config) else None,
                answers_per_query=None,
            )

            if rubrics_batch.batch is not None and "input_ids" in rubrics_batch.batch:
                print(f"[1.1] rubrics_batch input_ids shape: {rubrics_batch.batch['input_ids'].shape}")
            else:
                print(f"[1.1] rubrics_batch.batch is None or missing input_ids")

            t_start = time.time()
            rubrics_output = self.actor_rollout_wg.generate_sequences(rubrics_batch)
            t_rubrics_gen = time.time() - t_start

            print("[message omitted]")
            print("[message omitted]")
            print("[message omitted]")

            if rubrics_output.batch is not None and "responses" in rubrics_output.batch:
                response_ids = rubrics_output.batch["responses"]
                actual_num_responses = response_ids.size(0)
                n_gpus = config.trainer.n_gpus_per_node
                expected_padded_size = ((batch_size + n_gpus - 1) // n_gpus) * n_gpus

                if actual_num_responses < batch_size:
                    raise RuntimeError(
                        ""
                        ""
                        ""
                    )

                if actual_num_responses != expected_padded_size:
                    print("[message omitted]")

                all_generated_rubrics = []
                for q_idx in range(batch_size):
                    if q_idx < response_ids.size(0):
                        response_text = self._safe_decode_token_ids(response_ids[q_idx], sequence_idx=q_idx)
                        all_generated_rubrics.append(response_text)
                    else:
                        all_generated_rubrics.append("")
            else:
                all_generated_rubrics = [""] * batch_size

            print("[message omitted]")

            del rubrics_batch, rubrics_output

            self.log_file.write("")
            for q_idx, r in enumerate(all_generated_rubrics):
                self.log_file.write("")
            self.log_file.flush()

            print("[message omitted]")
            t_start = time.time()
            self.actor_rollout_wg.set_adapter(POLICY_LLM_ADAPTER)
            t_adapter_switch2 = time.time() - t_start

            print("[message omitted]")
            gen_batch = self._prepare_answer_generation_batch(batch, num_answers, config, use_diverse_sampling=False)

            t_start = time.time()
            gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)
            t_answers_gen = time.time() - t_start

            print("[message omitted]")
            print("[message omitted]")
            print("[message omitted]")

            if gen_batch_output.batch is not None and "responses" in gen_batch_output.batch:
                response_ids = gen_batch_output.batch["responses"]
                all_generated_answers = []
                expected_total = batch_size * num_answers
                actual_num_responses = response_ids.size(0)

                if actual_num_responses < expected_total:
                    raise RuntimeError(
                        ""
                        ""
                        ""
                    )
                if actual_num_responses > expected_total:
                    print(
                        ""
                    )

                for i in range(min(expected_total, response_ids.size(0))):
                    response_text = self._safe_decode_token_ids(response_ids[i], sequence_idx=i)
                    all_generated_answers.append(response_text)
            else:
                all_generated_answers = []

        print("[message omitted]")
        
        self.log_file.write("")
        for q_idx in range(batch_size):
            self.log_file.write("")
            start_idx = q_idx * num_answers
            end_idx = start_idx + num_answers
            for ans_idx, ans in enumerate(all_generated_answers[start_idx:end_idx]):
                self.log_file.write(f"  [Answer {ans_idx+1}]: {ans[:300]}...\n")
        self.log_file.flush()
        
        try:
            policy_prompts = []
            if gen_batch.non_tensor_batch is not None and "prompt" in gen_batch.non_tensor_batch:
                policy_prompts = list(gen_batch.non_tensor_batch["prompt"])
                if len(policy_prompts) > batch_size:
                    policy_prompts = policy_prompts[:batch_size]
                policy_prompts_expanded = []
                for prompt in policy_prompts:
                    policy_prompts_expanded.extend([prompt] * num_answers)
                policy_prompts = policy_prompts_expanded
            else:
                if gen_batch.batch is not None and "input_ids" in gen_batch.batch:
                    input_ids = gen_batch.batch["input_ids"]
                    for i in range(min(batch_size, input_ids.shape[0])):
                        prompt_text = self._safe_decode_token_ids(input_ids[i], sequence_idx=i)
                        policy_prompts.extend([prompt_text] * num_answers)
            
            self._save_prompts_and_outputs(
                step=self.global_steps,
                model_type="policy_llm",
                prompts=policy_prompts,
                outputs=all_generated_answers,
                questions=questions,
                metadata={"num_answers": num_answers, "num_rubrics": num_rubrics}
            )
        except Exception as e:
            print("[message omitted]")
            import traceback
            traceback.print_exc()
        
        print("[message omitted]")

        print("[message omitted]")
        adv_config_policy = self.config.get("adversarial", {})
        full_batch = self._prepare_full_batch_for_training_batch(
            gen_batch, gen_batch_output, num_answers, batch_size,
            "policy_llm", questions, all_generated_rubrics,
            golden_answer_list=None,
            policy_reward_config=None,
            conversation_texts=conversation_texts,
        )

        extra_info_array = None
        if full_batch.non_tensor_batch is not None:
            extra_info_array = full_batch.non_tensor_batch.get("extra_info")

        if extra_info_array is not None:
            try:
                from llm_evaluator import LLMEvaluator
                from reward_calculator import calculate_policy_llm_reward

                evaluator = LLMEvaluator()
                total_cached = 0
                total_samples = batch_size * num_answers
                print("[message omitted]")

                evaluation_tasks = []
                for q_idx in range(batch_size):
                    start_idx = q_idx * num_answers
                    rubrics_text = all_generated_rubrics[q_idx] if q_idx < len(all_generated_rubrics) else ""
                    if not rubrics_text:
                        continue
                    
                    question = questions[q_idx] if q_idx < len(questions) else ""
                    
                    for ans_offset in range(num_answers):
                        sample_idx = start_idx + ans_offset
                        if sample_idx >= len(all_generated_answers):
                            break
                        
                        answer = all_generated_answers[sample_idx]
                        if not answer:
                            continue
                        
                        evaluation_tasks.append({
                            'sample_idx': sample_idx,
                            'q_idx': q_idx,
                            'ans_offset': ans_offset,
                            'answer': answer,
                            'rubrics_text': rubrics_text,
                            'question': question
                        })

                def evaluate_single_answer(task):
                    """English docstring."""
                    try:
                        reward, _ = calculate_policy_llm_reward(
                            answer=task['answer'],
                            rubrics_list=[task['rubrics_text']],
                            llm_evaluator=evaluator,
                            question=task['question']
                        )
                        return task['sample_idx'], reward, None
                    except Exception as e:
                        print("[message omitted]")
                        return task['sample_idx'], 0.0, e

                adv_config = self.config.get("adversarial", {})
                max_concurrent_config = adv_config.get("max_concurrent_api_calls", 64)
                max_concurrent = min(len(evaluation_tasks), max_concurrent_config)
                print("[message omitted]")
                
                with ThreadPoolExecutor(max_workers=max_concurrent) as executor:
                    future_to_task = {
                        executor.submit(evaluate_single_answer, task): task
                        for task in evaluation_tasks
                    }
                    
                    for future in as_completed(future_to_task):
                        try:
                            sample_idx, reward, error = future.result()
                            if sample_idx >= len(extra_info_array):
                                continue
                            
                            extra_info = extra_info_array[sample_idx]
                            if isinstance(extra_info, dict):
                                updated_extra = extra_info.copy()
                            else:
                                updated_extra = {}
                            updated_extra["cached_reward"] = float(reward)
                            extra_info_array[sample_idx] = updated_extra
                            total_cached += 1
                            
                            if error:
                                print("[message omitted]")
                        except Exception as e:
                            task = future_to_task[future]
                            print("[message omitted]")

                print("[message omitted]")
            except Exception as e:
                print("[message omitted]")
                import traceback
                traceback.print_exc()

        print("[message omitted]")
        t_reward_start = time.time()
        reward_tensor, _ = compute_reward(full_batch, self.reward_fn)
        t_reward_total = time.time() - t_reward_start
        avg_time_per_sample = t_reward_total / len(full_batch) if len(full_batch) > 0 else 0
        print("[message omitted]")

        full_batch.batch["token_level_scores"] = reward_tensor
        full_batch.batch["token_level_rewards"] = reward_tensor

        if "attention_mask" in full_batch.batch and "prompts" in full_batch.batch:
            prompt_length = full_batch.batch["prompts"].shape[-1]
            attention_mask = full_batch.batch["attention_mask"]
            response_mask = attention_mask[:, prompt_length:]
            response_lens = response_mask.sum(dim=-1)

            all_sample_rewards = []
            for i in range(reward_tensor.size(0)):
                last_idx = int(response_lens[i].item()) - 1
                if last_idx >= 0 and last_idx < reward_tensor.size(1):
                    all_sample_rewards.append(reward_tensor[i, last_idx].item())
                else:
                    non_zero_mask = reward_tensor[i] != 0
                    if non_zero_mask.any():
                        all_sample_rewards.append(reward_tensor[i][non_zero_mask][-1].item())
                    else:
                        all_sample_rewards.append(0.0)

            for q_idx in range(batch_size):
                start_idx = q_idx * num_answers
                end_idx = start_idx + num_answers
                q_sample_rewards = all_sample_rewards[start_idx:end_idx]
                q_reward_mean = np.mean(q_sample_rewards)
                print("[message omitted]")
            
            if all_sample_rewards:
                metrics["reward_mean"] = np.mean(all_sample_rewards)
                print(f"[Metrics] Policy-LLM reward_mean = {metrics['reward_mean']:.4f} (from {len(all_sample_rewards)} samples)")
            
            if self._exemplar_trick_enabled(config):
                try:
                    self._pending_best_responses = []
                    for q_idx in range(batch_size):
                        start_idx = q_idx * num_answers
                        end_idx = start_idx + num_answers
                        q_rewards = all_sample_rewards[start_idx:end_idx]
                        q_answers = all_generated_answers[start_idx:end_idx]

                        sorted_indices = sorted(range(len(q_rewards)), key=lambda i: q_rewards[i], reverse=True)
                        top_2 = [(q_answers[idx], q_rewards[idx]) for idx in sorted_indices[:2]]

                        self._pending_best_responses.append({
                            'question': questions[q_idx],
                            'top_responses': top_2
                        })
                    print("[message omitted]")
                    for q_idx, cached in enumerate(self._pending_best_responses):
                        top_scores = [f"{s:.4f}" for _, s in cached['top_responses']]
                        print(f"  Query {q_idx+1}: top-2 scores = [{', '.join(top_scores)}]")
                except Exception as e:
                    print("[message omitted]")
                    import traceback
                    traceback.print_exc()

        print("[message omitted]")
        old_log_prob = self.actor_rollout_wg.compute_log_prob(full_batch)
        full_batch = full_batch.union(old_log_prob)

        if config.actor_rollout_ref.actor.use_kl_loss:
            print("[message omitted]")
            ref_log_prob = self.actor_rollout_wg.compute_ref_log_prob(full_batch)
            full_batch = full_batch.union(ref_log_prob)

        if full_batch.non_tensor_batch and "uid" in full_batch.non_tensor_batch:
            uids = full_batch.non_tensor_batch["uid"]
            uid_groups = {}
            for i, uid in enumerate(uids):
                if uid not in uid_groups:
                    uid_groups[uid] = []
                uid_groups[uid].append(i)
            print("[message omitted]")
            for uid, indices in sorted(uid_groups.items()):
                print("[message omitted]")
                if len(indices) != num_answers:
                    print("[message omitted]")
        
        print("[message omitted]")
        full_batch = compute_advantage(
            full_batch,
            adv_estimator=config.algorithm.adv_estimator,
            gamma=config.algorithm.gamma,
            lam=config.algorithm.lam,
            num_repeat=num_answers * batch_size,
            norm_adv_by_std_in_grpo=config.algorithm.get("norm_adv_by_std_in_grpo", True),
            config=config.algorithm,
        )

        if "attention_mask" in full_batch.batch:
            if full_batch.meta_info is None:
                full_batch.meta_info = {}
            full_batch.meta_info["global_token_num"] = torch.sum(
                full_batch.batch["attention_mask"], dim=-1
            ).tolist()

        print("[message omitted]")

        from verl.utils.memory_utils import aggressive_empty_cache

        adv_cfg_policy = self.config.get("adversarial", {})
        policy_update_freq = max(1, int(adv_cfg_policy.get("policy_update_freq", 1)))
        should_update_policy = (self.global_steps % policy_update_freq) == 0
        if should_update_policy:
            actor_output = self.actor_rollout_wg.update_actor(full_batch)
            actor_metrics = reduce_metrics(actor_output.meta_info["metrics"])
            metrics.update(actor_metrics)
            metrics["policy/update_skipped_by_freq"] = 0.0
        else:
            print(
                "[message omitted]"
                f"policy_update_freq={policy_update_freq}"
            )
            metrics["policy/update_skipped_by_freq"] = 1.0

        print("[message omitted]")
        
        del gen_batch, gen_batch_output

        print("[message omitted]")
        return metrics
    
    def _train_rubrics_generator_step_single(
        self,
        batch: DataProto,
        question: str,
        ground_truth_rubrics: str,
        num_rubrics: int,
        num_answers: int,
        config,
        golden_answer: str = "",
    ) -> Dict[str, float]:
        """English docstring."""
        import time
        metrics = {}

        conv_text = question
        if batch.non_tensor_batch is not None and "prompt_messages_json" in batch.non_tensor_batch:
            pmj = list(batch.non_tensor_batch["prompt_messages_json"])
            conv_text = self._format_conversation_text(pmj, 0, [question])

        print("[message omitted]")
        t_start = time.time()
        self.actor_rollout_wg.set_adapter(RUBRICS_GEN_ADAPTER)
        t_adapter_switch = time.time() - t_start
        
        print("[message omitted]")
        rubrics_batch = self._prepare_rubrics_generation_batch(question, num_rubrics, config)
        
        if rubrics_batch.batch is not None and 'input_ids' in rubrics_batch.batch:
            print(f"[2.1] rubrics_batch input_ids shape: {rubrics_batch.batch['input_ids'].shape}")
        else:
            print(f"[2.1] rubrics_batch.batch is None or missing input_ids")
        
        t_start = time.time()
        rubrics_output = self.actor_rollout_wg.generate_sequences(rubrics_batch)
        t_rubrics_gen = time.time() - t_start

        print("[message omitted]")
        print("[message omitted]")
        print("[message omitted]")
        
        if rubrics_output.batch is not None and "responses" in rubrics_output.batch:
            response_ids = rubrics_output.batch["responses"]
            generated_rubrics = []
            num_sequences_to_decode = min(num_rubrics, response_ids.size(0))
            pad_token_id = getattr(self.tokenizer, 'pad_token_id', None)
            for i in range(num_sequences_to_decode):
                token_ids = response_ids[i].cpu().tolist()
                valid_token_ids = [
                    tid for tid in token_ids 
                    if tid is not None and isinstance(tid, int) and tid >= 0 
                    and (pad_token_id is None or tid != pad_token_id)
                ]
                if not valid_token_ids:
                    generated_rubrics.append("")
                    continue
                response_text = self._safe_decode_token_ids(response_ids[i], sequence_idx=i)
                generated_rubrics.append(response_text)
        else:
            generated_rubrics = []
        
        print("[message omitted]")
        
        self.log_file.write("")
        for i, r in enumerate(generated_rubrics):
            self.log_file.write(f"[Rubrics {i+1}]: {r[:300]}...\n")
        self.log_file.flush()
        
        import os
        import json
        prompts_log_dir = os.path.join(self.config.trainer.default_local_dir, "training_logs", "prompts_and_outputs")
        os.makedirs(prompts_log_dir, exist_ok=True)
        
        rubrics_log_path = os.path.join(prompts_log_dir, f"step_{self.global_steps:06d}_rubrics_generator.json")
        rubrics_log_data = {
            "step": self.global_steps,
            "question": question,
            "num_rubrics": len(generated_rubrics),
            "rubrics_prompts_and_outputs": []
        }
        
        for i, rubrics_text in enumerate(generated_rubrics):
            messages = [
                {"role": "system", "content": RUBRICS_GENERATOR_SYSTEM_PROMPT},
                {"role": "user", "content": question}
            ]
            rubrics_prompt_full = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False
            )
            
            rubrics_log_data["rubrics_prompts_and_outputs"].append({
                "rubrics_idx": i,
                "question": question,
                "rubrics_prompt_full": rubrics_prompt_full,
                "generated_rubrics": rubrics_text,
                "prompt_length_tokens": len(self.tokenizer.encode(rubrics_prompt_full)),
                "response_length_tokens": len(self.tokenizer.encode(rubrics_text))
            })
        
        with open(rubrics_log_path, "w", encoding="utf-8") as f:
            json.dump(rubrics_log_data, f, ensure_ascii=False, indent=2)
        
        print("[message omitted]")
        print("[message omitted]")
        
        print("[message omitted]")
        t_start = time.time()
        self.actor_rollout_wg.set_adapter(POLICY_LLM_ADAPTER)
        t_adapter_switch2 = time.time() - t_start
        
        print("[message omitted]")
        gen_batch = self._prepare_answer_generation_batch(batch, num_answers, config, use_diverse_sampling=True)

        t_start = time.time()
        gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)
        t_answers_gen = time.time() - t_start

        print("[message omitted]")
        print("[message omitted]")
        print("[message omitted]")
        
        if gen_batch_output.batch is not None and "responses" in gen_batch_output.batch:
            response_ids = gen_batch_output.batch["responses"]
            num_answers_requested = gen_batch.meta_info.get("num_answers_requested", num_answers)
            
            generated_answers = []
            for i in range(min(num_answers_requested, response_ids.size(0))):
                response_text = self._safe_decode_token_ids(response_ids[i], sequence_idx=i)
                generated_answers.append(response_text)
        else:
            generated_answers = []
        
        print("[message omitted]")
        
        del gen_batch, gen_batch_output
        
        self.log_file.write("")
        for i, ans in enumerate(generated_answers):
            self.log_file.write(f"[Answer {i+1}]: {ans[:300]}...\n")
        self.log_file.flush()
        
        print("[message omitted]")
        adv_config = self.config.get("adversarial", {})
        
        enable_reflect_val = adv_config.get("enable_reflect")
        if enable_reflect_val is None:
            enable_reflect = False
        elif isinstance(enable_reflect_val, str):
            enable_reflect = enable_reflect_val.lower() in ("true", "1", "yes")
        else:
            enable_reflect = bool(enable_reflect_val)
        
        print(f"[2.2.5 DEBUG] enable_reflect = {enable_reflect} (type: {type(enable_reflect)}, raw: {enable_reflect_val})", file=sys.stderr, flush=True)
        
        reflect_answers_per_baseline = {}  # {baseline_idx: {rubrics_idx: revised_answer}}
        
        if enable_reflect:
            print("[message omitted]")
            
            print("[message omitted]")
            
            all_reflect_prompts = []
            reflect_prompt_mapping = []  
            
            conv_text = question
            if batch.non_tensor_batch is not None and "prompt_messages_json" in batch.non_tensor_batch:
                pmj = list(batch.non_tensor_batch["prompt_messages_json"])
                conv_text = self._format_conversation_text(pmj, 0, [question])

            for i, baseline_answer in enumerate(generated_answers):
                for k, rubrics in enumerate(generated_rubrics):
                    reflect_prompt = REFLECT_PROMPT_TEMPLATE.format(
                        question=conv_text,
                        baseline_answer=baseline_answer,
                        rubrics=rubrics
                    )
                    all_reflect_prompts.append(reflect_prompt)
                    reflect_prompt_mapping.append((i, k))
            
            print("[message omitted]")
            
            REFLECT_BATCH_SIZE = 4
            total_prompts = len(all_reflect_prompts)
            
            for i in range(len(generated_answers)):
                reflect_answers_per_baseline[i] = {}
            
            print("[message omitted]")
            
            from verl.utils.memory_utils import aggressive_empty_cache
            
            for batch_start in range(0, total_prompts, REFLECT_BATCH_SIZE):
                batch_end = min(batch_start + REFLECT_BATCH_SIZE, total_prompts)
                batch_prompts = all_reflect_prompts[batch_start:batch_end]
                
                print("[message omitted]")
                
                reflect_batch = self._prepare_multiple_questions_batch(
                    batch_prompts,
                    config,
                    use_diverse_sampling=False,
                    add_system_prompt=True,
                    system_prompt=REFLECT_SYSTEM_PROMPT,
                    max_prompt_length_override=3072,
                    temperature_override=float(
                        config.adversarial.get("reflect_generation_temperature", 0.0)
                    ),
                )
                reflect_output = self.actor_rollout_wg.generate_sequences(reflect_batch)
                
                if reflect_output.batch is not None and "responses" in reflect_output.batch:
                    response_ids = reflect_output.batch["responses"]
                    
                    for local_idx in range(min(len(batch_prompts), response_ids.size(0))):
                        global_idx = batch_start + local_idx
                        baseline_idx, rubrics_idx = reflect_prompt_mapping[global_idx]
                        revised_answer = self._safe_decode_token_ids(response_ids[local_idx], sequence_idx=local_idx)
                        reflect_answers_per_baseline[baseline_idx][rubrics_idx] = revised_answer
                else:
                    print("[message omitted]")
                    for local_idx in range(len(batch_prompts)):
                        global_idx = batch_start + local_idx
                        baseline_idx, rubrics_idx = reflect_prompt_mapping[global_idx]
                        reflect_answers_per_baseline[baseline_idx][rubrics_idx] = generated_answers[baseline_idx]
                
                del reflect_batch, reflect_output
            
            print("[message omitted]")
            
            import os
            import json
            prompts_log_dir = os.path.join(self.config.trainer.default_local_dir, "training_logs", "prompts_and_outputs")
            os.makedirs(prompts_log_dir, exist_ok=True)
            
            reflect_log_path = os.path.join(prompts_log_dir, f"step_{self.global_steps:06d}_reflect.json")
            reflect_log_data = {
                "step": self.global_steps,
                "question": question,
                "num_baseline_answers": len(generated_answers),
                "num_rubrics": len(generated_rubrics),
                "reflect_prompts_and_outputs": []
            }
            
            for i, baseline_answer in enumerate(generated_answers):
                for k, rubrics in enumerate(generated_rubrics):
                    reflect_prompt_raw = REFLECT_PROMPT_TEMPLATE.format(
                        question=conv_text,
                        baseline_answer=baseline_answer,
                        rubrics=rubrics
                    )
                    messages = [
                        {"role": "system", "content": REFLECT_SYSTEM_PROMPT},
                        {"role": "user", "content": reflect_prompt_raw}
                    ]
                    reflect_prompt_full = self.tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True,
                        enable_thinking=False
                    )
                    
                    revised_answer = reflect_answers_per_baseline[i].get(k, "")
                    
                    reflect_log_data["reflect_prompts_and_outputs"].append({
                        "baseline_idx": i,
                        "rubrics_idx": k,
                        "baseline_answer": baseline_answer,
                        "rubrics": rubrics,
                        "reflect_prompt_raw": reflect_prompt_raw,
                        "reflect_prompt_full": reflect_prompt_full,
                        "revised_answer": revised_answer,
                        "prompt_length_tokens": len(self.tokenizer.encode(reflect_prompt_full)),
                        "response_length_tokens": len(self.tokenizer.encode(revised_answer))
                    })
            
            with open(reflect_log_path, "w", encoding="utf-8") as f:
                json.dump(reflect_log_data, f, ensure_ascii=False, indent=2)
            
            print("[message omitted]")
            print("[message omitted]")
            
        print("[message omitted]")
        
        if reflect_answers_per_baseline:
            self.log_file.write(f"\n{'='*80}\n")
            self.log_file.write("")
            self.log_file.write(f"{'='*80}\n")
            for baseline_idx in range(len(generated_answers)):
                self.log_file.write("")
                self.log_file.write("")
                for rubrics_idx in range(len(generated_rubrics)):
                    if baseline_idx in reflect_answers_per_baseline and rubrics_idx in reflect_answers_per_baseline[baseline_idx]:
                        improved = reflect_answers_per_baseline[baseline_idx][rubrics_idx]
                        self.log_file.write("")
                    else:
                        self.log_file.write("")
        
        self.log_file.flush()
        
        self.actor_rollout_wg.set_adapter(RUBRICS_GEN_ADAPTER)
        
        adv_config = self.config.get("adversarial", {})
        
        enable_reflect_val = adv_config.get("enable_reflect")
        if enable_reflect_val is None:
            enable_reflect = False
        elif isinstance(enable_reflect_val, str):
            enable_reflect = enable_reflect_val.lower() in ("true", "1", "yes")
        else:
            enable_reflect = bool(enable_reflect_val)
        
        print("[message omitted]")
        print("[message omitted]")
        
        reward_config = {
            "similarity_weight": adv_config.get("similarity_weight"),
            "discrimination_weight": adv_config.get("discrimination_weight"),
            "diversity_weight": adv_config.get("diversity_weight"),
            "reflect_weight": adv_config.get("reflect_weight"),
            "similarity_penalty_weight": adv_config.get("similarity_penalty_weight", 0.0),
            "similarity_penalty_threshold": adv_config.get("similarity_penalty_threshold", 0.8),
            "enable_similarity": adv_config.get("enable_similarity"),
            "enable_discrimination": adv_config.get("enable_discrimination"),
            "enable_diversity": adv_config.get("enable_diversity"),
            "enable_reflect": enable_reflect,
        }
        
        try:
            full_batch = self._prepare_full_batch_for_rubrics_training(
                rubrics_batch, rubrics_output, num_rubrics, 
                question, ground_truth_rubrics, generated_rubrics, generated_answers,
                reward_config=reward_config,
                reflect_answers_per_baseline=reflect_answers_per_baseline if enable_reflect else None,
                question_idx=0,
                golden_answer=golden_answer,
            )
            print("[message omitted]")
        except Exception as e:
            print("[message omitted]")
            import traceback
            traceback.print_exc()
            raise
        
        self._precompute_rubrics_rewards(
            full_batch=full_batch,
            generated_rubrics=generated_rubrics,
            generated_answers=generated_answers,
            question=question,
            ground_truth_rubrics=ground_truth_rubrics,
            adv_config=adv_config,
            enable_reflect=enable_reflect,
            reflect_answers_per_baseline=reflect_answers_per_baseline if enable_reflect else None,
            golden_answer=golden_answer,
        )
        
        print("[message omitted]")
        try:
            t_reward_start = time.time()
            reward_tensor, reward_extra_infos_dict = compute_reward(full_batch, self.reward_fn)
            t_reward_total = time.time() - t_reward_start
            print("[message omitted]")
            print("[message omitted]")
            print(f"[2.3.4] reward_extra_infos_dict keys: {list(reward_extra_infos_dict.keys()) if reward_extra_infos_dict else 'None'}")
        except Exception as e:
            print("[message omitted]")
            import traceback
            traceback.print_exc()
            raise
        full_batch.batch["token_level_scores"] = reward_tensor
        full_batch.batch["token_level_rewards"] = reward_tensor
        
        if reward_extra_infos_dict:
            print("[message omitted]")
            sub_reward_keys = ['similarity_reward', 'discrimination_reward',
                              'diversity_reward', 'entropy_reward',
                              'reflect_reward']
            for key in sub_reward_keys:
                if key in reward_extra_infos_dict:
                    values = reward_extra_infos_dict[key]
                    if isinstance(values, list) and values:
                        valid_values = [float(v) for v in values if v is not None]
                        if valid_values:
                            mean_val = np.mean(valid_values)
                            metrics[f"{key}_mean"] = mean_val
                            print(f"[Metrics] {key}_mean = {mean_val:.4f} (from {len(valid_values)} samples)")
        
        if "attention_mask" in full_batch.batch and "prompts" in full_batch.batch:
            prompt_length = full_batch.batch["prompts"].shape[-1]
            attention_mask = full_batch.batch["attention_mask"]
            response_mask = attention_mask[:, prompt_length:]
            response_lens = response_mask.sum(dim=-1)
            
            sample_rewards = []
            for i in range(reward_tensor.size(0)):
                last_idx = int(response_lens[i].item()) - 1
                if last_idx >= 0 and last_idx < reward_tensor.size(1):
                    sample_rewards.append(reward_tensor[i, last_idx].item())
                else:
                    non_zero_mask = reward_tensor[i] != 0
                    if non_zero_mask.any():
                        sample_rewards.append(reward_tensor[i][non_zero_mask][-1].item())
                    else:
                        sample_rewards.append(0.0)

            metrics["reward_mean"] = np.mean(sample_rewards)
            print(f"  Rubrics Generator Rewards: mean={metrics['reward_mean']:.4f}, samples={sample_rewards}")
        

        self.actor_rollout_wg.set_adapter(RUBRICS_GEN_ADAPTER)
        print("[message omitted]")
        try:
            old_log_prob = self.actor_rollout_wg.compute_log_prob(full_batch)
            print("[message omitted]")
        except Exception as e:
            print("[message omitted]")
            import traceback
            traceback.print_exc()
            raise
        full_batch = full_batch.union(old_log_prob)
        
        if config.actor_rollout_ref.actor.use_kl_loss:
            print("[message omitted]")
            ref_log_prob = self.actor_rollout_wg.compute_ref_log_prob(full_batch)
            full_batch = full_batch.union(ref_log_prob)
        
        if full_batch.non_tensor_batch and "uid" in full_batch.non_tensor_batch:
            uids = full_batch.non_tensor_batch["uid"]
            unique_uids = set(uids)
            print("[message omitted]")
            if len(unique_uids) != 1:
                print("[message omitted]")
            else:
                uid_value = list(unique_uids)[0]
                count = list(uids).count(uid_value)
                print("[message omitted]")
                if count != num_rubrics:
                    print("[message omitted]")
        
        print("[message omitted]")
        full_batch = compute_advantage(
            full_batch,
            adv_estimator=config.algorithm.adv_estimator,
            gamma=config.algorithm.gamma,
            lam=config.algorithm.lam,
            num_repeat=num_rubrics,
            norm_adv_by_std_in_grpo=config.algorithm.get("norm_adv_by_std_in_grpo", True),
            config=config.algorithm,
        )
        
        if "attention_mask" in full_batch.batch:
            if full_batch.meta_info is None:
                full_batch.meta_info = {}
            full_batch.meta_info["global_token_num"] = torch.sum(
                full_batch.batch["attention_mask"], dim=-1
            ).tolist()
        
        print("[message omitted]")
        
        ppo_mini_batch_size = config.actor_rollout_ref.actor.ppo_mini_batch_size
        if ppo_mini_batch_size <= 0:
            raise ValueError(
                ""
                ""
            )
        
        from verl.utils.memory_utils import aggressive_empty_cache
        actor_output = self.actor_rollout_wg.update_actor(full_batch)
        actor_metrics = reduce_metrics(actor_output.meta_info["metrics"])
        metrics.update(actor_metrics)
        
        print("[message omitted]")
        return metrics
    
    def _prepare_rubrics_generation_batch_multi(
        self,
        questions: list,
        num_rubrics_per_question: int,
        config,
        temperature: Optional[float] = None,
        prompt_messages_json_list: list = None,
        instruction_list: list = None,
        exemplar_data: list = None,
        answers_per_query: Optional[List[List[str]]] = None,
    ) -> DataProto:
        """English docstring."""
        import copy
        import random

        if temperature is None:
            temperature = float(config.adversarial.get("generation_temperature", 1.0))
        
        dataset = AdversarialDataset(
            data_files=config.data.train_files,
            tokenizer=self.tokenizer,
            processor=self.processor,
            config=config.data,
            mode="rubrics_generator"
        )
        
        use_universal = config.adversarial.get("use_universal_rubrics_system_prompt", True)
        system_prompt = UNIVERSAL_RUBRICS_GENERATOR_SYSTEM_PROMPT if use_universal else dataset._get_rubrics_generator_system_prompt()
        
        exemplar_prefix = ""
        if exemplar_data and len(exemplar_data) > 0:
            try:
                exemplar = random.choice(exemplar_data)
                prev_question = exemplar.get('question', '')
                top_responses = exemplar.get('top_responses', [])
                
                if prev_question and len(top_responses) >= 2:
                    exemplar_prefix = RUBRICS_EXEMPLAR_TEMPLATE.format(
                        prev_question=prev_question,
                        top_response_1=top_responses[0][0],  # (text, score) -> text
                        top_response_2=top_responses[1][0]
                    )
                    print("[message omitted]")
                elif prev_question and len(top_responses) == 1:
                    exemplar_prefix = RUBRICS_EXEMPLAR_TEMPLATE.format(
                        prev_question=prev_question,
                        top_response_1=top_responses[0][0],
                        top_response_2="(Not available)"
                    )
                    print("[message omitted]")
            except Exception as e:
                print("[message omitted]")
                exemplar_prefix = ""

        answer_aware_num_refs = int(config.adversarial.get("answer_aware_num_refs", 2))
        use_answer_aware = answers_per_query is not None and answer_aware_num_refs > 0

        all_prompts = []
        for idx, question in enumerate(questions):
            instruction = ""
            if instruction_list is not None and idx < len(instruction_list):
                instruction = instruction_list[idx] or ""

            raw_messages = []
            if prompt_messages_json_list is not None and idx < len(prompt_messages_json_list):
                try:
                    candidate = json.loads(prompt_messages_json_list[idx])
                    if isinstance(candidate, list):
                        raw_messages = candidate
                except Exception:
                    raw_messages = []

            messages = [{"role": "system", "content": system_prompt}]
            if instruction:
                messages.append({"role": "system", "content": instruction})

            if raw_messages:
                if exemplar_prefix:
                    messages.append({"role": "user", "content": exemplar_prefix})
                for msg in raw_messages:
                    role = msg.get("role", "")
                    content = msg.get("content", "")
                    if role not in ("user", "assistant", "system") or not isinstance(content, str):
                        continue
                    messages.append({"role": role, "content": content})
            else:
                user_content = exemplar_prefix + question if exemplar_prefix else question
                messages.append({"role": "user", "content": user_content})

            q_answers = answers_per_query[idx] if idx < len(answers_per_query) else []
            if use_answer_aware and q_answers and len(q_answers) > 0:
                for _ in range(num_rubrics_per_question):
                    m = copy.deepcopy(messages)
                    self._append_answer_aware_to_last_user(m, q_answers)
                    prompt = self.tokenizer.apply_chat_template(
                        m,
                        tokenize=False,
                        add_generation_prompt=True,
                        enable_thinking=False,
                    )
                    all_prompts.append(prompt)
                if idx == 0:
                    print(
                        ""
                        ""
                        ""
                    )
            else:
                prompt = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
                all_prompts.extend([prompt] * num_rubrics_per_question)
        
        question_array = [q for q in questions for _ in range(num_rubrics_per_question)]
        
        n_gpus = config.trainer.n_gpus_per_node
        total_prompts = len(all_prompts)
        if total_prompts % n_gpus != 0:
            padded_size = ((total_prompts + n_gpus - 1) // n_gpus) * n_gpus
            all_prompts.extend([all_prompts[-1]] * (padded_size - total_prompts))
            question_array.extend([question_array[-1]] * (padded_size - len(question_array)))
        
        assert len(all_prompts) == len(question_array), \
            ""
        
        encoded = self.tokenizer(
            all_prompts,
            padding=True,
            truncation=False,
            return_tensors="pt"
        )
        
        attention_mask = encoded["attention_mask"]
        position_ids = attention_mask.long().cumsum(dim=1) - 1
        position_ids = position_ids.clamp(min=0)
        
        batch = DataProto.from_dict(
            tensors={
                "input_ids": encoded["input_ids"],
                "attention_mask": encoded["attention_mask"],
                "position_ids": position_ids
            },
            non_tensors={
                "prompt": np.array(all_prompts, dtype=object),
                "question": np.array(question_array, dtype=object)
            },
            meta_info={
                "n": 1,
                "do_sample": True,
                "temperature": temperature,
                "top_p": 0.95,
                "top_k": 50,
                "repetition_penalty": 1.0 if temperature <= 0.7 else 1.25,
                "num_rubrics_requested": len(all_prompts)
            }
        )
        
        return batch
    
    def _extract_question_batch(
        self, 
        gen_batch: DataProto, 
        gen_batch_output: DataProto, 
        question_idx: int, 
        num_answers: int,
        batch_size: int
    ) -> tuple:
        """English docstring."""
        from tensordict import TensorDict
        
        output_start_idx = question_idx * num_answers
        output_end_idx = output_start_idx + num_answers
        
        batch_start_idx = question_idx
        batch_end_idx = question_idx + 1
        
        if gen_batch.batch is None:
            q_batch = None
        else:
            q_batch = gen_batch.batch[batch_start_idx:batch_end_idx]
        
        if gen_batch_output.batch is None:
            q_batch_output = None
        else:
            q_batch_output = gen_batch_output.batch[output_start_idx:output_end_idx]
        
        q_non_tensor_batch = None
        if gen_batch.non_tensor_batch is not None:
            q_non_tensor_batch = {}
            for k, v in gen_batch.non_tensor_batch.items():
                if isinstance(v, (list, np.ndarray)) and len(v) > batch_start_idx:
                    if isinstance(v, list):
                        q_non_tensor_batch[k] = v[batch_start_idx:batch_end_idx]
                    else:  # np.ndarray
                        q_non_tensor_batch[k] = v[batch_start_idx:batch_end_idx]
                else:
                    q_non_tensor_batch[k] = v

        q_non_tensor_batch_output = None
        if gen_batch_output.non_tensor_batch is not None:
            q_non_tensor_batch_output = {}
            for k, v in gen_batch_output.non_tensor_batch.items():
                if isinstance(v, (list, np.ndarray)) and len(v) > output_start_idx:
                    if isinstance(v, list):
                        q_non_tensor_batch_output[k] = v[output_start_idx:output_end_idx]
                    else:  # np.ndarray
                        q_non_tensor_batch_output[k] = v[output_start_idx:output_end_idx]
                else:
                    q_non_tensor_batch_output[k] = v

        q_gen_batch = DataProto(
            batch=q_batch,
            non_tensor_batch=q_non_tensor_batch,
            meta_info=gen_batch.meta_info.copy() if gen_batch.meta_info is not None else None
        )

        q_gen_batch_output = DataProto(
            batch=q_batch_output,
            non_tensor_batch=q_non_tensor_batch_output,
            meta_info=gen_batch_output.meta_info.copy() if gen_batch_output.meta_info is not None else None
        )
        
        return q_gen_batch, q_gen_batch_output
    
    def _aggregate_metrics(self, metrics_list: list) -> dict:
        """English docstring."""
        if not metrics_list:
            return {}
        
        aggregated = {}
        
        all_keys = set()
        for m in metrics_list:
            all_keys.update(m.keys())
        
        for key in all_keys:
            values = [m[key] for m in metrics_list if key in m]
            if values:
                aggregated[key] = np.mean(values)
        
        return aggregated
    
    def _prepare_rubrics_generation_batch(self, question: str, num_rubrics: int, config) -> DataProto:
        """English docstring."""
        dataset = AdversarialDataset(
            data_files=config.data.train_files,
            tokenizer=self.tokenizer,
            processor=self.processor,
            config=config.data,
            mode="rubrics_generator"
        )
        
        use_universal = config.adversarial.get("use_universal_rubrics_system_prompt", True)
        system_prompt = UNIVERSAL_RUBRICS_GENERATOR_SYSTEM_PROMPT if use_universal else dataset._get_rubrics_generator_system_prompt()
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question}
        ]
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False
        )
        
        n_gpus = config.trainer.n_gpus_per_node
        if num_rubrics < n_gpus:
            prompts = [prompt] * n_gpus
            n_per_prompt = num_rubrics
        else:
            actual_batch_size = ((num_rubrics + n_gpus - 1) // n_gpus) * n_gpus
            prompts = [prompt] * actual_batch_size
            n_per_prompt = 1
        
        encoded = self.tokenizer(
            prompts,
            padding=True,
            truncation=False,
            return_tensors="pt"
        )
        
        attention_mask = encoded["attention_mask"]
        position_ids = attention_mask.long().cumsum(dim=1) - 1
        position_ids = position_ids.clamp(min=0)
        
        batch = DataProto.from_dict(
            tensors={
                "input_ids": encoded["input_ids"],
                "attention_mask": encoded["attention_mask"],
                "position_ids": position_ids
            },
            non_tensors={
                "prompt": np.array(prompts, dtype=object),
                "question": np.array([question] * len(prompts), dtype=object)
            },
            meta_info={
                "n": n_per_prompt,
                "do_sample": True,
                "temperature": float(config.adversarial.get("generation_temperature", 1.0)),
                "top_p": 0.95,
                "num_rubrics_requested": num_rubrics,
            }
        )
        
        return batch
    
    def _prepare_multiple_questions_batch(
        self,
        questions: List[str],
        config,
        use_diverse_sampling: bool = False,
        add_system_prompt: bool = True,
        system_prompt: str = None,
        return_token_entropy: bool = False,
        max_prompt_length_override: int = None,
        temperature_override: Optional[float] = None,
    ) -> DataProto:
        """English docstring."""
        prompts = []
        for question in questions:
            messages = []
            if add_system_prompt:
                prompt_to_use = system_prompt if system_prompt is not None else POLICY_LLM_SYSTEM_PROMPT
                messages.append({
                    "role": "system",
                    "content": prompt_to_use
                })
            messages.append({"role": "user", "content": question})
            
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False
            )
            prompts.append(prompt)
        
        n_gpus = config.trainer.n_gpus_per_node
        num_questions = len(questions)
        
        if num_questions % n_gpus != 0:
            actual_batch_size = ((num_questions + n_gpus - 1) // n_gpus) * n_gpus
            prompts.extend([prompts[-1]] * (actual_batch_size - num_questions))
        
        encoded = self.tokenizer(
            prompts,
            padding=True,
            truncation=False,
            return_tensors="pt"
        )
        
        attention_mask = encoded["attention_mask"]
        position_ids = attention_mask.long().cumsum(dim=1) - 1
        position_ids = position_ids.clamp(min=0)
        
        if temperature_override is not None:
            t_ov = float(temperature_override)
            if t_ov <= 0:
                do_sample = False
                temperature = 0.0
                repetition_penalty = 1.0
            else:
                do_sample = True
                temperature = t_ov
                repetition_penalty = 1.25 if t_ov > 0.7 else 1.0
        elif use_diverse_sampling:
            do_sample = True
            temperature = 1.5
            repetition_penalty = 1.25
        else:
            do_sample = True
            adv = getattr(config, "adversarial", None)
            gen_t = float(adv.get("generation_temperature", 1.0)) if adv is not None else 1.0
            temperature = gen_t
            repetition_penalty = 1.25 if gen_t > 0.7 else 1.0
        
        batch = DataProto.from_dict(
            tensors={
                "input_ids": encoded["input_ids"],
                "attention_mask": encoded["attention_mask"],
                "position_ids": position_ids
            },
            non_tensors={
                "prompt": np.array(prompts, dtype=object)
            },
            meta_info={
                "n": 1,
                "do_sample": do_sample,
                "temperature": temperature,
                "top_p": 0.95,
                "top_k": 50,
                "repetition_penalty": repetition_penalty,
                "num_questions_requested": num_questions,
                "return_token_entropy": return_token_entropy,
            }
        )
        
        return batch
    
    def _prepare_answer_generation_batch(
        self,
        batch: DataProto,
        num_answers: int,
        config,
        use_diverse_sampling: bool = False,
        add_system_prompt: bool = True,
    ) -> DataProto:
        """English docstring."""
        prompts = []
        if batch.non_tensor_batch is not None and "prompt_messages_json" in batch.non_tensor_batch:
            prompt_messages_json_list = list(batch.non_tensor_batch["prompt_messages_json"])
            for prompt_messages_json in prompt_messages_json_list:
                messages = []
                try:
                    raw_messages = json.loads(prompt_messages_json)
                except Exception:
                    raw_messages = []

                if add_system_prompt:
                    has_system = any(
                        isinstance(msg, dict) and msg.get("role") == "system"
                        for msg in raw_messages
                    )
                    if not has_system:
                        messages.append({"role": "system", "content": POLICY_LLM_SYSTEM_PROMPT})

                if isinstance(raw_messages, list):
                    for msg in raw_messages:
                        role = msg.get("role", "")
                        content = msg.get("content", "")
                        if role in ("user", "assistant", "system") and isinstance(content, str):
                            messages.append({"role": role, "content": content})

                if not messages:
                    messages = [{"role": "system", "content": POLICY_LLM_SYSTEM_PROMPT}]

                prompt = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
                prompts.append(prompt)
        elif add_system_prompt and batch.non_tensor_batch is not None and "question" in batch.non_tensor_batch:
            questions = batch.non_tensor_batch["question"]
            for q in questions:
                messages = [
                    {"role": "system", "content": POLICY_LLM_SYSTEM_PROMPT},
                    {"role": "user", "content": q},
                ]
                prompt = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
                prompts.append(prompt)
        elif batch.non_tensor_batch is not None and "prompt" in batch.non_tensor_batch:
            prompts = list(batch.non_tensor_batch["prompt"])
        else:
            prompts = [""]

        n_gpus = config.trainer.n_gpus_per_node
        original_batch_size = len(prompts)
        if original_batch_size < n_gpus:
            repeat_times = (n_gpus + original_batch_size - 1) // original_batch_size
            prompts = (prompts * repeat_times)[:n_gpus]

        actual_batch_size = len(prompts)

        encoded = self.tokenizer(
            prompts,
            padding=True,
            truncation=False,
            return_tensors="pt",
        )

        attention_mask = encoded["attention_mask"]
        position_ids = attention_mask.long().cumsum(dim=1) - 1
        position_ids = position_ids.clamp(min=0)

        gen_batch = DataProto.from_dict(
            tensors={
                "input_ids": encoded["input_ids"],
                "attention_mask": encoded["attention_mask"],
                "position_ids": position_ids,
            },
            non_tensors={
                "prompt": np.array(prompts, dtype=object),
            },
            meta_info=batch.meta_info.copy() if batch.meta_info else {},
        )
        
        if gen_batch.meta_info is None:
            gen_batch.meta_info = {}
        
        gen_batch.meta_info["n"] = num_answers
        gen_batch.meta_info["do_sample"] = True
        gen_batch.meta_info["top_p"] = 0.95
        gen_batch.meta_info["top_k"] = 50
        
        adv = getattr(config, "adversarial", None)
        gen_t = float(adv.get("generation_temperature", 1.0)) if adv is not None else 1.0
        if use_diverse_sampling:
            gen_batch.meta_info["temperature"] = 1.5
            gen_batch.meta_info["repetition_penalty"] = 1.25
        else:
            gen_batch.meta_info["temperature"] = gen_t
            gen_batch.meta_info["repetition_penalty"] = 1.25 if gen_t > 0.7 else 1.0
        
        gen_batch.meta_info["num_answers_requested"] = num_answers
        
        return gen_batch
    
    def _prepare_full_batch_for_training(
        self, gen_batch, gen_batch_output, num_answers, mode, question, rubrics_list
    ) -> DataProto:
        """English docstring."""
        from tensordict import TensorDict
        from verl.utils.model import compute_position_id_with_mask
        
        if gen_batch.batch is not None:
            prompt_attention_mask = gen_batch.batch.get("attention_mask", None)
            prompt_input_ids = gen_batch.batch.get("input_ids", None)
        else:
            prompt_attention_mask = None
            prompt_input_ids = None
        
        if gen_batch_output.batch is not None and "responses" in gen_batch_output.batch:
            responses = gen_batch_output.batch["responses"]
            response_batch_size = responses.shape[0]
        else:
            raise ValueError("")
        
        num_answers_requested = gen_batch.meta_info.get("num_answers_requested", num_answers)
        if response_batch_size > num_answers_requested:
            responses = responses[:num_answers_requested]
            response_batch_size = num_answers_requested
        
        if response_batch_size == 0:
            raise ValueError(
                ""
                f"num_answers_requested={num_answers_requested}, "
                ""
            )
        
        if prompt_input_ids is not None and prompt_input_ids.shape[0] > 1:
            prompt_input_ids = prompt_input_ids[:1]
            if prompt_attention_mask is not None:
                prompt_attention_mask = prompt_attention_mask[:1]
        
        expand_times = response_batch_size
        
        if expand_times > 1 and prompt_input_ids is not None:
            prompt_input_ids = prompt_input_ids.repeat_interleave(expand_times, dim=0)
            if prompt_attention_mask is not None:
                prompt_attention_mask = prompt_attention_mask.repeat_interleave(expand_times, dim=0)
        
        expanded_tensors = {}
        if prompt_input_ids is not None:
            expanded_tensors["input_ids"] = prompt_input_ids
        if prompt_attention_mask is not None:
            expanded_tensors["attention_mask"] = prompt_attention_mask
        
        expanded_non_tensors = {}
        if gen_batch.non_tensor_batch is not None:
            for key, val in gen_batch.non_tensor_batch.items():
                if isinstance(val, np.ndarray):
                    if len(val) > 1:
                        val = val[:1]
                    expanded_non_tensors[key] = np.repeat(val, expand_times)
                else:
                    expanded_non_tensors[key] = val
        
        original_batch = DataProto.from_dict(
            tensors=expanded_tensors,
            non_tensors=expanded_non_tensors,
            meta_info=gen_batch.meta_info.copy() if gen_batch.meta_info else {}
        )
        

        response_batch = TensorDict({"responses": responses}, batch_size=responses.shape[0])

        cleaned_non_tensor_batch = {}
        if gen_batch_output.non_tensor_batch is not None:
            for key, val in gen_batch_output.non_tensor_batch.items():
                if isinstance(val, np.ndarray):
                    if len(val) > num_answers_requested:
                        cleaned_non_tensor_batch[key] = val[:num_answers_requested]
                    else:
                        cleaned_non_tensor_batch[key] = val
                elif isinstance(val, list):
                    if len(val) > num_answers_requested:
                        cleaned_non_tensor_batch[key] = val[:num_answers_requested]
                    else:
                        cleaned_non_tensor_batch[key] = val
                else:
                    cleaned_non_tensor_batch[key] = val
        
        gen_batch_output_clean = DataProto(
            batch=response_batch,
            non_tensor_batch=cleaned_non_tensor_batch,
            meta_info=gen_batch_output.meta_info.copy() if gen_batch_output.meta_info else {}
        )
        
        full_batch = original_batch.union(gen_batch_output_clean)
        
        if prompt_input_ids is not None:
            full_batch.batch["prompts"] = prompt_input_ids
        
        if "responses" in full_batch.batch and prompt_input_ids is not None:
            responses = full_batch.batch["responses"]
            full_input_ids = torch.cat([prompt_input_ids, responses], dim=-1)
            full_batch.batch["input_ids"] = full_input_ids
            
            if prompt_attention_mask is not None:
                attention_mask = prompt_attention_mask
            else:
                pad_token_id = getattr(self.tokenizer, 'pad_token_id', None)
                if pad_token_id is not None:
                    attention_mask = (prompt_input_ids != pad_token_id).long()
                else:
                    attention_mask = torch.ones_like(prompt_input_ids, dtype=torch.long)
            
            pad_token_id = getattr(self.tokenizer, 'pad_token_id', None)
            if pad_token_id is not None:
                response_attention_mask = (responses != pad_token_id).long()
            else:
                response_attention_mask = torch.ones_like(responses, dtype=torch.long)
            
            full_attention_mask = torch.cat([attention_mask, response_attention_mask], dim=-1)
            full_batch.batch["attention_mask"] = full_attention_mask
            
            position_ids = compute_position_id_with_mask(full_attention_mask.unsqueeze(0)).squeeze(0)
            full_batch.batch["position_ids"] = position_ids
        
        if full_batch.non_tensor_batch is None:
            full_batch.non_tensor_batch = {}
        
        batch_size = len(full_batch)
        
        if "uid" not in full_batch.non_tensor_batch:
            timestamp = int(time.time() * 1000)

            uid_value = f"{mode}_{timestamp}_q0"
            full_batch.non_tensor_batch["uid"] = np.array(
                [uid_value] * batch_size, dtype=object
            )
        
        if "data_source" not in full_batch.non_tensor_batch:
            full_batch.non_tensor_batch["data_source"] = np.array(["custom"] * batch_size, dtype=object)
        
        if "reward_model" not in full_batch.non_tensor_batch:
            reward_model_list = [{"style": "custom", "ground_truth": ""} for _ in range(batch_size)]
            full_batch.non_tensor_batch["reward_model"] = np.array(reward_model_list, dtype=object)
        
        if "extra_info" not in full_batch.non_tensor_batch:
            full_batch.non_tensor_batch["extra_info"] = np.array([{} for _ in range(batch_size)], dtype=object)
        
        for i in range(batch_size):
            extra_info = full_batch.non_tensor_batch["extra_info"][i].copy() if isinstance(
                full_batch.non_tensor_batch["extra_info"][i], dict
            ) else {}
            extra_info["adversarial_rubrics_list"] = rubrics_list
            extra_info["adversarial_question"] = question
            extra_info["adversarial_conversation_text"] = conv_text
            extra_info["adversarial_mode"] = mode
            full_batch.non_tensor_batch["extra_info"][i] = extra_info
        
        if batch_size == 0:
            raise ValueError(
                ""
                f"response_batch_size={response_batch_size}, "
                f"num_answers_requested={num_answers_requested}"
            )
        
        return full_batch
    
    def _warmup_sentence_transformer(self):
        """English docstring."""
        if self._sentence_transformer_warmed:
            return
            
        import torch
        from reward_calculator import get_sentence_transformer_model
        
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        print("[message omitted]")
        
        try:
            model = get_sentence_transformer_model("sentence-transformers/all-MiniLM-L6-v2", device=device)
            _ = model.encode(["dummy text"], convert_to_numpy=True)
            self._sentence_transformer_warmed = True
            print("[message omitted]")
        except Exception as e:
            print("[message omitted]")
            if device != "cpu":
                try:
                    print("[message omitted]")
                    model = get_sentence_transformer_model("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
                    _ = model.encode(["dummy text"], convert_to_numpy=True)
                    self._sentence_transformer_warmed = True
                    print("[message omitted]")
                except Exception as e2:
                    print("[message omitted]")
    
    def _prepare_full_batch_for_rubrics_training(
        self, rubrics_batch, rubrics_output, num_rubrics,
        question, ground_truth_rubrics, generated_rubrics, generated_answers,
        reward_config=None, reflect_answers_per_baseline=None,
        question_idx: int | None = None,
        golden_answer: str = "",
    ) -> DataProto:
        """English docstring."""
        from tensordict import TensorDict
        from verl.utils.model import compute_position_id_with_mask
        
        if rubrics_output.batch is not None and "responses" in rubrics_output.batch:
            responses = rubrics_output.batch["responses"]
            num_sequences = responses.shape[0]
            if num_sequences > num_rubrics:
                responses = responses[:num_rubrics]
                print("[message omitted]")
            
            response_batch = TensorDict({"responses": responses}, batch_size=responses.shape[0])
            
            non_tensor_batch_clean = {}
            if rubrics_output.non_tensor_batch:
                for key, value in rubrics_output.non_tensor_batch.items():
                    if isinstance(value, (list, np.ndarray)):
                        non_tensor_batch_clean[key] = value[:responses.shape[0]] if len(value) > responses.shape[0] else value
                    else:
                        non_tensor_batch_clean[key] = value
            
            rubrics_output_clean = DataProto(
                batch=response_batch,
                non_tensor_batch=non_tensor_batch_clean,
                meta_info=rubrics_output.meta_info.copy() if rubrics_output.meta_info else {}
            )
        else:
            raise ValueError("")
        
        if rubrics_batch.batch is not None:
            original_batch_size = rubrics_batch.batch["input_ids"].shape[0]
            target_batch_size = responses.shape[0]
            
            if original_batch_size != target_batch_size:
                prompt_input_ids = rubrics_batch.batch["input_ids"][0:1]  # shape: [1, seq_len]
                prompt_attention_mask = rubrics_batch.batch["attention_mask"][0:1]  # shape: [1, seq_len]
                prompt_position_ids = rubrics_batch.batch.get("position_ids", None)
                if prompt_position_ids is not None:
                    prompt_position_ids = prompt_position_ids[0:1]
                
                prompt_input_ids = prompt_input_ids.repeat(target_batch_size, 1)  # shape: [target_batch_size, seq_len]
                prompt_attention_mask = prompt_attention_mask.repeat(target_batch_size, 1)
                if prompt_position_ids is not None:
                    prompt_position_ids = prompt_position_ids.repeat(target_batch_size, 1)
                
                rubrics_batch.batch["input_ids"] = prompt_input_ids
                rubrics_batch.batch["attention_mask"] = prompt_attention_mask
                if prompt_position_ids is not None:
                    rubrics_batch.batch["position_ids"] = prompt_position_ids
                
                if rubrics_batch.non_tensor_batch:
                    for key, value in rubrics_batch.non_tensor_batch.items():
                        if isinstance(value, (list, np.ndarray)):
                            if len(value) > 0:
                                first_item = value[0] if isinstance(value, list) else value[0]
                                if isinstance(value, list):
                                    rubrics_batch.non_tensor_batch[key] = [first_item] * target_batch_size
                                else:  # np.ndarray
                                    rubrics_batch.non_tensor_batch[key] = np.array([first_item] * target_batch_size, dtype=value.dtype)
            
            prompt_attention_mask = rubrics_batch.batch.get("attention_mask", None)
            prompt_input_ids = rubrics_batch.batch.get("input_ids", None)
        else:
            prompt_attention_mask = None
            prompt_input_ids = None
        
        print(f"[Rubrics Training] rubrics_batch batch_size={len(rubrics_batch)}, rubrics_output_clean batch_size={len(rubrics_output_clean)}")
        
        assert len(rubrics_batch) == len(rubrics_output_clean), \
            ""
        
        full_batch = rubrics_batch.union(rubrics_output_clean)
        print("[message omitted]")
        
        if prompt_input_ids is not None:
            full_batch.batch["prompts"] = prompt_input_ids
        
        if "responses" in full_batch.batch and prompt_input_ids is not None:
            responses = full_batch.batch["responses"]
            full_input_ids = torch.cat([prompt_input_ids, responses], dim=-1)
            full_batch.batch["input_ids"] = full_input_ids
            
            if prompt_attention_mask is not None:
                attention_mask = prompt_attention_mask
            else:
                pad_token_id = getattr(self.tokenizer, 'pad_token_id', None)
                if pad_token_id is not None:
                    attention_mask = (prompt_input_ids != pad_token_id).long()
                else:
                    attention_mask = torch.ones_like(prompt_input_ids, dtype=torch.long)
            
            pad_token_id = getattr(self.tokenizer, 'pad_token_id', None)
            if pad_token_id is not None:
                response_attention_mask = (responses != pad_token_id).long()
            else:
                response_attention_mask = torch.ones_like(responses, dtype=torch.long)
            
            full_attention_mask = torch.cat([attention_mask, response_attention_mask], dim=-1)
            full_batch.batch["attention_mask"] = full_attention_mask
            
            position_ids = compute_position_id_with_mask(full_attention_mask.unsqueeze(0)).squeeze(0)
            full_batch.batch["position_ids"] = position_ids
        
        if full_batch.non_tensor_batch is None:
            full_batch.non_tensor_batch = {}
        
        batch_size = len(full_batch)
        
        timestamp = int(time.time() * 1000)
        qid = 0 if question_idx is None else int(question_idx)
        uid_value = f"rubrics_gen_{timestamp}_q{qid}"
        full_batch.non_tensor_batch["uid"] = np.array(
            [uid_value] * batch_size, dtype=object
        )
        
        if "data_source" not in full_batch.non_tensor_batch:
            full_batch.non_tensor_batch["data_source"] = np.array(["custom"] * batch_size, dtype=object)
        
        if "reward_model" not in full_batch.non_tensor_batch:
            reward_model_list = [{"style": "custom", "ground_truth": ground_truth_rubrics} for _ in range(batch_size)]
            full_batch.non_tensor_batch["reward_model"] = np.array(reward_model_list, dtype=object)
        
        if "extra_info" not in full_batch.non_tensor_batch:
            full_batch.non_tensor_batch["extra_info"] = np.array([{} for _ in range(batch_size)], dtype=object)
        
        for i in range(batch_size):
            extra_info = full_batch.non_tensor_batch["extra_info"][i].copy() if isinstance(
                full_batch.non_tensor_batch["extra_info"][i], dict
            ) else {}
            extra_info["adversarial_policy_answers"] = generated_answers
            extra_info["adversarial_question"] = question
            extra_info["adversarial_conversation_text"] = conv_text
            extra_info["adversarial_ground_truth_rubrics"] = ground_truth_rubrics
            extra_info["adversarial_all_rubrics_list"] = generated_rubrics
            extra_info["adversarial_mode"] = "rubrics_generator"
            extra_info["rubrics_idx"] = i
            
            if reward_config is not None:
                extra_info["adversarial_similarity_weight"] = reward_config.get("similarity_weight")
                extra_info["adversarial_discrimination_weight"] = reward_config.get("discrimination_weight")
                extra_info["adversarial_diversity_weight"] = reward_config.get("diversity_weight")
                extra_info["adversarial_reflect_weight"] = reward_config.get("reflect_weight")
                extra_info["adversarial_similarity_penalty_weight"] = reward_config.get("similarity_penalty_weight", 0.0)
                extra_info["adversarial_similarity_penalty_threshold"] = reward_config.get("similarity_penalty_threshold", 0.8)
                extra_info["adversarial_enable_similarity"] = reward_config.get("enable_similarity")
                extra_info["adversarial_enable_discrimination"] = reward_config.get("enable_discrimination")
                extra_info["adversarial_enable_diversity"] = reward_config.get("enable_diversity")
                extra_info["adversarial_enable_reflect"] = reward_config.get("enable_reflect")

            if reflect_answers_per_baseline is not None:
                extra_info["adversarial_reflect_answers_per_baseline"] = reflect_answers_per_baseline
            
            full_batch.non_tensor_batch["extra_info"][i] = extra_info
        
        return full_batch

    def _precompute_rubrics_rewards(
        self,
        full_batch: DataProto,
        generated_rubrics: list[str],
        generated_answers: list[str],
        question: str,
        ground_truth_rubrics: str,
        adv_config: dict,
        enable_reflect: bool,
        reflect_answers_per_baseline: dict | None,
        golden_answer: str = "",
    ) -> None:
        """English docstring."""
        if full_batch.non_tensor_batch is None:
            return

        extra_info_array = full_batch.non_tensor_batch.get("extra_info")
        if extra_info_array is None or len(generated_rubrics) == 0:
            return

        def _to_bool(value: Any, default: bool) -> bool:
            if value is None:
                return default
            if isinstance(value, str):
                return value.lower() in ("true", "1", "yes")
            return bool(value)

        enable_similarity = _to_bool(adv_config.get("enable_similarity"), True)
        enable_discrimination = _to_bool(adv_config.get("enable_discrimination"), True)

        try:
            from llm_evaluator import LLMEvaluator
            from reward_calculator import (
                batch_calculate_similarity_rewards,
                batch_calculate_discrimination_rewards,
                batch_calculate_reflect_rewards,
                batch_calculate_diversity_rewards,
            )
            from concurrent.futures import ThreadPoolExecutor, as_completed
            import time

            evaluator = LLMEvaluator()
            total_samples = min(len(generated_rubrics), len(extra_info_array))
            max_concurrent_config = max(1, int(adv_config.get("max_concurrent_api_calls", 64)))

            print("[message omitted]")
            overall_start = time.time()
            
            def compute_similarity():
                """English docstring."""
                if not enable_similarity:
                    return [0.0] * total_samples
                print("[message omitted]")
                t_start = time.time()
                result = batch_calculate_similarity_rewards(
                    rubrics_list=generated_rubrics[:total_samples],
                    ground_truth_rubrics=ground_truth_rubrics,
                    llm_evaluator=evaluator,
                    question=question,
                )
                t_elapsed = time.time() - t_start
                print("[message omitted]")
                return result
            
            def compute_discrimination():
                """English docstring."""
                if not enable_discrimination:
                    return [0.0] * total_samples
                print("[message omitted]")
                t_start = time.time()
                result = batch_calculate_discrimination_rewards(
                    rubrics_list=generated_rubrics[:total_samples],
                    policy_answers=generated_answers,
                    llm_evaluator=evaluator,
                    question=question,
                    return_baseline_scores_dict=False,
                )
                t_elapsed = time.time() - t_start
                print("[message omitted]")
                return result
            
            def compute_diversity():
                """English docstring."""
                enable_diversity = _to_bool(adv_config.get("enable_diversity"), True)
                if not enable_diversity:
                    return [0.0] * total_samples
                print("[message omitted]")
                t_start = time.time()
                result = batch_calculate_diversity_rewards(
                    rubrics_list=generated_rubrics[:total_samples],
                )
                t_elapsed = time.time() - t_start
                print("[message omitted]")
                return result
            
            def compute_reflect():
                """English docstring."""
                if not enable_reflect or not reflect_answers_per_baseline:
                    return {}
                print("[message omitted]")
                t_start = time.time()
                
                batch_rubrics_list = []
                batch_gt_rubrics_list = []
                batch_baseline_answers = []
                batch_reflect_answers = []
                batch_questions = []
                processed_rubrics_indices = []
                
                for rubrics_idx in range(total_samples):
                    reflect_answers_for_rubric = {}
                    for baseline_idx, rubrics_dict in reflect_answers_per_baseline.items():
                        if isinstance(rubrics_dict, dict) and rubrics_idx in rubrics_dict:
                            reflect_answers_for_rubric[baseline_idx] = rubrics_dict[rubrics_idx]
                    
                    if reflect_answers_for_rubric:
                        batch_rubrics_list.append(generated_rubrics[rubrics_idx])
                        batch_gt_rubrics_list.append(generated_rubrics[rubrics_idx])
                        batch_baseline_answers.append(generated_answers)
                        batch_reflect_answers.append(reflect_answers_for_rubric)
                        batch_questions.append(conv_text)
                        processed_rubrics_indices.append(rubrics_idx)
                
                results = {}
                if batch_rubrics_list:
                    batch_reflect_rewards = batch_calculate_reflect_rewards(
                        rubrics_list=batch_rubrics_list,
                        ground_truth_rubrics_list=batch_gt_rubrics_list,
                        baseline_answers_list=batch_baseline_answers,
                        reflect_answers_per_rubrics=batch_reflect_answers,
                        llm_evaluator=evaluator,
                        questions=batch_questions,
                        max_concurrent=max_concurrent_config,
                    )
                    
                    for i, rubrics_idx in enumerate(processed_rubrics_indices):
                        if i < len(batch_reflect_rewards):
                            results[rubrics_idx] = batch_reflect_rewards[i]
                    
                    print("[message omitted]")
                
                t_elapsed = time.time() - t_start
                print("[message omitted]")
                return results
            
            # Contrast reward is permanently disabled in EvoRubric.
            with ThreadPoolExecutor(max_workers=4) as executor:
                future_similarity = executor.submit(compute_similarity)
                future_discrimination = executor.submit(compute_discrimination)
                future_diversity = executor.submit(compute_diversity)
                future_reflect = executor.submit(compute_reflect)

                similarity_rewards = future_similarity.result()
                discrimination_rewards = future_discrimination.result()
                diversity_rewards = future_diversity.result()
                reflect_rewards = future_reflect.result()

            overall_elapsed = time.time() - overall_start
            print(f"[Reward Cache] all rewards computed in {overall_elapsed:.2f}s")

            assert len(similarity_rewards) == total_samples, \
                f"similarity_rewards length mismatch: expected {total_samples}, got {len(similarity_rewards)}"
            assert len(discrimination_rewards) == total_samples, \
                f"discrimination_rewards length mismatch: expected {total_samples}, got {len(discrimination_rewards)}"
            assert len(diversity_rewards) == total_samples, \
                f"diversity_rewards length mismatch: expected {total_samples}, got {len(diversity_rewards)}"

            total_cached = 0
            for sample_idx in range(total_samples):
                extra_info = extra_info_array[sample_idx]
                if not isinstance(extra_info, dict):
                    extra_info = {}

                cached_details = {
                    "similarity_reward": similarity_rewards[sample_idx] if sample_idx < len(similarity_rewards) else 0.0,
                    "discrimination_reward": discrimination_rewards[sample_idx] if sample_idx < len(discrimination_rewards) else 0.0,
                    "diversity_reward": diversity_rewards[sample_idx] if sample_idx < len(diversity_rewards) else 0.0,
                    "reflect_reward": reflect_rewards.get(sample_idx, 0.0),
                }
                extra_info["cached_reward_details"] = cached_details
                extra_info_array[sample_idx] = extra_info
                total_cached += 1

            print(f"[Reward Cache] cached {total_cached} rubric reward dicts (similarity, discrimination, diversity, reflect)")
        except Exception as e:
            print("[message omitted]")
            import traceback
            traceback.print_exc()
    
    def _log_to_swanlab(self, metrics):
        """English docstring."""
        if self.swanlab_run is None:
            return
        try:
            import swanlab
            from custom_reward_fn import GLOBAL_REWARD_STATE

            scalar_suffix_to_canonical = {
                "policy_reward_mean": "policy_llm/reward_mean",
                "rubrics_reward_mean": "rubrics_generator/reward_mean",
                "similarity_reward_mean": "rubrics_generator/similarity_reward_mean",
                "discrimination_reward_mean": "rubrics_generator/discrimination_reward_mean",
                "diversity_reward_mean": "rubrics_generator/diversity_reward_mean",
                "reflect_reward_mean": "rubrics_generator/reflect_reward_mean",
                "similarity_penalty_mean": "rubrics_generator/similarity_penalty_mean",
                "intra_rubric_avg_cos_sim_mean": "rubrics_generator/intra_rubric_avg_cos_sim_mean",
            }
            for k, v in list(metrics.items()):
                if not (k.startswith("iter") and "/" in k):
                    continue
                if "/policy_phase/" in k:
                    suf = k.split("/policy_phase/", 1)[1]
                    if suf == "reward_mean":
                        metrics["policy_llm/reward_mean"] = v
                    elif suf.startswith("actor/"):
                        metrics["policy_llm/" + suf] = v
                    else:
                        metrics["policy_llm/" + suf] = v
                    continue
                if "/rubrics_phase/" in k:
                    suf = k.split("/rubrics_phase/", 1)[1]
                    if suf == "reward_mean":
                        metrics["rubrics_generator/reward_mean"] = v
                    elif suf.startswith("actor/"):
                        metrics["rubrics_generator/" + suf] = v
                    else:
                        metrics["rubrics_generator/" + suf] = v
                    continue
                _, suffix = k.split("/", 1)
                if suffix in scalar_suffix_to_canonical:
                    metrics[scalar_suffix_to_canonical[suffix]] = v
                if suffix.startswith("policy/actor/"):
                    metrics["policy_llm/actor/" + suffix[len("policy/actor/"):]] = v
                if suffix.startswith("rubrics/actor/"):
                    metrics["rubrics_generator/actor/" + suffix[len("rubrics/actor/"):]] = v

            swanlab_metrics = {}

            if "policy_llm/reward_mean" in metrics:
                swanlab_metrics["policy_llm/reward_mean"] = metrics["policy_llm/reward_mean"]
            for key in ["policy_llm/actor/loss", "policy_llm/actor/entropy",
                        "policy_llm/actor/kl_loss", "policy_llm/actor/pg_loss",
                        "policy_llm/actor/pg_clipfrac", "policy_llm/actor/ppo_kl"]:
                if key in metrics:
                    swanlab_metrics[key] = metrics[key]

            if "rubrics_generator/reward_mean" in metrics:
                swanlab_metrics["rubrics_generator/reward_mean"] = metrics["rubrics_generator/reward_mean"]
            for key in ["rubrics_generator/actor/loss", "rubrics_generator/actor/entropy",
                        "rubrics_generator/actor/kl_loss", "rubrics_generator/actor/pg_loss",
                        "rubrics_generator/actor/pg_clipfrac", "rubrics_generator/actor/ppo_kl"]:
                if key in metrics:
                    swanlab_metrics[key] = metrics[key]
            sub_reward_keys = [
                "similarity_reward", "discrimination_reward", "diversity_reward",
                "reflect_reward",
                "similarity_penalty",
                "intra_rubric_avg_cos_sim",
            ]
            for key in sub_reward_keys:
                metric_key = f"rubrics_generator/{key}_mean"
                if metric_key in metrics:
                    swanlab_metrics[metric_key] = metrics[metric_key]

            if not swanlab_metrics.get("rubrics_generator/reward_mean") and GLOBAL_REWARD_STATE.sub_rewards_history:
                for key in sub_reward_keys:
                    values = [d.get(key, 0) for d in GLOBAL_REWARD_STATE.sub_rewards_history
                              if isinstance(d, dict) and key in d]
                    if values:
                        swanlab_metrics[f"rubrics_generator/{key}_mean"] = np.mean(values)
                if "rubrics_generator/reward_mean" not in swanlab_metrics and GLOBAL_REWARD_STATE.sub_rewards_history:
                    totals = [d.get("total_reward", 0) for d in GLOBAL_REWARD_STATE.sub_rewards_history
                              if isinstance(d, dict) and "total_reward" in d]
                    if totals:
                        swanlab_metrics["rubrics_generator/reward_mean"] = np.mean(totals)
                GLOBAL_REWARD_STATE.sub_rewards_history.clear()

            policy_reward = metrics.get("policy_llm/reward_mean", 0)
            rubrics_reward = metrics.get("rubrics_generator/reward_mean", 0)
            swanlab_metrics["total/reward_mean"] = (float(policy_reward) + float(rubrics_reward)) / 2

            swanlab.log(swanlab_metrics, step=self.global_steps)
        except Exception as e:
            print("[message omitted]")
            import traceback
            traceback.print_exc()

    def _save_checkpoints(self, config, epoch):
        """English docstring."""
        base_dir = config.trainer.default_local_dir
        
        policy_path = os.path.join(base_dir, "policy_llm", f"step_{self.global_steps}")
        self.actor_rollout_wg.save_checkpoint(policy_path, POLICY_LLM_ADAPTER)
        
        rubrics_path = os.path.join(base_dir, "rubrics_generator", f"step_{self.global_steps}")
        self.actor_rollout_wg.save_checkpoint(rubrics_path, RUBRICS_GEN_ADAPTER)
        
        print("[message omitted]")

    def _prepare_full_batch_for_training_batch(
        self, gen_batch, gen_batch_output, num_answers, batch_size,
        mode, questions, rubrics_list,
        golden_answer_list=None,
        policy_reward_config: dict = None,
        conversation_texts: list = None,
        include_question_indices: Optional[List[int]] = None,
    ) -> DataProto:
        """English docstring."""
        from tensordict import TensorDict
        from verl.utils.model import compute_position_id_with_mask

        if gen_batch.batch is not None:
            prompt_attention_mask = gen_batch.batch.get("attention_mask", None)
            prompt_input_ids = gen_batch.batch.get("input_ids", None)
        else:
            prompt_attention_mask = None
            prompt_input_ids = None

        if gen_batch_output.batch is not None and "responses" in gen_batch_output.batch:
            responses = gen_batch_output.batch["responses"]
            total_response_batch_size = responses.shape[0]  # batch_size * num_answers
        else:
            raise ValueError("")

        q_list: List[int] = (
            [int(i) for i in include_question_indices]
            if include_question_indices is not None
            else list(range(batch_size))
        )
        if len(q_list) == 0:
            raise ValueError(
                ""
            )
        n_q = len(q_list)

        expected_total = batch_size * num_answers
        if total_response_batch_size != expected_total:
            print("[message omitted]")
            if total_response_batch_size > expected_total:
                responses = responses[:expected_total]
                total_response_batch_size = expected_total

        if include_question_indices is not None:
            _chunks = []
            for q_idx in q_list:
                s, e = q_idx * num_answers, (q_idx + 1) * num_answers
                if s < responses.shape[0]:
                    _chunks.append(responses[s : min(e, responses.shape[0])])
            responses = torch.cat(_chunks, dim=0) if _chunks else responses[:0]

        train_response_len = int(responses.shape[0])

        expanded_prompt_input_ids = []
        expanded_prompt_attention_mask = []
        expanded_non_tensors = {}

        if prompt_input_ids is not None:
            # prompt_input_ids.shape = (batch_size, seq_len)
            for q_idx in q_list:
                if q_idx >= prompt_input_ids.shape[0]:
                    continue
                q_prompt = prompt_input_ids[q_idx : q_idx + 1]  # (1, seq_len)
                q_prompt_repeated = q_prompt.repeat(num_answers, 1)  # (num_answers, seq_len)
                expanded_prompt_input_ids.append(q_prompt_repeated)
            expanded_prompt_input_ids = torch.cat(expanded_prompt_input_ids, dim=0)  # (n_q * num_answers, seq_len)

            if prompt_attention_mask is not None:
                for q_idx in q_list:
                    if q_idx >= prompt_attention_mask.shape[0]:
                        continue
                    q_mask = prompt_attention_mask[q_idx : q_idx + 1]
                    q_mask_repeated = q_mask.repeat(num_answers, 1)
                    expanded_prompt_attention_mask.append(q_mask_repeated)
                expanded_prompt_attention_mask = torch.cat(expanded_prompt_attention_mask, dim=0)

        if gen_batch.non_tensor_batch is not None:
            for key, val in gen_batch.non_tensor_batch.items():
                if isinstance(val, np.ndarray):
                    expanded_vals = []
                    for q_idx in q_list:
                        if q_idx >= len(val):
                            continue
                        q_val = val[q_idx : q_idx + 1]
                        q_val_repeated = np.repeat(q_val, num_answers)
                        expanded_vals.append(q_val_repeated)
                    expanded_non_tensors[key] = np.concatenate(expanded_vals, axis=0)
                else:
                    expanded_non_tensors[key] = val

        expanded_tensors = {}
        if prompt_input_ids is not None:
            expanded_tensors["input_ids"] = expanded_prompt_input_ids
        if prompt_attention_mask is not None:
            expanded_tensors["attention_mask"] = expanded_prompt_attention_mask

        original_batch = DataProto.from_dict(
            tensors=expanded_tensors,
            non_tensors=expanded_non_tensors,
            meta_info=gen_batch.meta_info.copy() if gen_batch.meta_info else {}
        )

        response_batch = TensorDict({"responses": responses}, batch_size=responses.shape[0])

        cleaned_non_tensor_batch = {}
        if gen_batch_output.non_tensor_batch is not None:
            for key, val in gen_batch_output.non_tensor_batch.items():
                if key in expanded_non_tensors:
                    continue
                if include_question_indices is not None:
                    if isinstance(val, np.ndarray) and len(val) >= expected_total:
                        parts = [
                            val[q_idx * num_answers : (q_idx + 1) * num_answers]
                            for q_idx in q_list
                            if (q_idx + 1) * num_answers <= len(val)
                        ]
                        if parts:
                            cleaned_non_tensor_batch[key] = np.concatenate(parts, axis=0)
                        else:
                            cleaned_non_tensor_batch[key] = val[:train_response_len]
                    elif isinstance(val, list) and len(val) >= expected_total:
                        rows: List[Any] = []
                        for q_idx in q_list:
                            s, e = q_idx * num_answers, (q_idx + 1) * num_answers
                            if e <= len(val):
                                rows.extend(val[s:e])
                        cleaned_non_tensor_batch[key] = rows if rows else val[:train_response_len]
                    else:
                        cleaned_non_tensor_batch[key] = val
                elif isinstance(val, np.ndarray) and len(val) >= total_response_batch_size:
                    cleaned_non_tensor_batch[key] = val[:total_response_batch_size]
                elif isinstance(val, list) and len(val) >= total_response_batch_size:
                    cleaned_non_tensor_batch[key] = val[:total_response_batch_size]
                else:
                    cleaned_non_tensor_batch[key] = val

        gen_batch_output_clean = DataProto(
            batch=response_batch,
            non_tensor_batch=cleaned_non_tensor_batch,
            meta_info=gen_batch_output.meta_info.copy() if gen_batch_output.meta_info else {}
        )

        full_batch = original_batch.union(gen_batch_output_clean)

        if prompt_input_ids is not None:
            full_batch.batch["prompts"] = expanded_prompt_input_ids

        if "responses" in full_batch.batch and expanded_prompt_input_ids is not None:
            responses = full_batch.batch["responses"]
            full_input_ids = torch.cat([expanded_prompt_input_ids, responses], dim=-1)
            full_batch.batch["input_ids"] = full_input_ids

            if expanded_prompt_attention_mask is not None:
                attention_mask = expanded_prompt_attention_mask
            else:
                pad_token_id = getattr(self.tokenizer, 'pad_token_id', None)
                if pad_token_id is not None:
                    attention_mask = (expanded_prompt_input_ids != pad_token_id).long()
                else:
                    attention_mask = torch.ones_like(expanded_prompt_input_ids, dtype=torch.long)

            pad_token_id = getattr(self.tokenizer, 'pad_token_id', None)
            if pad_token_id is not None:
                response_attention_mask = (responses != pad_token_id).long()
            else:
                response_attention_mask = torch.ones_like(responses, dtype=torch.long)

            full_attention_mask = torch.cat([attention_mask, response_attention_mask], dim=-1)
            full_batch.batch["attention_mask"] = full_attention_mask

            position_ids = compute_position_id_with_mask(full_attention_mask.unsqueeze(0)).squeeze(0)
            full_batch.batch["position_ids"] = position_ids

        if full_batch.non_tensor_batch is None:
            full_batch.non_tensor_batch = {}

        full_batch_size = len(full_batch)

        timestamp = int(time.time() * 1000)
        uids = []
        for q_idx in q_list:
            uid_for_question = f"{mode}_{timestamp}_q{q_idx}"
            for _a_idx in range(num_answers):
                uids.append(uid_for_question)
        full_batch.non_tensor_batch["uid"] = np.array(uids, dtype=object)

        if "data_source" not in full_batch.non_tensor_batch:
            full_batch.non_tensor_batch["data_source"] = np.array(["custom"] * full_batch_size, dtype=object)

        if "reward_model" not in full_batch.non_tensor_batch:
            reward_model_list = [{"style": "custom", "ground_truth": ""} for _ in range(full_batch_size)]
            full_batch.non_tensor_batch["reward_model"] = np.array(reward_model_list, dtype=object)

        if "extra_info" not in full_batch.non_tensor_batch:
            full_batch.non_tensor_batch["extra_info"] = np.array([{} for _ in range(full_batch_size)], dtype=object)

        sample_idx = 0
        for q_idx in q_list:
            question = questions[q_idx]
            rubrics_for_q = [rubrics_list[q_idx]]

            for a_idx in range(num_answers):
                extra_info = full_batch.non_tensor_batch["extra_info"][sample_idx].copy() if isinstance(
                    full_batch.non_tensor_batch["extra_info"][sample_idx], dict
                ) else {}

                extra_info["adversarial_rubrics_list"] = rubrics_for_q
                extra_info["adversarial_question"] = question
                extra_info["adversarial_conversation_text"] = (
                    conversation_texts[q_idx]
                    if conversation_texts and q_idx < len(conversation_texts)
                    else question
                )
                extra_info["adversarial_mode"] = mode
                extra_info["question_idx"] = q_idx

                full_batch.non_tensor_batch["extra_info"][sample_idx] = extra_info
                sample_idx += 1

        if full_batch_size == 0:
            raise ValueError(
                ""
                f"batch_size={batch_size}, num_answers={num_answers}"
            )

        sub = "" if include_question_indices is not None else ""
        print(
            "[message omitted]"
            f"(questions={n_q} × answers={num_answers}{sub})"
        )

        return full_batch

    def _train_rubrics_generator_step_batch(
        self,
        batch: DataProto,
        questions: list,
        ground_truth_rubrics_list: list,
        num_rubrics: int,
        num_answers: int,
        config,
        golden_answer_list: list = None,
    ) -> Dict[str, float]:
        """English docstring."""
        import time
        import sys
        from verl.utils.memory_utils import aggressive_empty_cache

        batch_size = len(questions)
        metrics = {}
        all_generated_rubrics = []
        all_generated_answers = []

        answer_aware_num_refs = int(config.adversarial.get("answer_aware_num_refs", 2))
        use_answer_aware = answer_aware_num_refs > 0

        prompt_messages_json_list = None
        instruction_list = None
        if batch.non_tensor_batch is not None:
            if "prompt_messages_json" in batch.non_tensor_batch:
                prompt_messages_json_list = list(batch.non_tensor_batch["prompt_messages_json"])
            if "instruction" in batch.non_tensor_batch:
                instruction_list = list(batch.non_tensor_batch["instruction"])

        conversation_texts = [
            self._format_conversation_text(prompt_messages_json_list, q_idx, questions)
            for q_idx in range(batch_size)
        ]

        gen_batch = None
        gen_batch_output = None

        if use_answer_aware:
            print(
                "[message omitted]"
            )
            t_answers_start = time.time()
            self.actor_rollout_wg.set_adapter(POLICY_LLM_ADAPTER)
            gen_batch = self._prepare_answer_generation_batch(batch, num_answers, config, use_diverse_sampling=True)
            gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)

            if gen_batch_output.batch is not None and "responses" in gen_batch_output.batch:
                response_ids = gen_batch_output.batch["responses"]
                total_expected = batch_size * num_answers
                actual_num_responses = response_ids.size(0)
                prompt_count = None
                try:
                    if gen_batch.batch is not None and "input_ids" in gen_batch.batch:
                        prompt_count = int(gen_batch.batch["input_ids"].shape[0])
                except Exception:
                    prompt_count = None
                if prompt_count is None:
                    prompt_count = len(gen_batch)

                if actual_num_responses < total_expected:
                    raise RuntimeError(
                        ""
                        ""
                        ""
                    )

                expected_from_prompts = prompt_count * num_answers
                if prompt_count > 0 and actual_num_responses != expected_from_prompts:
                    inferred_n = actual_num_responses / prompt_count if prompt_count else float("inf")
                    raise RuntimeError(
                        ""
                        ""
                        ""
                        ""
                        ""
                    )

                all_answers_flat = []
                for i in range(min(total_expected, response_ids.size(0))):
                    response_text = self._safe_decode_token_ids(response_ids[i], sequence_idx=i)
                    all_answers_flat.append(response_text)

                for q_idx in range(batch_size):
                    start_idx = q_idx * num_answers
                    end_idx = start_idx + num_answers
                    generated_answers = all_answers_flat[start_idx:end_idx]
                    all_generated_answers.append(generated_answers)
                    print("[message omitted]")
            else:
                for q_idx in range(batch_size):
                    all_generated_answers.append([])
                    print("[message omitted]")

            t_answers_total = time.time() - t_answers_start
            print("[message omitted]")

        print("[message omitted]")
        t_rubrics_start = time.time()
        self.actor_rollout_wg.set_adapter(RUBRICS_GEN_ADAPTER)

        rubrics_batch = self._prepare_rubrics_generation_batch_multi(
            questions,
            num_rubrics,
            config,
            prompt_messages_json_list=prompt_messages_json_list,
            instruction_list=instruction_list,
            exemplar_data=self.previous_best_responses if self._exemplar_trick_enabled(config) else None,
            answers_per_query=all_generated_answers if use_answer_aware else None,
        )
        rubrics_output = self.actor_rollout_wg.generate_sequences(rubrics_batch)

        if rubrics_output.batch is not None and "responses" in rubrics_output.batch:
            response_ids = rubrics_output.batch["responses"]
            total_expected = batch_size * num_rubrics
            
            actual_num_responses = response_ids.size(0)
            prompt_count = None
            try:
                if rubrics_batch.batch is not None and "input_ids" in rubrics_batch.batch:
                    prompt_count = int(rubrics_batch.batch["input_ids"].shape[0])
            except Exception:
                prompt_count = None
            if prompt_count is None:
                prompt_count = len(rubrics_batch)

            if actual_num_responses < total_expected:
                raise RuntimeError(
                    ""
                    ""
                    ""
                )

            if prompt_count > 0 and actual_num_responses != prompt_count:
                inferred_n = actual_num_responses / prompt_count if prompt_count else float("inf")
                raise RuntimeError(
                    ""
                    ""
                    ""
                    ""
                    ""
                )

            all_rubrics_flat = []
            for i in range(min(total_expected, response_ids.size(0))):
                response_text = self._safe_decode_token_ids(response_ids[i], sequence_idx=i)
                all_rubrics_flat.append(response_text)

            for q_idx in range(batch_size):
                start_idx = q_idx * num_rubrics
                end_idx = start_idx + num_rubrics
                generated_rubrics = all_rubrics_flat[start_idx:end_idx]
                all_generated_rubrics.append(generated_rubrics)
                print("[message omitted]")
        else:
            for q_idx in range(batch_size):
                all_generated_rubrics.append([])
                print("[message omitted]")

        t_rubrics_total = time.time() - t_rubrics_start
        print("[message omitted]")
        
        try:
            rubrics_prompts = []
            if rubrics_batch.non_tensor_batch is not None and "prompt" in rubrics_batch.non_tensor_batch:
                rubrics_prompts = list(rubrics_batch.non_tensor_batch["prompt"])
                expected_prompts = batch_size * num_rubrics
                if len(rubrics_prompts) > expected_prompts:
                    rubrics_prompts = rubrics_prompts[:expected_prompts]
            else:
                if rubrics_batch.batch is not None and "input_ids" in rubrics_batch.batch:
                    input_ids = rubrics_batch.batch["input_ids"]
                    for i in range(min(batch_size * num_rubrics, input_ids.shape[0])):
                        prompt_text = self._safe_decode_token_ids(input_ids[i], sequence_idx=i)
                        rubrics_prompts.append(prompt_text)
            
            rubrics_outputs_flat = []
            for rubrics_list in all_generated_rubrics:
                rubrics_outputs_flat.extend(rubrics_list)
            
            self._save_prompts_and_outputs(
                step=self.global_steps,
                model_type="rubrics_generator",
                prompts=rubrics_prompts,
                outputs=rubrics_outputs_flat,
                questions=questions,
                metadata={"num_rubrics": num_rubrics, "num_answers": num_answers}
            )
        except Exception as e:
            print("[message omitted]")
            import traceback
            traceback.print_exc()

        if not use_answer_aware:
            print("[message omitted]")
            t_answers_start = time.time()
            self.actor_rollout_wg.set_adapter(POLICY_LLM_ADAPTER)

            gen_batch = self._prepare_answer_generation_batch(batch, num_answers, config, use_diverse_sampling=True)
            gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)

            if gen_batch_output.batch is not None and "responses" in gen_batch_output.batch:
                response_ids = gen_batch_output.batch["responses"]
                total_expected = batch_size * num_answers

                actual_num_responses = response_ids.size(0)
                prompt_count = None
                try:
                    if gen_batch.batch is not None and "input_ids" in gen_batch.batch:
                        prompt_count = int(gen_batch.batch["input_ids"].shape[0])
                except Exception:
                    prompt_count = None
                if prompt_count is None:
                    prompt_count = len(gen_batch)

                if actual_num_responses < total_expected:
                    raise RuntimeError(
                        ""
                        ""
                        ""
                    )

                expected_from_prompts = prompt_count * num_answers
                if prompt_count > 0 and actual_num_responses != expected_from_prompts:
                    inferred_n = actual_num_responses / prompt_count if prompt_count else float("inf")
                    raise RuntimeError(
                        ""
                        ""
                        ""
                        ""
                        ""
                    )

                all_answers_flat = []
                for i in range(min(total_expected, response_ids.size(0))):
                    response_text = self._safe_decode_token_ids(response_ids[i], sequence_idx=i)
                    all_answers_flat.append(response_text)

                for q_idx in range(batch_size):
                    start_idx = q_idx * num_answers
                    end_idx = start_idx + num_answers
                    generated_answers = all_answers_flat[start_idx:end_idx]
                    all_generated_answers.append(generated_answers)
                    print("[message omitted]")
            else:
                for q_idx in range(batch_size):
                    all_generated_answers.append([])
                    print("[message omitted]")

            t_answers_total = time.time() - t_answers_start
            print("[message omitted]")

        adv_config = self.config.get("adversarial", {})
        enable_reflect_val = adv_config.get("enable_reflect")
        enable_reflect = (
            enable_reflect_val.lower() in ("true", "1", "yes")
            if isinstance(enable_reflect_val, str) else bool(enable_reflect_val)
            if enable_reflect_val is not None else False
        )

        print("[message omitted]")

        reflect_answers_per_question = {}  # {q_idx: {baseline_idx: {rubrics_idx: revised_answer}}}

        if enable_reflect:
            print("[message omitted]")

            all_reflect_prompts = []  
            reflect_prompt_mapping = []
            
            for q_idx in range(batch_size):
                reflect_answers_per_question[q_idx] = {}
                generated_answers_for_q = all_generated_answers[q_idx]
                for baseline_idx in range(len(generated_answers_for_q)):
                    reflect_answers_per_question[q_idx][baseline_idx] = {}

            for q_idx in range(batch_size):
                conv_text = self._format_conversation_text(
                    prompt_messages_json_list, q_idx, questions
                )
                generated_rubrics_for_q = all_generated_rubrics[q_idx]
                generated_answers_for_q = all_generated_answers[q_idx]

                for baseline_idx, baseline_answer in enumerate(generated_answers_for_q):
                    for rubrics_idx, rubrics in enumerate(generated_rubrics_for_q):
                        reflect_prompt = REFLECT_PROMPT_TEMPLATE.format(
                            question=conv_text,
                            baseline_answer=baseline_answer,
                            rubrics=rubrics
                        )
                        all_reflect_prompts.append(reflect_prompt)
                        reflect_prompt_mapping.append((q_idx, baseline_idx, rubrics_idx))

            total_reflect_prompts = len(all_reflect_prompts)
            print("[message omitted]")

            REFLECT_BATCH_SIZE = 64
            num_batches = (total_reflect_prompts + REFLECT_BATCH_SIZE - 1) // REFLECT_BATCH_SIZE
            print("[message omitted]")
            from verl.utils.memory_utils import aggressive_empty_cache

            for batch_start in range(0, total_reflect_prompts, REFLECT_BATCH_SIZE):
                batch_end = min(batch_start + REFLECT_BATCH_SIZE, total_reflect_prompts)
                batch_prompts = all_reflect_prompts[batch_start:batch_end]
                batch_num = batch_start // REFLECT_BATCH_SIZE + 1

                print("[message omitted]")

                reflect_batch = self._prepare_multiple_questions_batch(
                    batch_prompts,
                    config,
                    use_diverse_sampling=False,
                    add_system_prompt=True,
                    system_prompt=REFLECT_SYSTEM_PROMPT,
                    max_prompt_length_override=3072,
                    temperature_override=float(
                        config.adversarial.get("reflect_generation_temperature", 0.0)
                    ),
                )
                reflect_output = self.actor_rollout_wg.generate_sequences(reflect_batch)

                if reflect_output.batch is not None and "responses" in reflect_output.batch:
                    response_ids = reflect_output.batch["responses"]

                    for local_idx in range(min(len(batch_prompts), response_ids.size(0))):
                        global_idx = batch_start + local_idx
                        q_idx, baseline_idx, rubrics_idx = reflect_prompt_mapping[global_idx]
                        revised_answer = self._safe_decode_token_ids(response_ids[local_idx], sequence_idx=local_idx)
                        reflect_answers_per_question[q_idx][baseline_idx][rubrics_idx] = revised_answer
                else:
                    print("[message omitted]")
                    for local_idx in range(len(batch_prompts)):
                        global_idx = batch_start + local_idx
                        q_idx, baseline_idx, rubrics_idx = reflect_prompt_mapping[global_idx]
                        reflect_answers_per_question[q_idx][baseline_idx][rubrics_idx] = all_generated_answers[q_idx][baseline_idx]

                del reflect_batch, reflect_output
                aggressive_empty_cache()

            for q_idx in range(batch_size):
                reflect_answers_for_q = reflect_answers_per_question[q_idx]
                total_for_q = sum(len(v) for v in reflect_answers_for_q.values())
                print("[message omitted]")

            print("[message omitted]")
            
            import os
            import json
            prompts_log_dir = os.path.join(self.config.trainer.default_local_dir, "training_logs", "prompts_and_outputs")
            os.makedirs(prompts_log_dir, exist_ok=True)
            
            reflect_log_path = os.path.join(prompts_log_dir, f"step_{self.global_steps:06d}_reflect.json")
            reflect_log_data = {
                "step": self.global_steps,
                "batch_size": batch_size,
                "questions": [],
                "total_reflect_prompts": total_reflect_prompts
            }
            
            for q_idx in range(batch_size):
                question = questions[q_idx]
                conv_text = self._format_conversation_text(
                    prompt_messages_json_list, q_idx, questions
                )
                generated_rubrics_for_q = all_generated_rubrics[q_idx]
                generated_answers_for_q = all_generated_answers[q_idx]
                reflect_answers_for_q = reflect_answers_per_question[q_idx]
                
                question_data = {
                    "question_idx": q_idx,
                    "question": question,
                    "num_baseline_answers": len(generated_answers_for_q),
                    "num_rubrics": len(generated_rubrics_for_q),
                    "reflect_prompts_and_outputs": []
                }
                
                for baseline_idx, baseline_answer in enumerate(generated_answers_for_q):
                    for rubrics_idx, rubrics in enumerate(generated_rubrics_for_q):
                        reflect_prompt_raw = REFLECT_PROMPT_TEMPLATE.format(
                            question=conv_text,
                            baseline_answer=baseline_answer,
                            rubrics=rubrics
                        )
                        
                        messages = [
                            {"role": "system", "content": REFLECT_SYSTEM_PROMPT},
                            {"role": "user", "content": reflect_prompt_raw}
                        ]
                        reflect_prompt_full = self.tokenizer.apply_chat_template(
                            messages,
                            tokenize=False,
                            add_generation_prompt=True,
                            enable_thinking=False
                        )
                        
                        revised_answer = reflect_answers_for_q.get(baseline_idx, {}).get(rubrics_idx, "")
                        
                        question_data["reflect_prompts_and_outputs"].append({
                            "baseline_idx": baseline_idx,
                            "rubrics_idx": rubrics_idx,
                            "baseline_answer": baseline_answer,
                            "rubrics": rubrics,
                            "reflect_prompt_raw": reflect_prompt_raw,
                            "reflect_prompt_full": reflect_prompt_full,
                            "revised_answer": revised_answer,
                            "prompt_length_tokens": len(self.tokenizer.encode(reflect_prompt_full)),
                            "response_length_tokens": len(self.tokenizer.encode(revised_answer)) if revised_answer else 0
                        })
                
                reflect_log_data["questions"].append(question_data)
            
            with open(reflect_log_path, "w", encoding="utf-8") as f:
                json.dump(reflect_log_data, f, ensure_ascii=False, indent=2)
            
            print("[message omitted]")
            print("[message omitted]")

        print("[message omitted]")

        all_full_batches = []  

        for q_idx in range(batch_size):
            print("[message omitted]")

            question = questions[q_idx]
            ground_truth_rubrics = ground_truth_rubrics_list[q_idx]
            golden_answer = (
                golden_answer_list[q_idx]
                if (golden_answer_list is not None and q_idx < len(golden_answer_list))
                else ""
            )
            generated_rubrics_for_q = all_generated_rubrics[q_idx]
            generated_answers_for_q = all_generated_answers[q_idx]

            q_batch_dict = {
                k: v[q_idx:q_idx+1] if torch.is_tensor(v) and len(v) > q_idx else v
                for k, v in batch.batch.items()
            }
            if batch.non_tensor_batch is not None:
                for key, val in batch.non_tensor_batch.items():
                    if isinstance(val, np.ndarray) and len(val) > q_idx:
                        q_batch_dict[key] = val[q_idx:q_idx+1]

            q_batch = self._create_dataproto(q_batch_dict)

            start_idx = q_idx * num_rubrics
            end_idx = (q_idx + 1) * num_rubrics
            if rubrics_batch.batch is not None:
                q_rubrics_batch_tensors = rubrics_batch.batch[start_idx:end_idx]
            else:
                q_rubrics_batch_tensors = None

            q_rubrics_non_tensor = {}
            if rubrics_batch.non_tensor_batch:
                for key, value in rubrics_batch.non_tensor_batch.items():
                    if isinstance(value, (list, np.ndarray)):
                        q_rubrics_non_tensor[key] = value[start_idx:end_idx] if len(value) > end_idx else value[start_idx:]
                    else:
                        q_rubrics_non_tensor[key] = value

            q_rubrics_batch = DataProto(
                batch=q_rubrics_batch_tensors,
                non_tensor_batch=q_rubrics_non_tensor,
                meta_info=rubrics_batch.meta_info.copy() if rubrics_batch.meta_info else {}
            )
            
            if rubrics_output.batch is not None and "responses" in rubrics_output.batch:
                from tensordict import TensorDict
                q_responses = rubrics_output.batch["responses"][start_idx:end_idx]
                q_response_batch = TensorDict({"responses": q_responses}, batch_size=q_responses.shape[0])

                q_non_tensor_batch = {}
                if rubrics_output.non_tensor_batch:
                    for key, value in rubrics_output.non_tensor_batch.items():
                        if isinstance(value, (list, np.ndarray)):
                            q_non_tensor_batch[key] = value[start_idx:end_idx] if len(value) > end_idx else value[start_idx:]
                        else:
                            q_non_tensor_batch[key] = value
                
                q_rubrics_output = DataProto(
                    batch=q_response_batch,
                    non_tensor_batch=q_non_tensor_batch,
                    meta_info=rubrics_output.meta_info.copy() if rubrics_output.meta_info else {}
                )
            else:
                raise ValueError("")
            
            ans_start_idx = q_idx * num_answers
            ans_end_idx = (q_idx + 1) * num_answers
            
            if gen_batch_output.batch is not None and "responses" in gen_batch_output.batch:
                pass
            else:
                print("[message omitted]")

            reflect_answers_for_q = reflect_answers_per_question.get(q_idx)

            reward_config = {
                "similarity_weight": adv_config.get("similarity_weight", 0.25),
                "discrimination_weight": adv_config.get("discrimination_weight", 0.25),
                "diversity_weight": adv_config.get("diversity_weight", 0.25),
                "reflect_weight": adv_config.get("reflect_weight", 0.25),
                "similarity_penalty_weight": adv_config.get("similarity_penalty_weight", 0.0),
                "similarity_penalty_threshold": adv_config.get("similarity_penalty_threshold", 0.8),
                "enable_similarity": adv_config.get("enable_similarity", True),
                "enable_discrimination": adv_config.get("enable_discrimination", True),
                "enable_diversity": adv_config.get("enable_diversity", True),
                "enable_reflect": adv_config.get("enable_reflect", False),
            }

            full_batch = self._prepare_full_batch_for_rubrics_training(
                q_rubrics_batch, q_rubrics_output, num_rubrics,
                question, ground_truth_rubrics, generated_rubrics_for_q, generated_answers_for_q,
                reward_config=reward_config,
                reflect_answers_per_baseline=reflect_answers_for_q,
                question_idx=q_idx,
                golden_answer=golden_answer,
            )

            all_full_batches.append(full_batch)
            print("[message omitted]")

        del rubrics_batch, rubrics_output, gen_batch, gen_batch_output

        print("[message omitted]")

        merged_batch = self._merge_rubrics_training_batches(all_full_batches)

        self._warmup_sentence_transformer()

        extra_info_array = None
        if merged_batch.non_tensor_batch is not None:
            extra_info_array = merged_batch.non_tensor_batch.get("extra_info")

        if extra_info_array is not None:
            try:
                from llm_evaluator import LLMEvaluator
                from reward_calculator import _similarity_via_llm, parse_rubrics, score_answer_with_rubrics
                from concurrent.futures import ThreadPoolExecutor, as_completed

                evaluator = LLMEvaluator()
                total_samples = len(merged_batch)
                num_rubrics = config.data.get("num_rubrics", 4)
                print("[message omitted]")

                similarity_tasks = []
                discrimination_tasks = []

                if len(extra_info_array) > 0:
                    first_extra = extra_info_array[0]
                    print("[message omitted]")
                    if isinstance(first_extra, dict):
                        print(f"[DEBUG] adversarial_ground_truth_rubrics: {first_extra.get('adversarial_ground_truth_rubrics', 'KEY_NOT_FOUND')[:100] if first_extra.get('adversarial_ground_truth_rubrics') else 'EMPTY_OR_NONE'}")
                        print(f"[DEBUG] adversarial_policy_answers: {len(first_extra.get('adversarial_policy_answers', [])) if first_extra.get('adversarial_policy_answers') else 'EMPTY_OR_NONE'}")
                        print(f"[DEBUG] adversarial_reflect_answers_per_baseline: {list(first_extra.get('adversarial_reflect_answers_per_baseline', {}).keys()) if first_extra.get('adversarial_reflect_answers_per_baseline') else 'EMPTY_OR_NONE'}")
                        all_rubrics_list = first_extra.get('adversarial_all_rubrics_list', [])
                        print("[message omitted]")
                        if all_rubrics_list and len(all_rubrics_list) > 0:
                            print("[message omitted]")
                        print(f"[DEBUG] adversarial_enable_similarity: {first_extra.get('adversarial_enable_similarity', 'KEY_NOT_FOUND')}")
                        print(f"[DEBUG] adversarial_enable_diversity: {first_extra.get('adversarial_enable_diversity', 'KEY_NOT_FOUND')}")
                        print(f"[DEBUG] adversarial_enable_reflect: {first_extra.get('adversarial_enable_reflect', 'KEY_NOT_FOUND')}")
                
                for sample_idx in range(total_samples):
                    if sample_idx >= len(extra_info_array):
                        continue
                    
                    extra_info = extra_info_array[sample_idx]
                    if not isinstance(extra_info, dict):
                        continue
                    
                    if merged_batch.batch is not None and "responses" in merged_batch.batch:
                        response_ids = merged_batch.batch["responses"]
                        if isinstance(response_ids, torch.Tensor):
                            if response_ids.dim() == 2:
                                # shape: (batch_size, seq_len)
                                rubrics_text = self._safe_decode_token_ids(response_ids[sample_idx], sequence_idx=sample_idx)
                            elif response_ids.dim() == 1:
                                if sample_idx == 0:
                                    rubrics_text = self._safe_decode_token_ids(response_ids, sequence_idx=0)
                                else:
                                    continue
                            else:
                                print("[message omitted]")
                                continue
                        else:
                            continue
                    else:
                        continue
                    
                    ground_truth_rubrics = extra_info.get("adversarial_ground_truth_rubrics", "")
                    policy_answers = extra_info.get("adversarial_policy_answers", [])
                    question = extra_info.get("adversarial_conversation_text", extra_info.get("adversarial_question", ""))
                    enable_similarity = extra_info.get("adversarial_enable_similarity", True)
                    enable_discrimination = extra_info.get("adversarial_enable_discrimination", True)
                    enable_reflect = extra_info.get("adversarial_enable_reflect", False)
                    reflect_answers_per_baseline = extra_info.get("adversarial_reflect_answers_per_baseline", None)

                    if enable_similarity:
                        if ground_truth_rubrics and ground_truth_rubrics.strip():
                            similarity_tasks.append({
                                'sample_idx': sample_idx,
                                'rubrics_text': rubrics_text,
                                'ground_truth_rubrics': ground_truth_rubrics
                            })
                        else:
                            print("[message omitted]")

                    if enable_discrimination and policy_answers:
                        rubrics = parse_rubrics(rubrics_text)
                        if rubrics:
                            positive_sum = sum(r.points for r in rubrics if r.points > 0)
                            negative_sum = sum(r.points for r in rubrics if r.points < 0)
                            if positive_sum > 0 or negative_sum < 0:
                                for answer_idx, answer in enumerate(policy_answers):
                                    discrimination_tasks.append({
                                        'sample_idx': sample_idx,
                                        'answer_idx': answer_idx,
                                        'rubrics': rubrics,
                                        'rubrics_text': rubrics_text,
                                        'answer': answer,
                                        'question': question,
                                        'positive_sum': positive_sum,
                                        'negative_sum': negative_sum
                                    })

                adv_config = self.config.get("adversarial", {})
                max_concurrent_config = adv_config.get("max_concurrent_api_calls", 64)
                enable_diversity = adv_config.get("enable_diversity", True)
                enable_similarity_flag = adv_config.get("enable_similarity", True)
                enable_discrimination_flag = adv_config.get("enable_discrimination", True)
                enable_reflect_flag = adv_config.get("enable_reflect", False)
                enable_contrast_flag = False
                
                print("[message omitted]")
                overall_start_time = time.time()
                
                similarity_rewards = {}
                discrimination_scores = {}
                diversity_rewards = {}
                reflect_rewards = {}
                
                def compute_batch_similarity():
                    """English docstring."""
                    if not enable_similarity_flag or not similarity_tasks:
                        return {}
                    
                    print("[message omitted]")
                    t_start = time.time()
                    
                    from reward_calculator import _similarity_via_llm
                    results = {}
                    
                    max_concurrent = min(len(similarity_tasks), max_concurrent_config)
                    with ThreadPoolExecutor(max_workers=max_concurrent) as executor:
                        def evaluate_similarity(task):
                            try:
                                score, _ = _similarity_via_llm(
                                    text_a=task['rubrics_text'],
                                    text_b=task['ground_truth_rubrics']
                                )
                                if score is None:
                                    score = 0.0
                                return task['sample_idx'], score, None
                            except Exception as e:
                                return task['sample_idx'], 0.0, e
                        
                        future_to_task = {
                            executor.submit(evaluate_similarity, task): task
                            for task in similarity_tasks
                        }
                        for future in as_completed(future_to_task):
                            try:
                                sample_idx, score, error = future.result()
                                results[sample_idx] = score
                            except Exception as e:
                                task = future_to_task[future]
                                results[task['sample_idx']] = 0.0
                    
                    t_elapsed = time.time() - t_start
                    print("[message omitted]")
                    return results
                
                def compute_batch_discrimination():
                    """English docstring."""
                    if not enable_discrimination_flag or not discrimination_tasks:
                        return {}
                    
                    print("[message omitted]")
                    t_start = time.time()
                    
                    from reward_calculator import normalize_score_to_minus1_1
                    rubrics_scores_dict = {}
                    
                    max_concurrent = min(len(discrimination_tasks), max_concurrent_config)
                    with ThreadPoolExecutor(max_workers=max_concurrent) as executor:
                        def evaluate_discrimination(task):
                            try:
                                total_score, _ = score_answer_with_rubrics(
                                    answer=task['answer'],
                                    rubrics=task['rubrics'],
                                    llm_evaluator=evaluator,
                                    question=task['question']
                                )
                                normalized_score = normalize_score_to_minus1_1(
                                    total_score, task['positive_sum'], task['negative_sum']
                                )
                                return task['sample_idx'], task['answer_idx'], normalized_score, None
                            except Exception as e:
                                return task['sample_idx'], task['answer_idx'], 0.0, e
                        
                        future_to_task = {
                            executor.submit(evaluate_discrimination, task): task
                            for task in discrimination_tasks
                        }
                        for future in as_completed(future_to_task):
                            try:
                                sample_idx, answer_idx, normalized_score, error = future.result()
                                if sample_idx not in rubrics_scores_dict:
                                    rubrics_scores_dict[sample_idx] = {}
                                rubrics_scores_dict[sample_idx][answer_idx] = normalized_score
                            except Exception as e:
                                pass
                    
                    results = {}
                    for sample_idx, scores_dict in rubrics_scores_dict.items():
                        scores = [scores_dict.get(i, 0.0) for i in range(num_answers)]
                        if len(scores) >= 2:
                            results[sample_idx] = float(np.std(scores))
                        else:
                            results[sample_idx] = 0.0
                    
                    t_elapsed = time.time() - t_start
                    print("[message omitted]")
                    return results
                
                def compute_batch_diversity():
                    """English docstring."""
                    if not enable_diversity:
                        return {}
                    
                    print("[message omitted]")
                    t_start = time.time()
                    
                    all_rubrics_texts = []
                    diversity_sample_indices = []
                    for sample_idx in range(total_samples):
                        if sample_idx >= len(extra_info_array):
                            continue
                        
                        if merged_batch.batch is not None and "responses" in merged_batch.batch:
                            response_ids = merged_batch.batch["responses"]
                            if isinstance(response_ids, torch.Tensor) and response_ids.dim() >= 2:
                                rubrics_text = self._safe_decode_token_ids(response_ids[sample_idx], sequence_idx=sample_idx)
                                all_rubrics_texts.append(rubrics_text)
                                diversity_sample_indices.append(sample_idx)
                    
                    results = {}
                    if all_rubrics_texts:
                        from reward_calculator import batch_calculate_diversity_rewards
                        
                        diversity_rewards_list = batch_calculate_diversity_rewards(
                            rubrics_list=all_rubrics_texts
                        )
                        
                        for i, sample_idx in enumerate(diversity_sample_indices):
                            if i < len(diversity_rewards_list):
                                results[sample_idx] = diversity_rewards_list[i]
                    
                    t_elapsed = time.time() - t_start
                    print("[message omitted]")
                    return results
                
                def compute_batch_reflect():
                    """English docstring."""
                    if not enable_reflect_flag or not reflect_answers_per_question:
                        return {}
                    
                    print("[message omitted]")
                    t_start = time.time()
                    
                    batch_rubrics_list = []
                    batch_gt_rubrics_list = []
                    batch_baseline_answers = []
                    batch_reflect_answers = []
                    batch_questions = []
                    sample_idx_mapping = []
                    
                    for sample_idx in range(total_samples):
                        if sample_idx >= len(extra_info_array):
                            continue
                        
                        extra_info = extra_info_array[sample_idx]
                        if not isinstance(extra_info, dict):
                            continue
                        
                        rubrics_text = self._safe_decode_token_ids(
                            merged_batch.batch["responses"][sample_idx], sequence_idx=sample_idx
                        ) if merged_batch.batch is not None and "responses" in merged_batch.batch else ""
                        
                        ground_truth_rubrics = extra_info.get("adversarial_ground_truth_rubrics", "")
                        policy_answers = extra_info.get("adversarial_policy_answers", [])
                        conv_text = extra_info.get("adversarial_conversation_text", extra_info.get("adversarial_question", ""))
                        reflect_answers_per_baseline = extra_info.get("adversarial_reflect_answers_per_baseline", None)
                        
                        if not reflect_answers_per_baseline or not ground_truth_rubrics:
                            continue
                        
                        rubrics_idx = extra_info.get("rubrics_idx", sample_idx % num_rubrics)
                        reflect_answers_for_this_rubrics = {}
                        for baseline_idx, rubrics_dict in reflect_answers_per_baseline.items():
                            if isinstance(rubrics_dict, dict) and rubrics_idx in rubrics_dict:
                                reflect_answers_for_this_rubrics[baseline_idx] = rubrics_dict[rubrics_idx]
                        
                        if not reflect_answers_for_this_rubrics:
                            continue
                        
                        batch_rubrics_list.append(rubrics_text)
                        batch_gt_rubrics_list.append(rubrics_text)
                        batch_baseline_answers.append(policy_answers)
                        batch_reflect_answers.append(reflect_answers_for_this_rubrics)
                        batch_questions.append(conv_text)
                        sample_idx_mapping.append(sample_idx)
                    
                    results = {}
                    if batch_rubrics_list:
                        from reward_calculator import batch_calculate_reflect_rewards
                        
                        batch_reflect_rewards = batch_calculate_reflect_rewards(
                            rubrics_list=batch_rubrics_list,
                            ground_truth_rubrics_list=batch_gt_rubrics_list,
                            baseline_answers_list=batch_baseline_answers,
                            reflect_answers_per_rubrics=batch_reflect_answers,
                            llm_evaluator=evaluator,
                            questions=batch_questions,
                            max_concurrent=max_concurrent_config
                        )
                        
                        for i, sample_idx in enumerate(sample_idx_mapping):
                            if i < len(batch_reflect_rewards):
                                results[sample_idx] = batch_reflect_rewards[i]
                    
                    t_elapsed = time.time() - t_start
                    print("[message omitted]")
                    return results
                
                with ThreadPoolExecutor(max_workers=4) as executor:
                    future_similarity = executor.submit(compute_batch_similarity)
                    future_discrimination = executor.submit(compute_batch_discrimination)
                    future_diversity = executor.submit(compute_batch_diversity)
                    future_reflect = executor.submit(compute_batch_reflect)

                    similarity_rewards = future_similarity.result()
                    discrimination_scores = future_discrimination.result()
                    diversity_rewards = future_diversity.result()
                    reflect_rewards = future_reflect.result()
                
                overall_elapsed = time.time() - overall_start_time
                print("[message omitted]")
                print("[message omitted]")
                print("[message omitted]")
                print("[message omitted]")
                print("[message omitted]")
                print("[message omitted]")
                
                total_cached = 0
                for sample_idx in range(total_samples):
                    if sample_idx >= len(extra_info_array):
                        continue
                    
                    extra_info = extra_info_array[sample_idx]
                    if not isinstance(extra_info, dict):
                        extra_info = {}
                    
                    cached_details = {
                        'similarity_reward': similarity_rewards.get(sample_idx, 0.0),
                        'discrimination_reward': discrimination_scores.get(sample_idx, 0.0),
                        'diversity_reward': diversity_rewards.get(sample_idx, 0.0),
                        'reflect_reward': reflect_rewards.get(sample_idx, 0.0),
                    }
                    extra_info["cached_reward_details"] = cached_details
                    extra_info_array[sample_idx] = extra_info
                    total_cached += 1

                print("[message omitted]")
            except Exception as e:
                print("[message omitted]")
                import traceback
                traceback.print_exc()

        print("[message omitted]")
        t_reward_start = time.time()
        reward_tensor, reward_extra = compute_reward(merged_batch, self.reward_fn)
        t_reward_total = time.time() - t_reward_start
        avg_time_per_sample = t_reward_total / len(merged_batch) if len(merged_batch) > 0 else 0
        print("[message omitted]")
        merged_batch.batch["token_level_scores"] = reward_tensor
        merged_batch.batch["token_level_rewards"] = reward_tensor

        if reward_extra and isinstance(reward_extra, dict):
            sub_reward_keys = ['similarity_reward', 'discrimination_reward',
                              'diversity_reward', 'entropy_reward',
                              'reflect_reward']

            for key in sub_reward_keys:
                values = reward_extra.get(key, [])
                if values and isinstance(values, list) and len(values) > 0:
                    mean_value = float(np.mean(values))
                    metrics[f"{key}_mean"] = mean_value
                    print(f"[Metrics] {key}_mean = {mean_value:.4f} (from {len(values)} samples)")

        if reward_extra and isinstance(reward_extra, dict):
            total_rewards = reward_extra.get("score", [])

            if total_rewards and len(total_rewards) == len(merged_batch):
                all_sample_rewards = total_rewards
                print("[message omitted]")

                metrics["reward_mean"] = float(np.mean(all_sample_rewards))
                print("[message omitted]")
            else:
                if "attention_mask" in merged_batch.batch and "prompts" in merged_batch.batch:
                    prompt_length = merged_batch.batch["prompts"].shape[-1]
                    attention_mask = merged_batch.batch["attention_mask"]
                    response_mask = attention_mask[:, prompt_length:]
                    response_lens = response_mask.sum(dim=-1)

                    all_sample_rewards = []
                    for i in range(reward_tensor.size(0)):
                        last_idx = int(response_lens[i].item()) - 1
                        if last_idx >= 0 and last_idx < reward_tensor.size(1):
                            reward_value = reward_tensor[i, last_idx].item()
                            all_sample_rewards.append(reward_value)
                        else:
                            non_zero_mask = reward_tensor[i] != 0
                            if non_zero_mask.any():
                                reward_value = reward_tensor[i][non_zero_mask][-1].item()
                                all_sample_rewards.append(reward_value)
                            else:
                                all_sample_rewards.append(0.0)
                    print("[message omitted]")

                    metrics["reward_mean"] = float(np.mean(all_sample_rewards))
                    print("[message omitted]")
                else:
                    all_sample_rewards = [0.0] * len(merged_batch)
                    print("[message omitted]")

                    metrics["reward_mean"] = 0.0
                    print("[message omitted]")
        else:
            if "attention_mask" in merged_batch.batch and "prompts" in merged_batch.batch:
                prompt_length = merged_batch.batch["prompts"].shape[-1]
                attention_mask = merged_batch.batch["attention_mask"]
                response_mask = attention_mask[:, prompt_length:]
                response_lens = response_mask.sum(dim=-1)

                all_sample_rewards = []
                for i in range(reward_tensor.size(0)):
                    last_idx = int(response_lens[i].item()) - 1
                    if last_idx >= 0 and last_idx < reward_tensor.size(1):
                        reward_value = reward_tensor[i, last_idx].item()
                        all_sample_rewards.append(reward_value)
                    else:
                        non_zero_mask = reward_tensor[i] != 0
                        if non_zero_mask.any():
                            reward_value = reward_tensor[i][non_zero_mask][-1].item()
                            all_sample_rewards.append(reward_value)
                        else:
                            all_sample_rewards.append(0.0)
                print("[message omitted]")
            else:
                all_sample_rewards = [0.0] * len(merged_batch)
                print("[message omitted]")

            for q_idx in range(batch_size):
                start_idx = q_idx * num_rubrics
                end_idx = start_idx + num_rubrics
                q_sample_rewards = all_sample_rewards[start_idx:end_idx]
                if q_sample_rewards:
                    q_reward_mean = np.mean(q_sample_rewards)
                    print("[message omitted]")

            if all_sample_rewards:
                metrics["reward_mean"] = np.mean(all_sample_rewards)
                print("[message omitted]")

        self.actor_rollout_wg.set_adapter(RUBRICS_GEN_ADAPTER)
        print("[message omitted]")
        old_log_prob = self.actor_rollout_wg.compute_log_prob(merged_batch)
        merged_batch = merged_batch.union(old_log_prob)

        if config.actor_rollout_ref.actor.use_kl_loss:
            print("[message omitted]")
            ref_log_prob = self.actor_rollout_wg.compute_ref_log_prob(merged_batch)
            merged_batch = merged_batch.union(ref_log_prob)

        if merged_batch.non_tensor_batch and "uid" in merged_batch.non_tensor_batch:
            uids = merged_batch.non_tensor_batch["uid"]
            uid_groups = {}
            for i, uid in enumerate(uids):
                if uid not in uid_groups:
                    uid_groups[uid] = []
                uid_groups[uid].append(i)
            print("[message omitted]")
            for uid, indices in sorted(uid_groups.items()):
                print("[message omitted]")
                if len(indices) != num_rubrics:
                    print("[message omitted]")
        
        print("[message omitted]")
        merged_batch = compute_advantage(
            merged_batch,
            adv_estimator=config.algorithm.adv_estimator,
            gamma=config.algorithm.gamma,
            lam=config.algorithm.lam,
            num_repeat=num_rubrics,
            norm_adv_by_std_in_grpo=config.algorithm.get("norm_adv_by_std_in_grpo", True),
            config=config.algorithm,
        )

        if "attention_mask" in merged_batch.batch:
            if merged_batch.meta_info is None:
                merged_batch.meta_info = {}
            merged_batch.meta_info["global_token_num"] = torch.sum(
                merged_batch.batch["attention_mask"], dim=-1
            ).tolist()

        print("[message omitted]")

        adv_cfg_r = self.config.get("adversarial", {})
        rubrics_update_freq = max(1, int(adv_cfg_r.get("rubrics_update_freq", 1)))
        should_update_rubrics = (self.global_steps % rubrics_update_freq) == 0
        if should_update_rubrics:
            actor_output = self.actor_rollout_wg.update_actor(merged_batch)
            actor_metrics = reduce_metrics(actor_output.meta_info["metrics"])
            metrics.update(actor_metrics)
            metrics["rubrics/update_skipped_by_freq"] = 0.0
        else:
            print(
                "[message omitted]"
                f"rubrics_update_freq={rubrics_update_freq}"
            )
            metrics["rubrics/update_skipped_by_freq"] = 1.0

        print("[message omitted]")

        return metrics

    def _merge_training_batches(self, batch_list: list, uid_prefix: str = "merged") -> DataProto:
        """English docstring."""
        from tensordict import TensorDict

        if not batch_list:
            raise ValueError("")

        merged_tensors = {}
        first_batch = batch_list[0]
        pad_token_id = getattr(self.tokenizer, 'pad_token_id', 0)

        if first_batch.batch is not None:
            for key in first_batch.batch.keys():
                if key not in ["uid"]: 
                    tensor_list = [b.batch[key] for b in batch_list if b.batch is not None and key in b.batch]
                    if not tensor_list:
                        continue

                    if key in ["input_ids", "attention_mask", "position_ids", "prompts", "responses"]:
                        max_length = max(t.shape[-1] for t in tensor_list)

                        padded_tensors = []
                        for tensor in tensor_list:
                            current_length = tensor.shape[-1]
                            if current_length < max_length:
                                pad_size = max_length - current_length

                                if key == "input_ids":
                                    padding = torch.full((tensor.shape[0], pad_size), pad_token_id,
                                                         dtype=tensor.dtype, device=tensor.device)
                                    tensor = torch.cat([tensor, padding], dim=-1)
                                elif key == "attention_mask":
                                    padding = torch.zeros((tensor.shape[0], pad_size),
                                                          dtype=tensor.dtype, device=tensor.device)
                                    tensor = torch.cat([tensor, padding], dim=-1)
                                elif key == "position_ids":
                                    padding = torch.zeros((tensor.shape[0], pad_size),
                                                          dtype=tensor.dtype, device=tensor.device)
                                    tensor = torch.cat([tensor, padding], dim=-1)
                                elif key in ["prompts", "responses"]:
                                    padding = torch.full((tensor.shape[0], pad_size), pad_token_id,
                                                         dtype=tensor.dtype, device=tensor.device)
                                    tensor = torch.cat([tensor, padding], dim=-1)

                            padded_tensors.append(tensor)

                        merged_tensors[key] = torch.cat(padded_tensors, dim=0)

                        if key == "input_ids":
                            print(f"[Merge] {key} padded to max_length={max_length}")
                    else:
                        try:
                            merged_tensors[key] = torch.cat(tensor_list, dim=0)
                        except RuntimeError as e:
                            print("[message omitted]")
                            merged_tensors[key] = tensor_list[0]

        merged_non_tensors = {}
        if first_batch.non_tensor_batch is not None:
            for key in first_batch.non_tensor_batch.keys():
                if key not in ["uid", "extra_info"]:
                    array_list = [b.non_tensor_batch[key] for b in batch_list if b.non_tensor_batch is not None and key in b.non_tensor_batch]
                    if array_list:
                        try:
                            merged_non_tensors[key] = np.concatenate(array_list, axis=0)
                        except Exception as e:
                            merged_non_tensors[key] = array_list[0]

        merged_extra_info = []
        for batch in batch_list:
            if batch.non_tensor_batch is not None and "extra_info" in batch.non_tensor_batch:
                for extra_info in batch.non_tensor_batch["extra_info"]:
                    merged_extra_info.append(extra_info)
        merged_non_tensors["extra_info"] = np.array(merged_extra_info, dtype=object)

        total_batch_size = sum(len(b) for b in batch_list)
        uids = []
        can_reuse_original_uids = all(
            (b.non_tensor_batch is not None and "uid" in b.non_tensor_batch and len(b.non_tensor_batch["uid"]) == len(b))
            for b in batch_list
        )
        if can_reuse_original_uids:
            for batch_idx, batch in enumerate(batch_list):
                original_uids = batch.non_tensor_batch["uid"]
                for uid in original_uids:
                    uids.append(f"{uid_prefix}_acc{batch_idx}_{uid}")
        else:
            timestamp = int(time.time() * 1000)
            for batch_idx, batch in enumerate(batch_list):
                uid_for_batch = f"{uid_prefix}_{timestamp}_b{batch_idx}"
                batch_size = len(batch)
                for _ in range(batch_size):
                    uids.append(uid_for_batch)
        merged_non_tensors["uid"] = np.array(uids, dtype=object)

        merged_meta_info = {}
        for batch in batch_list:
            if batch.meta_info is not None:
                merged_meta_info.update(batch.meta_info)

        merged_batch = DataProto.from_dict(
            tensors=merged_tensors,
            non_tensors=merged_non_tensors,
            meta_info=merged_meta_info if merged_meta_info else {}
        )

        print("[message omitted]")

        return merged_batch

    def _merge_rubrics_training_batches(self, batch_list: list) -> DataProto:
        """English docstring."""
        return self._merge_training_batches(batch_list, uid_prefix="rubrics_gen")

    def _load_test_dataset(self, config) -> list:
        """English docstring."""
        import json
        import os
        
        eval_config = config.trainer.get("eval", {})
        test_file = eval_config.get("test_file")
        num_samples = eval_config.get("num_samples", 10)
        hf_split = eval_config.get("hf_split", "test")
        
        if not test_file:
            print("[message omitted]")
            return []

        test_data = []
        if os.path.exists(test_file):
            print("[message omitted]")
            with open(test_file, "r", encoding="utf-8") as f:
                test_data = json.load(f)
        else:
            ds_name = test_file
            if isinstance(ds_name, str) and ds_name.startswith("hf://"):
                ds_name = ds_name[len("hf://") :]

            try:
                from datasets import load_dataset
                print("[message omitted]")
                ds = load_dataset(ds_name, split=hf_split)

                for ex in ds:
                    problem = (
                        ex.get("problem")
                        or ex.get("question")
                        or ex.get("prompt")
                        or ex.get("input")
                        or ""
                    )

                    test_data.append(
                        {
                            "prompt": [{"role": "user", "content": problem}],
                            "rubrics": [],
                            "dataset_mode": "open_rubrics",
                            "data_source": ds_name,
                        }
                    )
            except Exception as e:
                print("[message omitted]")
                return []
        
        if num_samples > 0 and len(test_data) > num_samples:
            test_data = test_data[:num_samples]
        
        print("[message omitted]")
        
        return test_data
    
    def _format_golden_rubrics(self, rubrics_list: list) -> str:
        """English docstring."""
        formatted_lines = []
        for i, rubric in enumerate(rubrics_list, 1):
            criterion = rubric.get("criterion", "")
            points = rubric.get("points", 0)
            tags = rubric.get("tags", [])
            
            axis = "general"
            for tag in tags:
                if tag.startswith("axis:"):
                    axis = tag.split(":", 1)[1]
                    break
            
            formatted_lines.append(f"{i}. {criterion}")
            formatted_lines.append(f"   points: {points}")
            formatted_lines.append(f"   axis: {axis}")
            formatted_lines.append("")
        
        return "\n".join(formatted_lines)
    
    def _load_precomputed_guided_rubrics_jsonl(self, jsonl_path: str) -> Tuple[Dict[str, str], Dict[int, str]]:
        """English docstring."""
        by_prompt_id: Dict[str, str] = {}
        by_idx: Dict[int, str] = {}
        with open(jsonl_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                text = (rec.get("generated_rubrics") or "").strip()
                if "idx" in rec and rec["idx"] is not None:
                    try:
                        by_idx[int(rec["idx"])] = text
                    except (TypeError, ValueError):
                        pass
                pid = rec.get("prompt_id")
                if pid is not None:
                    by_prompt_id[str(pid).strip()] = text
        return by_prompt_id, by_idx

    def _align_precomputed_rubrics_to_items(
        self,
        test_data: List[Dict[str, Any]],
        by_prompt_id: Dict[str, str],
        by_idx: Dict[int, str],
    ) -> List[str]:
        out: List[str] = [""] * len(test_data)
        for i, item in enumerate(test_data):
            pid = item.get("prompt_id")
            if pid is not None:
                key = str(pid).strip()
                if key and key in by_prompt_id:
                    out[i] = by_prompt_id[key]
                    continue
            if i in by_idx:
                out[i] = by_idx[i]
        return out

    @staticmethod
    def _sanitize_prompt_messages(raw_messages: Any) -> List[Dict[str, str]]:
        out: List[Dict[str, str]] = []
        if not isinstance(raw_messages, list):
            return out
        for msg in raw_messages:
            if not isinstance(msg, dict):
                continue
            role = msg.get("role", "")
            content = msg.get("content", "")
            if role in ("user", "assistant", "system") and isinstance(content, str):
                out.append({"role": role, "content": content})
        return out

    def _run_evaluation(self, config, step: int) -> Dict[str, Any]:
        """English docstring."""
        import json
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        print(f"\n{'='*80}")
        print("[message omitted]")
        print(f"{'='*80}")
        
        test_data = self._load_test_dataset(config)
        if not test_data:
            print("[message omitted]")
            return {}
        
        self.actor_rollout_wg.set_adapter(POLICY_LLM_ADAPTER)
        
        eval_results = {
            "step": step,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "num_samples": len(test_data),
            "samples": [],
            "summary": {},
            "eval_mode": "golden_rubrics",
        }
        
        all_scores = []
        all_normalized_scores = []
        all_guided_scores = []
        all_guided_normalized_scores = []
        all_uplifts = []
        
        questions = []
        eval_prompt_messages_list: List[List[Dict[str, str]]] = []
        golden_rubrics_list = []
        for item in test_data:
            prompt_messages = item.get("prompt", [])
            sanitized_messages = self._sanitize_prompt_messages(prompt_messages)
            user_question = ""
            if sanitized_messages:
                for m in reversed(sanitized_messages):
                    if m.get("role") == "user":
                        user_question = m.get("content", "")
                        break
                if not user_question:
                    user_question = "\n\n".join(
                        m.get("content", "") for m in sanitized_messages if m.get("role") != "system"
                    )
            else:
                user_question = str(prompt_messages)
            question = user_question
            
            rubrics = item.get("rubrics", [])
            if not rubrics:
                gt_text = (item.get("ground_truth_rubrics") or "").strip()
                if gt_text:
                    from reward_calculator import parse_rubrics

                    for ri in parse_rubrics(gt_text):
                        tags = []
                        ax = getattr(ri, "axis", None) or "unknown"
                        if ax and ax != "unknown":
                            tags.append(f"axis:{ax}")
                        rubrics.append(
                            {
                                "criterion": ri.description,
                                "points": ri.points,
                                "tags": tags,
                            }
                        )
            golden_rubrics_text = self._format_golden_rubrics(rubrics)
            
            questions.append(question)
            eval_prompt_messages_list.append(sanitized_messages)
            golden_rubrics_list.append({
                "text": golden_rubrics_text,
                "rubrics": rubrics
            })
        
        eval_cfg = config.trainer.get("eval", {})
        guided_cfg = {}
        guided_enabled = False
        skip_baseline = False

        n_q = len(questions)
        if skip_baseline:
            print(
                "[message omitted]"
                ""
            )
            generated_answers = [""] * n_q
        else:
            print(
                "[message omitted]"
                ""
            )
            batch = self._prepare_evaluation_batch(
                questions,
                config,
                prompt_messages_list=eval_prompt_messages_list,
            )
            print("[message omitted]")
            gen_output = self.actor_rollout_wg.generate_sequences(batch)

            generated_answers = []
            if gen_output.batch is not None and "responses" in gen_output.batch:
                response_ids = gen_output.batch["responses"]
                for i in range(min(len(questions), response_ids.size(0))):
                    answer = self._safe_decode_token_ids(response_ids[i], sequence_idx=i)
                    generated_answers.append(answer)
            else:
                generated_answers = [""] * len(questions)

            print("[message omitted]")

        generated_rubrics_texts = [""] * len(questions)
        guided_answers = [""] * len(questions)

        if guided_enabled:
            if skip_baseline:
                print(
                    "[message omitted]"
                    ""
                )
            else:
                print(
                    "[message omitted]"
                    ""
                )
            precomputed = guided_cfg.get("precomputed_rubrics_jsonl")
            precomputed_path = (str(precomputed).strip() if precomputed is not None else "")
            used_precomputed = False
            if precomputed_path and os.path.isfile(precomputed_path):
                try:
                    by_pid, by_idx = self._load_precomputed_guided_rubrics_jsonl(precomputed_path)
                    generated_rubrics_texts = self._align_precomputed_rubrics_to_items(
                        test_data, by_pid, by_idx
                    )
                    used_precomputed = True
                    n_ok = sum(1 for t in generated_rubrics_texts if (t or "").strip())
                    print(
                        ""
                        f"({precomputed_path})"
                    )
                except Exception as e:  # noqa: BLE001
                    print("[message omitted]")
                    generated_rubrics_texts = [""] * len(questions)

            if not used_precomputed:
                print("[message omitted]")
                rubrics_model_path = guided_cfg.get("rubrics_model_path") or config.actor_rollout_ref.model.path
                rubrics_served_model = guided_cfg.get("served_model_name")
                if not rubrics_served_model:
                    rubrics_served_model = os.path.basename(str(rubrics_model_path))

                rt_cfg = RubricsVLLMConfig(
                    model_path=str(rubrics_model_path),
                    served_model_name=str(rubrics_served_model),
                    host=str(guided_cfg.get("vllm_host", "127.0.0.1")),
                    port=int(guided_cfg.get("vllm_port", 18000)),
                    gpu_memory_utilization=float(guided_cfg.get("vllm_gpu_memory_utilization", 0.75)),
                    start_vllm=bool(guided_cfg.get("start_vllm", True)),
                    keep_user_site=bool(guided_cfg.get("vllm_keep_user_site", False)),
                    trust_remote_code=bool(guided_cfg.get("vllm_trust_remote_code", False)),
                    timeout_s=float(guided_cfg.get("vllm_timeout_s", 600.0)),
                    request_timeout_s=float(guided_cfg.get("request_timeout_s", 180.0)),
                    max_tokens=int(guided_cfg.get("rubrics_max_tokens", 1024)),
                    temperature=float(guided_cfg.get("rubrics_temperature", 0.2)),
                    max_workers=int(guided_cfg.get("rubrics_concurrency", 16)),
                    api_key=str(guided_cfg.get("rubrics_api_key", "EMPTY")),
                    progress_bar=bool(guided_cfg.get("rubrics_progress_bar", False)),
                )
                runtime = RubricsVLLMRuntime(rt_cfg)
                try:
                    runtime.start()
                    generated_rubrics_texts = runtime.generate_rubrics_batch(questions)
                except Exception as e:
                    print("[message omitted]")
                    guided_enabled = False
                finally:
                    runtime.close()

            if guided_enabled:
                guided_batch = self._prepare_evaluation_batch(
                    questions,
                    config,
                    prompt_messages_list=eval_prompt_messages_list,
                    guided_rubrics_texts=generated_rubrics_texts,
                )
                print("[message omitted]")
                guided_output = self.actor_rollout_wg.generate_sequences(guided_batch)
                if guided_output.batch is not None and "responses" in guided_output.batch:
                    response_ids = guided_output.batch["responses"]
                    for i in range(min(len(questions), response_ids.size(0))):
                        guided_answers[i] = self._safe_decode_token_ids(response_ids[i], sequence_idx=i)
                print("[message omitted]")
        
        try:
                from llm_evaluator import LLMEvaluator
                from reward_calculator import (
                    score_answer_with_rubrics,
                    normalize_score_by_max_possible,
                    RubricItem,
                )
                
                evaluator = LLMEvaluator()
                
                adv_config = config.get("adversarial", {})
                max_concurrent = adv_config.get("max_concurrent_api_calls", 64)
                
                def convert_rubrics_to_items(rubrics_raw: list) -> list:
                    """English docstring."""
                    rubric_items = []
                    for i, r in enumerate(rubrics_raw):
                        axis = "general"
                        tags = r.get("tags", [])
                        for tag in tags:
                            if isinstance(tag, str) and tag.startswith("axis:"):
                                axis = tag.split(":", 1)[1]
                                break
                        
                        rubric_item = RubricItem(
                            number=i + 1,
                            description=r.get("criterion", ""),
                            points=r.get("points", 0),
                            axis=axis
                        )
                        rubric_items.append(rubric_item)
                    return rubric_items
                
                eval_tasks = []
                for i, (question, answer, golden_rubrics) in enumerate(zip(questions, generated_answers, golden_rubrics_list)):
                    positive_sum = sum(r["points"] for r in golden_rubrics["rubrics"] if r["points"] > 0)
                    negative_sum = sum(r["points"] for r in golden_rubrics["rubrics"] if r["points"] < 0)
                    
                    rubric_items = convert_rubrics_to_items(golden_rubrics["rubrics"])
                    
                    eval_tasks.append({
                        "idx": i,
                        "question": question,
                        "answer": answer,
                        "guided_answer": guided_answers[i] if (guided_enabled and i < len(guided_answers)) else "",
                        "generated_rubrics": generated_rubrics_texts[i] if i < len(generated_rubrics_texts) else "",
                        "rubric_items": rubric_items,
                        "rubrics_raw": golden_rubrics["rubrics"],
                        "positive_sum": positive_sum,
                        "negative_sum": negative_sum,
                        "skip_baseline_scoring": skip_baseline,
                    })
                
                guided_note = (
                    ""
                    if skip_baseline
                    else ("" if guided_enabled else "")
                )
                print(
                    "[message omitted]"
                    ""
                    ""
                )
                
                def evaluate_single(task):
                    """English docstring."""
                    try:
                        if task.get("skip_baseline_scoring"):
                            total_score: Optional[float] = None
                            normalized_score: Optional[float] = None
                            details = {}
                        else:
                            total_score, details = score_answer_with_rubrics(
                                answer=task["answer"],
                                rubrics=task["rubric_items"],
                                llm_evaluator=evaluator,
                                question=task["question"],
                            )
                            normalized_score = normalize_score_by_max_possible(
                                total_score,
                                task["positive_sum"],
                            )
                        guided_total_score = None
                        guided_normalized_score = None
                        guided_details = None
                        if task.get("guided_answer", "").strip():
                            guided_total_score, guided_details = score_answer_with_rubrics(
                                answer=task["guided_answer"],
                                rubrics=task["rubric_items"],
                                llm_evaluator=evaluator,
                                question=task["question"],
                            )
                            guided_normalized_score = normalize_score_by_max_possible(
                                guided_total_score,
                                task["positive_sum"],
                            )

                        return (
                            task["idx"],
                            total_score,
                            normalized_score,
                            details,
                            guided_total_score,
                            guided_normalized_score,
                            guided_details,
                            None,
                        )
                    except Exception as e:
                        return task["idx"], 0.0, 0.0, {}, None, None, None, e
                
                sample_results = [None] * len(eval_tasks)
                
                with ThreadPoolExecutor(max_workers=min(len(eval_tasks), max_concurrent)) as executor:
                    future_to_task = {executor.submit(evaluate_single, task): task for task in eval_tasks}
                    
                    for future in as_completed(future_to_task):
                        try:
                            (
                                idx,
                                total_score,
                                normalized_score,
                                details,
                                guided_total_score,
                                guided_normalized_score,
                                guided_details,
                                error,
                            ) = future.result()
                            task = future_to_task[future]
                            
                            if error:
                                print("[message omitted]")
                            
                            sample_results[idx] = {
                                "idx": idx,
                                "question": task["question"][:200] + "..." if len(task["question"]) > 200 else task["question"],
                                "answer": task["answer"][:500] + "..." if len(task["answer"]) > 500 else task["answer"],
                                "guided_answer": (task["guided_answer"][:500] + "...") if len(task["guided_answer"]) > 500 else task["guided_answer"],
                                "generated_rubrics": task["generated_rubrics"],
                                "golden_rubrics": task["rubrics_raw"],
                                "total_score": total_score,
                                "normalized_score": normalized_score,
                                "details": details,
                                "guided_total_score": guided_total_score,
                                "guided_normalized_score": guided_normalized_score,
                                "guided_details": guided_details,
                                "normalized_uplift": (
                                    (guided_normalized_score - normalized_score)
                                    if (
                                        guided_normalized_score is not None
                                        and normalized_score is not None
                                    )
                                    else None
                                ),
                            }
                            
                            if normalized_score is not None:
                                all_scores.append(float(total_score) if total_score is not None else 0.0)
                                all_normalized_scores.append(float(normalized_score))
                            if guided_normalized_score is not None:
                                all_guided_scores.append(float(guided_total_score or 0.0))
                                all_guided_normalized_scores.append(float(guided_normalized_score))
                            if (
                                guided_normalized_score is not None
                                and normalized_score is not None
                            ):
                                all_uplifts.append(float(guided_normalized_score - normalized_score))
                            
                        except Exception as e:
                            task = future_to_task[future]
                            print("[message omitted]")
                            sample_results[task["idx"]] = {
                                "idx": task["idx"],
                                "question": task["question"][:200],
                                "answer": task["answer"][:500],
                                "guided_answer": task.get("guided_answer", "")[:500],
                                "generated_rubrics": task.get("generated_rubrics", ""),
                                "golden_rubrics": task["rubrics_raw"],
                                "total_score": 0.0,
                                "normalized_score": 0.0,
                                "error": str(e)
                            }
                
                eval_results["samples"] = [r for r in sample_results if r is not None]
            
        except Exception as e:
            print("[message omitted]")
            import traceback
            traceback.print_exc()
            return {"error": str(e)}
        
        eval_results["baseline_skipped"] = skip_baseline

        if all_scores:
            if eval_results.get("eval_mode") == "math":
                num_correct = int(sum(all_scores))
                accuracy = float(np.mean(all_scores))
                eval_results["summary"] = {
                    "accuracy": accuracy,
                    "num_correct": num_correct,
                    "total": len(all_scores),
                    "mean_score": accuracy,
                    "mean_normalized_score": accuracy,
                    "std_score": 0.0,
                    "std_normalized_score": 0.0,
                }
                print("[message omitted]")
                print("[message omitted]")
                print("[message omitted]")
                print("[message omitted]")
            else:
                eval_results["summary"] = {
                    "mean_score": float(np.mean(all_scores)),
                    "std_score": float(np.std(all_scores)),
                    "min_score": float(np.min(all_scores)),
                    "max_score": float(np.max(all_scores)),
                    "mean_normalized_score": float(np.mean(all_normalized_scores)),
                    "std_normalized_score": float(np.std(all_normalized_scores)),
                }
                if all_guided_normalized_scores:
                    guided_extra: Dict[str, float] = {
                        "guided_mean_score": float(np.mean(all_guided_scores)),
                        "guided_mean_normalized_score": float(np.mean(all_guided_normalized_scores)),
                        "guided_std_normalized_score": float(np.std(all_guided_normalized_scores)),
                    }
                    if all_uplifts:
                        guided_extra["mean_normalized_uplift"] = float(np.mean(all_uplifts))
                    eval_results["summary"].update(guided_extra)
                print("[message omitted]")
                print("[message omitted]")
                print("[message omitted]")
                print("[message omitted]")
                print("[message omitted]")
                if all_guided_normalized_scores:
                    print(
                        ""
                        f"{eval_results['summary']['guided_mean_normalized_score']:.4f}"
                    )
                    if all_uplifts:
                        print(
                            ""
                            f"{eval_results['summary']['mean_normalized_uplift']:.4f}"
                        )
        elif skip_baseline and eval_results.get("eval_mode") != "math" and all_guided_normalized_scores:
            eval_results["summary"] = {
                "guided_mean_score": float(np.mean(all_guided_scores)),
                "guided_mean_normalized_score": float(np.mean(all_guided_normalized_scores)),
                "guided_std_normalized_score": float(np.std(all_guided_normalized_scores)),
                "num_samples_scored": len(all_guided_normalized_scores),
            }
            print("[message omitted]")
            print("[message omitted]")
            print(
                "[message omitted]"
                f"{eval_results['summary']['guided_mean_normalized_score']:.4f}"
            )
        
        self._save_evaluation_results(config, step, eval_results)
        
        return eval_results
    
    def _prepare_evaluation_batch(
        self,
        questions: list,
        config,
        prompt_messages_list: Optional[List[List[Dict[str, str]]]] = None,
        guided_rubrics_texts: Optional[List[str]] = None,
    ) -> DataProto:
        """English docstring."""
        prompts = []
        for i, question in enumerate(questions):
            raw_messages = []
            if prompt_messages_list is not None and i < len(prompt_messages_list):
                raw_messages = prompt_messages_list[i] or []

            messages = []
            if raw_messages:
                has_system = any(m.get("role") == "system" for m in raw_messages)
                if not has_system:
                    messages.append({"role": "system", "content": POLICY_LLM_SYSTEM_PROMPT})
                messages.extend(raw_messages)
            else:
                messages = [
                    {"role": "system", "content": POLICY_LLM_SYSTEM_PROMPT},
                    {"role": "user", "content": question},
                ]

            if guided_rubrics_texts is not None and i < len(guided_rubrics_texts):
                rubrics_text = (guided_rubrics_texts[i] or "").strip()
                if rubrics_text:
                    merged = False
                    for j in range(len(messages) - 1, -1, -1):
                        if messages[j].get("role") == "user":
                            messages[j]["content"] = RubricsVLLMRuntime.build_rubric_guided_query(
                                messages[j].get("content", ""),
                                rubrics_text,
                            )
                            merged = True
                            break
                    if not merged:
                        messages.append(
                            {
                                "role": "user",
                                "content": RubricsVLLMRuntime.build_rubric_guided_query(
                                    question, rubrics_text
                                ),
                            }
                        )

            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False
            )
            prompts.append(prompt)
        
        n_gpus = config.trainer.n_gpus_per_node
        if len(prompts) % n_gpus != 0:
            padded_size = ((len(prompts) + n_gpus - 1) // n_gpus) * n_gpus
            prompts.extend([prompts[-1]] * (padded_size - len(prompts)))
        
        encoded = self.tokenizer(
            prompts,
            padding=True,
            truncation=False,
            return_tensors="pt"
        )
        
        attention_mask = encoded["attention_mask"]
        position_ids = attention_mask.long().cumsum(dim=1) - 1
        position_ids = position_ids.clamp(min=0)
        
        batch = DataProto.from_dict(
            tensors={
                "input_ids": encoded["input_ids"],
                "attention_mask": encoded["attention_mask"],
                "position_ids": position_ids
            },
            non_tensors={
                "prompt": np.array(prompts, dtype=object)
            },
            meta_info={
                "n": 1,
                "do_sample": False,
                "temperature": 0.0,
                "top_p": 1.0,
            }
        )
        
        return batch
    
    def _save_evaluation_results(self, config, step: int, results: Dict[str, Any]):
        """English docstring."""
        import json
        
        eval_log_dir = os.path.join(config.trainer.default_local_dir, "eval_logs")
        os.makedirs(eval_log_dir, exist_ok=True)
        
        eval_file = os.path.join(eval_log_dir, f"eval_step_{step:06d}.json")
        
        with open(eval_file, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        print("[message omitted]")
        
        summary_file = os.path.join(eval_log_dir, "eval_summary.jsonl")
        summary_record = {
            "step": step,
            "timestamp": results.get("timestamp", ""),
            "num_samples": results.get("num_samples", 0),
            **results.get("summary", {})
        }
        
        with open(summary_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(summary_record, ensure_ascii=False) + "\n")
        
        print("[message omitted]")
        
        if self.swanlab_run is not None and results.get("summary"):
            try:
                import swanlab
                summary = results["summary"]
                swanlab_metrics = {
                    "eval/mean_score": summary.get("mean_score", 0),
                    "eval/mean_normalized_score": summary.get("mean_normalized_score", 0),
                    "eval/std_score": summary.get("std_score", 0),
                }
                if results.get("eval_mode") == "math":
                    swanlab_metrics["eval/accuracy"] = summary.get("accuracy", 0)
                    swanlab_metrics["eval/num_correct"] = summary.get("num_correct", 0)
                swanlab.log(swanlab_metrics, step=step)
                print("[message omitted]")
            except Exception as e:
                print("[message omitted]")

