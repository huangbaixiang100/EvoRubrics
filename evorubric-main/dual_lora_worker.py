"""EvoRubrics training module."""
import asyncio
import datetime
import functools
import json
import logging
import os
import warnings
from collections import OrderedDict
from dataclasses import asdict
from typing import Any, Dict, Optional
import numpy as np
import torch
import torch.distributed as dist
from omegaconf import DictConfig, OmegaConf, open_dict
from peft import LoraConfig, TaskType, get_peft_model, get_peft_model_state_dict, PeftModel
from peft.utils.save_and_load import load_peft_weights
from safetensors import safe_open
from safetensors.torch import save_file
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.api import FullStateDictConfig, ShardedStateDictConfig, StateDictType
try:
    from torch.distributed.tensor import DTensor
except ImportError:
    from torch.distributed._tensor import DTensor
from verl import DataProto
from verl.single_controller.base import Worker
from verl.single_controller.base.decorator import Dispatch, register, make_nd_compute_dataproto_dispatch_fn
from verl.utils import hf_tokenizer, hf_processor
from verl.utils.checkpoint.fsdp_checkpoint_manager import FSDPCheckpointManager
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.device import get_device_id, get_device_name, get_nccl_backend, get_torch_device, set_expandable_segments
from verl.utils.flops_counter import FlopsCounter
from verl.utils.fs import copy_to_local
from verl.utils.fsdp_utils import (
    fsdp_version,
    get_fsdp_wrap_policy,
    get_init_weight_context_manager,
    init_fn,
    layered_summon_lora_params,
    load_fsdp_model_to_gpu,
    load_fsdp_optimizer,
    offload_fsdp_model_to_cpu,
    offload_fsdp_optimizer,
    replace_lora_wrapper,
    collect_lora_params,
)
from verl.utils.import_utils import import_external_libs
from verl.utils.memory_utils import aggressive_empty_cache
from verl.utils.model import compute_position_id_with_mask, convert_weight_keys
from verl.utils.profiler import DistProfiler, DistProfilerExtension, ProfilerConfig, log_gpu_memory_usage
from verl.utils.py_functional import convert_to_regular_types
from verl.utils.ray_utils import get_event_loop
from verl.utils.vllm import TensorLoRARequest, VLLMHijack, is_version_ge
from verl.workers.config import FSDPEngineConfig, HFModelConfig, RolloutConfig
from verl.workers.config.optimizer import build_optimizer
from verl.workers.rollout import get_rollout_class
from verl.workers.sharding_manager.fsdp_ulysses import FSDPUlyssesShardingManager
from verl.workers.fsdp_workers import ActorRolloutRefWorker, create_device_mesh, get_sharding_strategy

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

device_name = get_device_name()

POLICY_LLM_ADAPTER = "policy_llm"
RUBRICS_GEN_ADAPTER = "rubrics_generator"
POLICY_LLM_LORA_ID = 1
RUBRICS_GEN_LORA_ID = 2


class DualLoRAActorRolloutRefWorker(Worker, DistProfilerExtension):
    """English docstring."""
    
    def __init__(self, config: DictConfig, role: str, **kwargs):
        Worker.__init__(self)
        
        self.config = config
        
        if not torch.distributed.is_initialized():
            rank = int(os.environ.get("RANK", 0))
            world_size = int(os.environ.get("WORLD_SIZE", 1))
            torch.distributed.init_process_group(
                backend=f"cpu:gloo,{get_device_name()}:{get_nccl_backend()}",
                rank=rank,
                world_size=world_size,
                timeout=datetime.timedelta(seconds=self.config.get("nccl_timeout", 600)),
                init_method=os.environ.get("DIST_INIT_METHOD", None),
            )
        
        world_size = torch.distributed.get_world_size()
        self.device_mesh = create_device_mesh(
            world_size=world_size, 
            fsdp_size=self.config.actor.fsdp_config.fsdp_size
        )
        
        # Ulysses sequence parallel
        self.ulysses_device_mesh = None
        self.ulysses_sequence_parallel_size = self.config.actor.get("ulysses_sequence_parallel_size", 1)
        dp = world_size // self.ulysses_sequence_parallel_size
        if self.ulysses_sequence_parallel_size > 1:
            self.ulysses_device_mesh = init_device_mesh(
                device_name, mesh_shape=(dp, self.ulysses_sequence_parallel_size), mesh_dim_names=["dp", "sp"]
            )
        
        # Dispatch info
        if self.ulysses_device_mesh is not None:
            is_collect = self.ulysses_device_mesh["sp"].get_local_rank() == 0
            self._register_dispatch_collect_info(
                "actor", dp_rank=self.ulysses_device_mesh["dp"].get_local_rank(), is_collect=is_collect
            )
        else:
            self._register_dispatch_collect_info("actor", dp_rank=self.rank, is_collect=True)
        
        self.ulysses_sharding_manager = FSDPUlyssesShardingManager(self.ulysses_device_mesh)
        
        self._lora_rank = self.config.model.get("lora_rank", 8)
        self._lora_alpha = self.config.model.get("lora_alpha", 16)
        self._is_dual_lora = True
        
        self.role = role
        assert self.role in ["actor", "rollout", "ref", "actor_rollout", "actor_rollout_ref"]
        
        self._is_actor = self.role in ["actor", "actor_rollout", "actor_rollout_ref"]
        self._is_rollout = self.role in ["rollout", "actor_rollout", "actor_rollout_ref"]
        self._is_ref = self.role in ["ref", "actor_rollout_ref"]
        self.use_orig_params = self.config.actor.fsdp_config.get("use_orig_params", False)
        
        # Profiler
        omega_profiler_config = config.actor.get("profiler", {})
        profiler_config = omega_conf_to_dataclass(omega_profiler_config, dataclass_type=ProfilerConfig)
        tool_config = None
        DistProfilerExtension.__init__(
            self, DistProfiler(rank=self.rank, config=profiler_config, tool_config=tool_config)
        )
        
        self._is_offload_param = self.config.actor.fsdp_config.get("param_offload", False)
        self._is_offload_optimizer = self.config.actor.fsdp_config.get("optimizer_offload", False)
        
        self.current_adapter = POLICY_LLM_ADAPTER
        self._last_synced_adapter = None
        self._force_sync_required = True
        
        self.policy_optimizer = None
        self.rubrics_optimizer = None
        self.policy_lr_scheduler = None
        self.rubrics_lr_scheduler = None
        
        # Normalize config
        if self._is_actor:
            self.config.actor.ppo_mini_batch_size *= self.config.rollout.n
            dp_size = self.device_mesh.size() // self.ulysses_sequence_parallel_size
            if dp_size > 0:
                self.config.actor.ppo_mini_batch_size = max(1, self.config.actor.ppo_mini_batch_size // dp_size)
            else:
                self.config.actor.ppo_mini_batch_size = max(1, self.config.actor.ppo_mini_batch_size)
            
            if self.config.actor.ppo_micro_batch_size_per_gpu is not None:
                assert self.config.actor.ppo_mini_batch_size % self.config.actor.ppo_micro_batch_size_per_gpu == 0

    def _adapter_weights_have_named_keys(self, adapter_path: str, adapter_name: str) -> bool:
        """English docstring."""
        safetensors_path = os.path.join(adapter_path, "adapter_model.safetensors")
        if os.path.isfile(safetensors_path):
            with safe_open(safetensors_path, framework="pt", device="cpu") as f:
                for key in f.keys():
                    if f".{adapter_name}." in key or key.endswith(f".{adapter_name}.weight"):
                        return True
            return False

        try:
            state_dict = load_peft_weights(adapter_path, device="cpu")
        except Exception:
            return False
        return any(
            f".{adapter_name}." in key or key.endswith(f".{adapter_name}.weight")
            for key in state_dict.keys()
        )
    
    def _detect_checkpoint_format(self, adapter_path: str) -> str:
        """English docstring."""
        safetensors_path = os.path.join(adapter_path, "adapter_model.safetensors")
        if os.path.isfile(safetensors_path):
            with safe_open(safetensors_path, framework="pt", device="cpu") as f:
                for key in f.keys():
                    if ".lora_A." in key or ".lora_B." in key:
                        parts = key.split(".")
                        for i, part in enumerate(parts):
                            if part in ["lora_A", "lora_B"]:
                                if i + 1 < len(parts) and parts[i + 1] == "weight":
                                    return "standard"
                                else:
                                    return "named"
        return "standard"
    
    def _load_checkpoint_with_key_conversion(
        self, 
        peft_model: PeftModel, 
        adapter_path: str, 
        target_adapter_name: str,
        source_adapter_name: str = "default"
    ) -> int:
        """English docstring."""
        safetensors_path = os.path.join(adapter_path, "adapter_model.safetensors")
        
        loaded_weights = {}
        with safe_open(safetensors_path, framework="pt", device="cpu") as f:
            for key in f.keys():
                loaded_weights[key] = f.get_tensor(key)
        
        converted_weights = {}
        for key, tensor in loaded_weights.items():
            new_key = key
            new_key = new_key.replace(f".lora_A.{source_adapter_name}.", ".lora_A.")
            new_key = new_key.replace(f".lora_B.{source_adapter_name}.", ".lora_B.")
            converted_weights[new_key] = tensor
        
        if self.rank == 0:
            sample_keys = list(converted_weights.keys())[:3]
            print(f"[DualLoRA] Converted weight keys (showing first 3):")
            for k in sample_keys:
                print(f"  {k}")
        
        self._load_adapter_state_dict_direct(peft_model, converted_weights, target_adapter_name)
        
        return len(converted_weights)
    
    def _rename_adapter_in_parameters(self, peft_model: PeftModel, old_name: str, new_name: str):
        """English docstring."""
        renamed_count = 0
        
        def rename_in_module(module):
            nonlocal renamed_count
            for attr_name in ['lora_A', 'lora_B']:
                if hasattr(module, attr_name):
                    attr = getattr(module, attr_name)
                    if isinstance(attr, torch.nn.ModuleDict) and old_name in attr:
                        attr[new_name] = attr.pop(old_name)
                        renamed_count += 1
            
            for child in module.children():
                rename_in_module(child)
        
        rename_in_module(peft_model)
        
        if self.rank == 0 and renamed_count > 0:
            print(f"[DualLoRA] Renamed {renamed_count} adapter layers from '{old_name}' to '{new_name}'")

    def _strip_adapter_name_from_lora_keys(
        self, state_dict: Dict[str, torch.Tensor], adapter_name: str
    ) -> Dict[str, torch.Tensor]:
        """English docstring."""
        adapter_token = f".{adapter_name}."
        adapter_suffix = f".{adapter_name}"
        cleaned_state_dict = {}
        for key, value in state_dict.items():
            if "lora" in key.lower() and adapter_token in key:
                cleaned_state_dict[key.replace(adapter_token, ".")] = value
            elif "lora" in key.lower() and key.endswith(adapter_suffix):
                cleaned_state_dict[key[: -len(adapter_suffix)]] = value
            else:
                cleaned_state_dict[key] = value
        return cleaned_state_dict

    def _attach_adapter_name_to_lora_keys(
        self, state_dict: Dict[str, torch.Tensor], adapter_name: str
    ) -> Dict[str, torch.Tensor]:
        """English docstring."""
        marker_replacements = (
            ("lora_A", f"lora_A.{adapter_name}"),
            ("lora_B", f"lora_B.{adapter_name}"),
            ("lora_embedding_A", f"lora_embedding_A.{adapter_name}"),
            ("lora_embedding_B", f"lora_embedding_B.{adapter_name}"),
        )
        converted = {}
        for key, value in state_dict.items():
            new_key = key
            for src, dst in marker_replacements:
                marker = f".{src}."
                if marker in new_key and f".{dst}." not in new_key:
                    new_key = new_key.replace(marker, f".{dst}.")
            converted[new_key] = value
        return converted

    def _load_adapter_state_dict_direct(
        self, peft_model: PeftModel, state_dict: Dict[str, torch.Tensor], adapter_name: str
    ) -> None:
        """English docstring."""
        adapted_state_dict = self._attach_adapter_name_to_lora_keys(state_dict, adapter_name)
        load_result = peft_model.load_state_dict(adapted_state_dict, strict=False)

        if self.rank != 0:
            return

        adapter_token = f".{adapter_name}."
        lora_missing = [
            key
            for key in load_result.missing_keys
            if "lora" in key.lower() and adapter_token in key
        ]
        lora_unexpected = [
            key
            for key in load_result.unexpected_keys
            if "lora" in key.lower()
        ]
        if lora_missing:
            warnings.warn(f"[DualLoRA] Loading adapter '{adapter_name}' missing keys: {lora_missing}")
        if lora_unexpected:
            warnings.warn(f"[DualLoRA] Loading adapter '{adapter_name}' unexpected keys: {lora_unexpected}")

    def _load_adapter_state_dict(
        self, peft_model: PeftModel, adapter_path: str, adapter_name: str, strip_named_keys: bool
    ) -> None:
        """English docstring."""
        state_dict = load_peft_weights(adapter_path, device="cpu")
        if strip_named_keys:
            state_dict = self._strip_adapter_name_from_lora_keys(state_dict, adapter_name)
        self._load_adapter_state_dict_direct(peft_model, state_dict, adapter_name)
    
    def _build_dual_lora_model(
        self,
        model_path,
        fsdp_config: FSDPEngineConfig,
        optim_config,
        override_model_config,
        use_remove_padding=False,
        enable_gradient_checkpointing=False,
        trust_remote_code=False,
    ):
        """English docstring."""
        from torch.distributed.fsdp import CPUOffload, MixedPrecision
        from transformers import AutoConfig, AutoModelForCausalLM
        from verl.utils.model import get_generation_config, print_model_size, update_model_config
        from verl.utils.torch_dtypes import PrecisionType
        
        log_gpu_memory_usage("Before init dual LoRA model", logger=logger)
        local_path = model_path
        
        self.tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
        self.processor = hf_processor(local_path, trust_remote_code=trust_remote_code)
        
        torch_dtype = fsdp_config.get("model_dtype", None)
        if torch_dtype is None:
            torch_dtype = torch.float32 if self._is_actor else torch.bfloat16
        else:
            torch_dtype = PrecisionType.to_dtype(torch_dtype)
        
        attn_implementation = override_model_config.get("attn_implementation", "flash_attention_2")
        model_config = AutoConfig.from_pretrained(
            local_path, trust_remote_code=trust_remote_code, attn_implementation=attn_implementation
        )
        
        self.generation_config = get_generation_config(local_path, trust_remote_code=trust_remote_code)
        
        override_config_kwargs = {
            "bos_token_id": self.tokenizer.bos_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "pad_token_id": self.tokenizer.pad_token_id,
        }
        override_config_kwargs.update(override_model_config)
        update_model_config(model_config, override_config_kwargs=override_config_kwargs)
        
        if self.rank == 0:
            print(f"Model config after override: {model_config}")
        
        init_context = get_init_weight_context_manager(
            use_meta_tensor=not model_config.tie_word_embeddings, mesh=self.device_mesh
        )
        
        with init_context(), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            base_model = AutoModelForCausalLM.from_pretrained(
                local_path,
                config=model_config,
                torch_dtype=torch_dtype,
                trust_remote_code=trust_remote_code,
                attn_implementation=attn_implementation,
            )
        
        base_model.to(torch_dtype)
        
        if enable_gradient_checkpointing:
            base_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        
        def has_fsdp_wrapper(module):
            """English docstring."""
            if isinstance(module, FSDP):
                return True
            for child in module.children():
                if has_fsdp_wrapper(child):
                    return True
            return False
        
        if has_fsdp_wrapper(base_model):
            raise ValueError("Base model already has FSDP wrapper! This should not happen.")
        
        print(f"[DualLoRA] Applying dual LoRA adapters to model")
        base_model.enable_input_require_grads()
        
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=self._lora_rank,
            lora_alpha=self._lora_alpha,
            target_modules=convert_to_regular_types(self.config.model.target_modules),
            bias="none",
        )
        policy_lora_path = self.config.model.get("policy_lora_path", None)
        rubrics_lora_path = self.config.model.get("rubrics_lora_path", None)
        
        if policy_lora_path and os.path.isdir(policy_lora_path):
            if self.rank == 0:
                print(f"[DualLoRA] Loading policy LoRA from: {policy_lora_path}")
            
            ckpt_format = self._detect_checkpoint_format(policy_lora_path)
            if self.rank == 0:
                print(f"[DualLoRA] Detected checkpoint format: {ckpt_format}")
            
            if self._adapter_weights_have_named_keys(policy_lora_path, POLICY_LLM_ADAPTER):
                peft_model = get_peft_model(base_model, lora_config, adapter_name=POLICY_LLM_ADAPTER)
                self._load_adapter_state_dict(
                    peft_model, policy_lora_path, POLICY_LLM_ADAPTER, strip_named_keys=True
                )
            elif ckpt_format == "named":
                if self.rank == 0:
                    print(f"[DualLoRA] Loading non-standard format checkpoint (converting .default. keys)")
                peft_model = get_peft_model(base_model, lora_config, adapter_name=POLICY_LLM_ADAPTER)
                loaded_count = self._load_checkpoint_with_key_conversion(
                    peft_model, policy_lora_path, POLICY_LLM_ADAPTER, source_adapter_name="default"
                )
                if self.rank == 0:
                    print(f"[DualLoRA] Loaded {loaded_count} weights for '{POLICY_LLM_ADAPTER}'")
            else:
                if self.rank == 0:
                    print(f"[DualLoRA] Loading standard PEFT format checkpoint (direct state_dict path)")
                peft_model = get_peft_model(base_model, lora_config, adapter_name=POLICY_LLM_ADAPTER)
                self._load_adapter_state_dict(
                    peft_model, policy_lora_path, POLICY_LLM_ADAPTER, strip_named_keys=False
                )
        elif policy_lora_path:
            if self.rank == 0:
                print(f"[DualLoRA] WARNING: policy_lora_path not found, fallback to fresh adapter: {policy_lora_path}")
            peft_model = get_peft_model(base_model, lora_config, adapter_name=POLICY_LLM_ADAPTER)
        else:
            peft_model = get_peft_model(base_model, lora_config, adapter_name=POLICY_LLM_ADAPTER)
        
        if has_fsdp_wrapper(peft_model):
            if self.rank == 0:
                print("[DualLoRA] WARNING: PEFT model has FSDP wrapper, this may cause issues")
        
        if rubrics_lora_path and os.path.isdir(rubrics_lora_path):
            if self.rank == 0:
                print(f"[DualLoRA] Loading rubrics LoRA from: {rubrics_lora_path}")
            
            ckpt_format = self._detect_checkpoint_format(rubrics_lora_path)
            if self.rank == 0:
                print(f"[DualLoRA] Detected checkpoint format: {ckpt_format}")
            
            if self._adapter_weights_have_named_keys(rubrics_lora_path, RUBRICS_GEN_ADAPTER):
                if RUBRICS_GEN_ADAPTER not in peft_model.peft_config:
                    peft_model.add_adapter(RUBRICS_GEN_ADAPTER, lora_config)
                self._load_adapter_state_dict(
                    peft_model, rubrics_lora_path, RUBRICS_GEN_ADAPTER, strip_named_keys=True
                )
            elif ckpt_format == "named":
                if self.rank == 0:
                    print(f"[DualLoRA] Loading non-standard format checkpoint (converting .default. keys)")
                if RUBRICS_GEN_ADAPTER not in peft_model.peft_config:
                    peft_model.add_adapter(RUBRICS_GEN_ADAPTER, lora_config)
                loaded_count = self._load_checkpoint_with_key_conversion(
                    peft_model, rubrics_lora_path, RUBRICS_GEN_ADAPTER, source_adapter_name="default"
                )
                if self.rank == 0:
                    print(f"[DualLoRA] Loaded {loaded_count} weights for '{RUBRICS_GEN_ADAPTER}'")
            else:
                if self.rank == 0:
                    print(f"[DualLoRA] Loading standard PEFT format checkpoint (direct state_dict path)")
                if RUBRICS_GEN_ADAPTER not in peft_model.peft_config:
                    peft_model.add_adapter(RUBRICS_GEN_ADAPTER, lora_config)
                self._load_adapter_state_dict(
                    peft_model, rubrics_lora_path, RUBRICS_GEN_ADAPTER, strip_named_keys=False
                )
        elif rubrics_lora_path:
            if self.rank == 0:
                print(f"[DualLoRA] WARNING: rubrics_lora_path not found, fallback to fresh adapter: {rubrics_lora_path}")
            if RUBRICS_GEN_ADAPTER not in peft_model.peft_config:
                peft_model.add_adapter(RUBRICS_GEN_ADAPTER, lora_config)
        else:
            if RUBRICS_GEN_ADAPTER not in peft_model.peft_config:
                peft_model.add_adapter(RUBRICS_GEN_ADAPTER, lora_config)
        
        if has_fsdp_wrapper(peft_model):
            raise ValueError("PEFT model has FSDP wrapper after adding adapters! This should not happen.")
        
        lora_param_count = 0
        policy_count = 0
        rubrics_count = 0
        
        policy_param_refs = []
        rubrics_param_refs = []
        
        for name, param in peft_model.named_parameters():
            if "lora_" in name.lower():
                lora_param_count += 1
                if POLICY_LLM_ADAPTER in name:
                    policy_param_refs.append(param)
                    policy_count += 1
                    param.requires_grad = True
                elif RUBRICS_GEN_ADAPTER in name:
                    rubrics_param_refs.append(param)
                    rubrics_count += 1
                    param.requires_grad = False
        
        peft_model.set_adapter(POLICY_LLM_ADAPTER)
        
        self._policy_param_refs = policy_param_refs
        self._rubrics_param_refs = rubrics_param_refs
        
        self.current_adapter = POLICY_LLM_ADAPTER
        
        if self.rank == 0:
            print(f"[DualLoRA] Total LoRA params set to trainable: {lora_param_count}")
            print(f"[DualLoRA] Policy-LLM params (before FSDP): {policy_count}")
            print(f"[DualLoRA] Rubrics Generator params (before FSDP): {rubrics_count}")
        
        if self.rank == 0:
            print(f"[DualLoRA] Created adapters: {list(peft_model.peft_config.keys())}")
            print(f"[DualLoRA] Active adapter: {POLICY_LLM_ADAPTER}")
            print_model_size(peft_model)
            print("[DualLoRA] Sample param names (before FSDP):")
            for i, (name, param) in enumerate(peft_model.named_parameters()):
                if param.requires_grad and i < 10:
                    print(f"  {name}")
        
        torch.distributed.barrier()
        log_gpu_memory_usage("After creating dual LoRA model", logger=logger)
        
        mixed_precision_config = fsdp_config.get("mixed_precision", None)
        if mixed_precision_config is not None:
            param_dtype = PrecisionType.to_dtype(mixed_precision_config.get("param_dtype", "bf16"))
            reduce_dtype = PrecisionType.to_dtype(mixed_precision_config.get("reduce_dtype", "fp32"))
            buffer_dtype = PrecisionType.to_dtype(mixed_precision_config.get("buffer_dtype", "fp32"))
        else:
            param_dtype = PrecisionType.to_dtype(fsdp_config.dtype)
            reduce_dtype = torch.float32
            buffer_dtype = torch.float32
        
        if self.rank == 0:
            print(f"[DualLoRA] Converting LoRA parameters to {param_dtype} before FSDP wrap")
        
        lora_params_converted = 0
        for name, param in peft_model.named_parameters():
            if "lora_" in name.lower() and param.dtype != param_dtype:
                param.data = param.data.to(dtype=param_dtype)
                lora_params_converted += 1
        
        if self.rank == 0:
            print(f"[DualLoRA] Converted {lora_params_converted} LoRA parameters to {param_dtype}")
        
        mixed_precision = MixedPrecision(param_dtype=param_dtype, reduce_dtype=reduce_dtype, buffer_dtype=buffer_dtype)
        
        wrap_policy_config = fsdp_config.get("wrap_policy", None)
        if wrap_policy_config is None:
            wrap_policy_config = {}
        
        from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy, _or_policy, lambda_auto_wrap_policy
        from transformers.trainer_pt_utils import get_module_class_from_name
        
        policies = []
        
        # Transformer layer policy
        default_transformer_cls_names_to_wrap = [
            "Qwen2DecoderLayer",
            "DecoderLayer",
            "GPTNeoXLayer",
            "LlamaDecoderLayer",
        ]
        fsdp_transformer_layer_cls_to_wrap = wrap_policy_config.get(
            "transformer_layer_cls_to_wrap", default_transformer_cls_names_to_wrap
        )
        
        transformer_cls_to_wrap = set()
        for layer_class in fsdp_transformer_layer_cls_to_wrap:
            transformer_cls = get_module_class_from_name(peft_model, layer_class)
            if transformer_cls is not None:
                transformer_cls_to_wrap.add(transformer_cls)
        
        if len(transformer_cls_to_wrap) > 0:
            transformer_policy = functools.partial(
                transformer_auto_wrap_policy,
                transformer_layer_cls=transformer_cls_to_wrap,
            )
            policies.append(transformer_policy)
        
        
        if len(policies) > 0:
            auto_wrap_policy = functools.partial(_or_policy, policies=policies)
        else:
            auto_wrap_policy = None
        
        fsdp_mesh = self.device_mesh
        sharding_strategy = get_sharding_strategy(fsdp_mesh)
        
        use_orig_params_for_lora = True
        
        if self.rank == 0:
            print(f"[DualLoRA] Using use_orig_params={use_orig_params_for_lora} for FSDP (required for mixed requires_grad)")
        
        fsdp_strategy = self.config.actor.strategy
        if fsdp_strategy == "fsdp":
            cpu_offload_config = None
            if fsdp_config.get("param_offload", False):
                cpu_offload_config = CPUOffload(offload_params=True)
                if self.rank == 0:
                    print(f"[DualLoRA] FSDP CPU offload enabled: offload_params=True")
                    print("[message omitted]")
            else:
                if self.rank == 0:
                    print(f"[DualLoRA] FSDP CPU offload disabled")
            
            model_fsdp = FSDP(
                peft_model,
                cpu_offload=cpu_offload_config,
                param_init_fn=init_fn,
                sync_module_states=True,
                auto_wrap_policy=auto_wrap_policy,
                mixed_precision=mixed_precision,
                sharding_strategy=sharding_strategy,
                device_mesh=fsdp_mesh,
                device_id=get_device_id(),
                use_orig_params=use_orig_params_for_lora,
            )
        else:
            raise NotImplementedError(f"Strategy {fsdp_strategy} not supported for dual LoRA")
        
        log_gpu_memory_usage("After FSDP wrap", logger=logger)
        
        if optim_config is not None:
            self._create_dual_optimizers(model_fsdp, optim_config, fsdp_config)
        
        return model_fsdp, model_config
    
    def _create_dual_optimizers(self, model_fsdp, optim_config, fsdp_config):
        """English docstring."""
        from verl.utils.config import omega_conf_to_dataclass
        from verl.workers.config import FSDPOptimizerConfig
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        
        if self.rank == 0:
            print(f"[DualLoRA] Creating DUAL independent optimizers (after FSDP wrap)...")
            print(f"[DualLoRA] Each adapter will have its own optimizer, step counter, and LR scheduler")
        
        peft_model = getattr(model_fsdp, "_fsdp_wrapped_module", model_fsdp)
        
        policy_lora_params = []
        rubrics_lora_params = []
        policy_param_names = set()
        rubrics_param_names = set()
        
        for name, param in peft_model.named_parameters():
            if "lora_" not in name.lower():
                continue
            
            if POLICY_LLM_ADAPTER in name or f".{POLICY_LLM_ADAPTER}." in name:
                if name not in policy_param_names:
                    policy_lora_params.append(param)
                    policy_param_names.add(name)
            elif RUBRICS_GEN_ADAPTER in name or f".{RUBRICS_GEN_ADAPTER}." in name:
                if name not in rubrics_param_names:
                    rubrics_lora_params.append(param)
                    rubrics_param_names.add(name)
        
        if self.rank == 0:
            print(f"[DualLoRA] Policy-LLM LoRA params: {len(policy_lora_params)}")
            print(f"[DualLoRA] Rubrics Generator LoRA params: {len(rubrics_lora_params)}")
        
        if len(policy_lora_params) == 0:
            raise ValueError(f"No LoRA parameters found for adapter '{POLICY_LLM_ADAPTER}'!")
        if len(rubrics_lora_params) == 0:
            raise ValueError(f"No LoRA parameters found for adapter '{RUBRICS_GEN_ADAPTER}'!")
        
        policy_lr = optim_config.get("policy_llm_lr", None)
        rubrics_lr = optim_config.get("rubrics_generator_lr", None)
        
        from omegaconf import OmegaConf
        from dataclasses import replace
        
        optim_config_clean = OmegaConf.to_container(optim_config, resolve=True)
        optim_config_clean.pop("policy_llm_lr", None)
        optim_config_clean.pop("rubrics_generator_lr", None)
        optim_config_clean = OmegaConf.create(optim_config_clean)
        
        optim_config_dataclass = omega_conf_to_dataclass(optim_config_clean, dataclass_type=FSDPOptimizerConfig)
        
        if policy_lr is None:
            policy_lr = optim_config_dataclass.lr
        if rubrics_lr is None:
            rubrics_lr = optim_config_dataclass.lr
        
        policy_optim_config = replace(optim_config_dataclass, lr=policy_lr)
        rubrics_optim_config = replace(optim_config_dataclass, lr=rubrics_lr)
        
        if self.rank == 0:
            print(f"[DualLoRA] Policy-LLM learning rate: {policy_lr}")
            print(f"[DualLoRA] Rubrics Generator learning rate: {rubrics_lr}")
        
        self.policy_optimizer = build_optimizer(
            parameters=policy_lora_params,
            config=policy_optim_config,
        )
        
        self.rubrics_optimizer = build_optimizer(
            parameters=rubrics_lora_params,
            config=rubrics_optim_config,
        )
        
        if self.rank == 0:
            print(f"[DualLoRA] Created independent Policy-LLM optimizer with {len(policy_lora_params)} params, lr={policy_lr}")
            print(f"[DualLoRA] Created independent Rubrics Generator optimizer with {len(rubrics_lora_params)} params, lr={rubrics_lr}")
            print(f"[DualLoRA] Optimizer type: {type(self.policy_optimizer).__name__}")
        
        from transformers import get_cosine_schedule_with_warmup, get_constant_schedule_with_warmup
        
        total_steps = optim_config.get("total_training_steps", 1000)
        num_warmup_steps = int(optim_config.get("lr_warmup_steps_ratio", 0.0) * total_steps)
        lr_scheduler_type = optim_config.get("lr_scheduler_type", "constant")
        
        if self.rank == 0:
            print(f"[DualLoRA] LR scheduler type: {lr_scheduler_type}")
            print(f"[DualLoRA] Total steps: {total_steps}, Warmup steps: {num_warmup_steps}")
        
        if lr_scheduler_type == "constant":
            self.policy_lr_scheduler = get_constant_schedule_with_warmup(
                optimizer=self.policy_optimizer, num_warmup_steps=num_warmup_steps
            )
            self.rubrics_lr_scheduler = get_constant_schedule_with_warmup(
                optimizer=self.rubrics_optimizer, num_warmup_steps=num_warmup_steps
            )
        elif lr_scheduler_type == "cosine":
            self.policy_lr_scheduler = get_cosine_schedule_with_warmup(
                optimizer=self.policy_optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=total_steps,
            )
            self.rubrics_lr_scheduler = get_cosine_schedule_with_warmup(
                optimizer=self.rubrics_optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=total_steps,
            )
        else:
            raise ValueError(f"Unsupported lr_scheduler_type: {lr_scheduler_type}")
        
        if self.rank == 0:
            print(f"[DualLoRA] Created independent LR schedulers for both adapters")
            print(f"[DualLoRA] ==================== Optimizer Isolation Summary ====================")
            print(f"[DualLoRA] Policy-LLM:        optimizer={id(self.policy_optimizer)}, scheduler={id(self.policy_lr_scheduler)}")
            print(f"[DualLoRA] Rubrics Generator: optimizer={id(self.rubrics_optimizer)}, scheduler={id(self.rubrics_lr_scheduler)}")
            print(f"[DualLoRA] Isolation verified: optimizer_isolated={self.policy_optimizer is not self.rubrics_optimizer}, "
                  f"scheduler_isolated={self.policy_lr_scheduler is not self.rubrics_lr_scheduler}")
            print(f"[DualLoRA] =====================================================================")
        
        log_gpu_memory_usage("After creating dual optimizers", logger=logger)
    
    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        """English docstring."""
        from verl.workers.actor import DataParallelPPOActor
        
        import_external_libs(self.config.model.get("external_lib", None))
        
        override_model_config = OmegaConf.to_container(
            OmegaConf.create(self.config.model.get("override_config", {}))
        )
        use_remove_padding = self.config.model.get("use_remove_padding", False)
        use_shm = self.config.model.get("use_shm", False)
        
        if self._is_actor or self._is_rollout:
            if self._is_actor:
                optim_config = self.config.actor.optim
                fsdp_config = omega_conf_to_dataclass(self.config.actor.fsdp_config)
            else:
                optim_config = None
                fsdp_config = FSDPEngineConfig()
            
            local_path = copy_to_local(self.config.model.path, use_shm=use_shm)
            
            self.actor_module_fsdp, self.actor_model_config = self._build_dual_lora_model(
                model_path=local_path,
                fsdp_config=fsdp_config,
                optim_config=optim_config,
                override_model_config=override_model_config,
                use_remove_padding=use_remove_padding,
                enable_gradient_checkpointing=self.config.model.get("enable_gradient_checkpointing", False),
                trust_remote_code=self.config.model.get("trust_remote_code", False),
            )
            
            if fsdp_version(self.actor_module_fsdp) == 1:
                self.actor_module = self.actor_module_fsdp._fsdp_wrapped_module
            
            if self._is_offload_param:
                offload_fsdp_model_to_cpu(self.actor_module_fsdp)
            
            if self._is_offload_optimizer:
                if self.policy_optimizer:
                    offload_fsdp_optimizer(optimizer=self.policy_optimizer)
                if self.rubrics_optimizer:
                    offload_fsdp_optimizer(optimizer=self.rubrics_optimizer)
        
        if self._is_actor:
            actor_cfg_dict = OmegaConf.to_container(self.config.actor, resolve=True)
            if isinstance(actor_cfg_dict, dict):
                optim_cfg = actor_cfg_dict.get("optim", {})
                if isinstance(optim_cfg, dict):
                    optim_cfg.pop("policy_llm_lr", None)
                    optim_cfg.pop("rubrics_generator_lr", None)
                    actor_cfg_dict["optim"] = optim_cfg
                if "use_remove_padding" not in actor_cfg_dict:
                    actor_cfg_dict["use_remove_padding"] = use_remove_padding
                actor_cfg = omega_conf_to_dataclass(OmegaConf.create(actor_cfg_dict))
            else:
                actor_cfg = omega_conf_to_dataclass(self.config.actor)
                actor_cfg_dict = OmegaConf.to_container(self.config.actor, resolve=True)
                if isinstance(actor_cfg_dict, dict):
                    actor_cfg_dict["use_remove_padding"] = use_remove_padding
                    actor_cfg = omega_conf_to_dataclass(OmegaConf.create(actor_cfg_dict))
            self.actor = DataParallelPPOActor(
                config=actor_cfg, 
                actor_module=self.actor_module_fsdp, 
                actor_optimizer=self.policy_optimizer
            )
        
        if self._is_rollout:
            self._build_rollout(trust_remote_code=self.config.model.get("trust_remote_code", False))
        
        if self._is_ref:
            OmegaConf.set_struct(self.config.ref, True)
            with open_dict(self.config.ref):
                self.config.ref.use_remove_padding = use_remove_padding
            self.ref_module_fsdp = self.actor_module_fsdp
            self.ref_policy = DataParallelPPOActor(config=self.config.ref, actor_module=self.ref_module_fsdp)
        
        if self._is_actor:
            self.flops_counter = FlopsCounter(self.actor_model_config)
            self.checkpoint_manager = FSDPCheckpointManager(
                model=self.actor_module_fsdp,
                optimizer=self.policy_optimizer,
                lr_scheduler=self.policy_lr_scheduler,
                processing_class=self.processor if self.processor is not None else self.tokenizer,
                checkpoint_config=self.config.actor.checkpoint,
            )
            self._rubrics_checkpoint_state = {
                "optimizer": self.rubrics_optimizer,
                "lr_scheduler": self.rubrics_lr_scheduler,
            }
            if self.rank == 0:
                print(f"[DualLoRA] Checkpoint manager initialized for dual optimizers")
        
        self.base_sync_done = "dummy" not in self.config.rollout.load_format
        self.layered_summon = self.config.rollout.get("layered_summon", False)
    
    def _build_rollout(self, trust_remote_code=False):
        """English docstring."""
        from torch.distributed.device_mesh import init_device_mesh
        
        rollout_config: RolloutConfig = omega_conf_to_dataclass(self.config.rollout)
        model_config: HFModelConfig = omega_conf_to_dataclass(self.config.model, dataclass_type=HFModelConfig)
        self.model_config = model_config
        
        infer_tp = self.config.rollout.tensor_model_parallel_size * self.config.rollout.data_parallel_size
        infer_pp = self.config.rollout.pipeline_model_parallel_size
        infer_world_size = infer_tp * infer_pp
        dp = self.world_size // infer_world_size
        
        rollout_device_mesh = init_device_mesh(
            device_name, mesh_shape=(dp, infer_tp, infer_pp), mesh_dim_names=["dp", "infer_tp", "infer_pp"]
        )
        
        is_collect = (
            rollout_device_mesh["infer_tp"].get_local_rank() == 0
            and rollout_device_mesh["infer_pp"].get_local_rank() == 0
        )
        self._register_dispatch_collect_info(
            "rollout", dp_rank=rollout_device_mesh["dp"].get_local_rank(), is_collect=is_collect
        )
        
        self.torch_random_states = get_torch_device().get_rng_state()
        gen_dp_rank = rollout_device_mesh["dp"].get_local_rank()
        get_torch_device().manual_seed(gen_dp_rank + 1000)
        self.gen_random_states = get_torch_device().get_rng_state()
        get_torch_device().set_rng_state(self.torch_random_states)
        
        log_gpu_memory_usage(f"Before building {self.config.rollout.name} rollout", logger=logger)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            logger.info(f"[DualLoRA] GPU cache cleared before vLLM initialization")
        
        self.rollout = get_rollout_class(rollout_config.name, rollout_config.mode)(
            config=rollout_config, model_config=model_config, device_mesh=rollout_device_mesh
        )
        log_gpu_memory_usage(f"After building {self.config.rollout.name} rollout", logger=logger)
        
        if torch.distributed.get_world_size() == 1 and fsdp_version(self.actor_module_fsdp) == 1:
            FSDP.set_state_dict_type(
                self.actor_module_fsdp,
                state_dict_type=StateDictType.FULL_STATE_DICT,
                state_dict_config=FullStateDictConfig(),
            )
        elif fsdp_version(self.actor_module_fsdp) == 1:
            FSDP.set_state_dict_type(
                self.actor_module_fsdp,
                state_dict_type=StateDictType.SHARDED_STATE_DICT,
                state_dict_config=ShardedStateDictConfig(),
            )
        
        if rollout_config.mode == "sync" and self._is_actor:
            try:
                loop = get_event_loop()
                if hasattr(loop, 'is_running') and loop.is_running():
                    if self.rank == 0:
                        print("[DualLoRA] Event loop is already running, trainer_mode() will be called by AgentLoopManager")
                else:
                    loop.run_until_complete(self.trainer_mode())
            except RuntimeError as e:
                if "already running" in str(e):
                    if self.rank == 0:
                        print("[DualLoRA] Event loop is already running, trainer_mode() will be called by AgentLoopManager")
                else:
                    raise
    
    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def set_adapter(self, adapter_name: str):
        """English docstring."""
        assert adapter_name in [POLICY_LLM_ADAPTER, RUBRICS_GEN_ADAPTER]
        
        peft_model = getattr(self.actor_module_fsdp, "_fsdp_wrapped_module", self.actor_module_fsdp)
        if hasattr(peft_model, "set_adapter"):
            old_adapter = self.current_adapter
            if old_adapter != adapter_name:
                print("[message omitted]")
                
                self._set_adapter_requires_grad(peft_model, old_adapter, False)
                
                peft_model.set_adapter(adapter_name)
                self.current_adapter = adapter_name
                
                self._set_adapter_requires_grad(peft_model, adapter_name, True)

                self._force_sync_required = True

                self._verify_adapter_switch(peft_model, adapter_name)

                if dist.is_initialized():
                    dist.barrier()

                if self.rank == 0:
                    print(f"[DualLoRA] Switched to adapter: {adapter_name} (sync required)")
            else:
                if self.rank == 0:
                    print(f"[DualLoRA] Adapter already set to: {adapter_name}")
    
    def _set_adapter_requires_grad(self, peft_model, adapter_name: str, requires_grad: bool):
        """English docstring."""
        count = 0
        for name, param in peft_model.named_parameters():
            if "lora_" in name.lower():
                if (adapter_name in name or
                    f".{adapter_name}." in name or
                    name.endswith(f".{adapter_name}")):
                    if param.requires_grad != requires_grad:
                        param.requires_grad = requires_grad
                        count += 1

        if self.rank == 0 and count > 0:
            status = "enabled" if requires_grad else "disabled"
            print(f"[DualLoRA] {status} {count} parameters for adapter: {adapter_name}")

    def _verify_adapter_switch(self, peft_model, expected_adapter: str):
        """English docstring."""
        if hasattr(peft_model, "active_adapter"):
            actual_adapter = peft_model.active_adapter
            if actual_adapter != expected_adapter:
                raise RuntimeError(
                    f"Adapter switch verification failed! "
                    f"Expected: {expected_adapter}, Actual: {actual_adapter}"
                )

        target_params_enabled = 0
        other_params_enabled = 0
        for name, param in peft_model.named_parameters():
            if "lora_" in name.lower():
                is_target = (expected_adapter in name or
                           f".{expected_adapter}." in name or
                           name.endswith(f".{expected_adapter}"))

                if is_target and param.requires_grad:
                    target_params_enabled += 1
                elif not is_target and param.requires_grad:
                    other_params_enabled += 1

        if target_params_enabled == 0:
            raise RuntimeError(
                f"Adapter switch verification failed! "
                f"No parameters have requires_grad=True for adapter: {expected_adapter}"
            )

        if self.rank == 0:
            print(f"[DualLoRA] Adapter verification passed for: {expected_adapter}")
            print(f"[DualLoRA]   - Target adapter params with grad: {target_params_enabled}")
        if other_params_enabled > 0:
            raise RuntimeError(
                f"Gradient contamination detected: {other_params_enabled} "
                f"non-target adapter parameters have requires_grad=True."
            )
    
    def get_current_optimizer(self):
        """English docstring."""
        if self.current_adapter == POLICY_LLM_ADAPTER:
            return self.policy_optimizer
        else:
            return self.rubrics_optimizer
    
    def get_current_lr_scheduler(self):
        """English docstring."""
        if self.current_adapter == POLICY_LLM_ADAPTER:
            return self.policy_lr_scheduler
        else:
            return self.rubrics_lr_scheduler
    
    def get_optimizer_stats(self):
        """English docstring."""
        stats = {
            "isolation_verified": {
                "optimizer_isolated": self.policy_optimizer is not self.rubrics_optimizer,
                "scheduler_isolated": self.policy_lr_scheduler is not self.rubrics_lr_scheduler,
            },
            "policy_llm": {
                "optimizer_id": id(self.policy_optimizer) if self.policy_optimizer else None,
                "scheduler_id": id(self.policy_lr_scheduler) if self.policy_lr_scheduler else None,
                "step_count": self._get_optimizer_step_count(self.policy_optimizer),
                "current_lr": self._get_current_lr(self.policy_optimizer),
            },
            "rubrics_generator": {
                "optimizer_id": id(self.rubrics_optimizer) if self.rubrics_optimizer else None,
                "scheduler_id": id(self.rubrics_lr_scheduler) if self.rubrics_lr_scheduler else None,
                "step_count": self._get_optimizer_step_count(self.rubrics_optimizer),
                "current_lr": self._get_current_lr(self.rubrics_optimizer),
            },
            "current_adapter": self.current_adapter,
        }
        return stats
    
    def _get_optimizer_step_count(self, optimizer):
        """English docstring."""
        if optimizer is None:
            return None
        try:
            for param_group in optimizer.param_groups:
                for param in param_group["params"]:
                    if param in optimizer.state:
                        state = optimizer.state[param]
                        if "step" in state:
                            step = state["step"]
                            if hasattr(step, "item"):
                                return step.item()
                            return step
        except Exception:
            pass
        return 0
    
    def _get_current_lr(self, optimizer):
        """English docstring."""
        if optimizer is None:
            return None
        try:
            return optimizer.param_groups[0]["lr"]
        except Exception:
            return None
    
    def print_optimizer_stats(self):
        """English docstring."""
        if self.rank == 0:
            stats = self.get_optimizer_stats()
            print(f"[DualLoRA] ==================== Optimizer Stats ====================")
            print(f"[DualLoRA] Current adapter: {stats['current_adapter']}")
            print(f"[DualLoRA] Isolation verified:")
            print(f"[DualLoRA]   Optimizer isolated: {stats['isolation_verified']['optimizer_isolated']}")
            print(f"[DualLoRA]   Scheduler isolated: {stats['isolation_verified']['scheduler_isolated']}")
            print(f"[DualLoRA] Policy-LLM:")
            print(f"[DualLoRA]   Step count: {stats['policy_llm']['step_count']}")
            print(f"[DualLoRA]   Current LR: {stats['policy_llm']['current_lr']}")
            print(f"[DualLoRA] Rubrics Generator:")
            print(f"[DualLoRA]   Step count: {stats['rubrics_generator']['step_count']}")
            print(f"[DualLoRA]   Current LR: {stats['rubrics_generator']['current_lr']}")
            print(f"[DualLoRA] ============================================================")
    
    def diagnose_lora_params(self, adapter_name: str = None):
        """English docstring."""
        if adapter_name is None:
            adapter_name = self.current_adapter
        
        peft_model = getattr(self.actor_module_fsdp, "_fsdp_wrapped_module", self.actor_module_fsdp)
        
        stats = {
            "total_params": 0,
            "params_with_grad": 0,
            "params_requires_grad_true": 0,
            "param_norm_sum": 0.0,
            "grad_norm_sum": 0.0,
            "sample_params": []
        }
        
        for name, param in peft_model.named_parameters():
            if "lora_" not in name.lower():
                continue
            is_target = (adapter_name in name or 
                        f".{adapter_name}." in name or 
                        name.endswith(f".{adapter_name}"))
            if not is_target:
                continue
            
            stats["total_params"] += 1
            if param.requires_grad:
                stats["params_requires_grad_true"] += 1
            if param.grad is not None:
                stats["params_with_grad"] += 1
                stats["grad_norm_sum"] += param.grad.data.norm().item()
            
            param_data = param.data
            if hasattr(param_data, "full_tensor"):
                param_data = param_data.full_tensor()
            stats["param_norm_sum"] += param_data.norm().item()
            
            if len(stats["sample_params"]) < 3:
                stats["sample_params"].append({
                    "name": name,
                    "mean": param_data.float().mean().item(),
                    "std": param_data.float().std().item(),
                    "requires_grad": param.requires_grad,
                    "has_grad": param.grad is not None
                })
        
        if self.rank == 0:
            print("[message omitted]")
            print("[message omitted]")
            print(f"[LoRA Diagnose] requires_grad=True: {stats['params_requires_grad_true']}")
            print("[message omitted]")
            print("[message omitted]")
            print("[message omitted]")
            for sp in stats["sample_params"]:
                print("[message omitted]")
                print(f"                mean={sp['mean']:.6f}, std={sp['std']:.6f}, ")
                print(f"                requires_grad={sp['requires_grad']}, has_grad={sp['has_grad']}")
            print(f"[LoRA Diagnose] ============================================================\n")
        
            return stats
    
    @staticmethod
    def _lora_param_to_cpu_clone(param):
        """English docstring."""
        x = param
        if hasattr(x, "to_local"):
            x = x.to_local()
        if hasattr(x, "full_tensor"):
            x = x.full_tensor()
        elif hasattr(x, "data"):
            x = x.data
        return x.detach().cpu().clone()
    
    def _collect_lora_params_with_adapter(
        self, module: FSDP, adapter_name: str, layered_summon: bool, base_sync_done: bool
    ) -> OrderedDict:
        """English docstring."""
        def _to_cpu_tensor(x):
            if hasattr(x, "to_local"):
                x = x.to_local()
            if hasattr(x, "full_tensor"):
                x = x.full_tensor()
            if torch.is_tensor(x):
                return x.detach().cpu()
            return x

        lora_params = OrderedDict()
        peft_model = getattr(module, "_fsdp_wrapped_module", module)
        
        if fsdp_version(module) > 0:
            if layered_summon:
                if not base_sync_done:
                    raise ValueError(
                        "To use layered_summon, you must make sure base-model "
                        "is preloaded in vllm, e.g. let rollout.load_format=safetensors"
                    )
                lora_params = layered_summon_lora_params(module)
            else:
                with FSDP.summon_full_params(module, writeback=False):
                    try:
                        state = get_peft_model_state_dict(
                            peft_model,
                            adapter_name=adapter_name,
                        )
                        if state and len(state) > 0:
                            lora_params = OrderedDict(
                                (k, _to_cpu_tensor(v)) for k, v in state.items()
                            )
                        else:
                            raise ValueError("empty state dict from get_peft_model_state_dict")
                    except Exception as e:
                        if self.rank == 0:
                            print(
                                f"[DualLoRA] get_peft_model_state_dict failed ({e}), "
                                f"fallback to manual collection for adapter {adapter_name}"
                            )
                        lora_params = self._manual_collect_adapter_params(peft_model, adapter_name)
                get_torch_device().empty_cache()
        else:
            try:
                state = get_peft_model_state_dict(
                    peft_model,
                    adapter_name=adapter_name,
                )
                if state and len(state) > 0:
                    lora_params = OrderedDict(
                        (k, _to_cpu_tensor(v)) for k, v in state.items()
                    )
                else:
                    raise ValueError("empty state dict from get_peft_model_state_dict")
            except Exception as e:
                if self.rank == 0:
                    print(
                        f"[DualLoRA] get_peft_model_state_dict (non-FSDP) failed ({e}), "
                        f"fallback to manual collection for adapter {adapter_name}"
                    )
            lora_params = self._manual_collect_adapter_params(peft_model, adapter_name)
        
        if self.rank == 0 and len(lora_params) == 0:
            print(f"[DualLoRA] WARNING: No LoRA params collected for adapter {adapter_name}!")
            print(f"[DualLoRA] This will cause generation to fail!")

        return lora_params
    
    def _manual_collect_adapter_params(self, peft_model, adapter_name: str) -> OrderedDict:
        """English docstring."""
        lora_params = OrderedDict()
        
        all_lora_param_names = []
        matched_param_names = []
        
        for name, param in peft_model.named_parameters():
            if "lora_" not in name.lower():
                continue
            
            all_lora_param_names.append(name)
            
            # base_model.model.model.layers.X.self_attn.q_proj.lora_A.{adapter_name}.weight
            
            adapter_in_name = False
            if f".{adapter_name}." in name or name.endswith(f".{adapter_name}"):
                adapter_in_name = True
            elif f"lora_A.{adapter_name}" in name or f"lora_B.{adapter_name}" in name:
                adapter_in_name = True
            elif adapter_name in name.split("."):
                adapter_in_name = True
            
            if not adapter_in_name:
                continue
            
            matched_param_names.append(name)
            
            clean_name = self._clean_adapter_param_name(name, adapter_name)
            
            lora_params[clean_name] = self._lora_param_to_cpu_clone(param)
        
        if self.rank == 0 and len(lora_params) == 0:
            print(f"[DualLoRA] WARNING: No params matched for adapter '{adapter_name}'")
            print(f"[DualLoRA] Total LoRA params found: {len(all_lora_param_names)}")
            if all_lora_param_names:
                print(f"[DualLoRA] Sample LoRA param names:")
                for name in all_lora_param_names[:5]:
                    print(f"[DualLoRA]   {name}")
        
        return lora_params
    
    def _clean_adapter_param_name(self, name: str, adapter_name: str) -> str:
        """English docstring."""
        clean_name = name.replace(f".{adapter_name}.", ".")
        if clean_name.endswith(f".{adapter_name}"):
            clean_name = clean_name[:-len(f".{adapter_name}")]
        
        return clean_name
    
    def _normalize_lora_key_for_save(self, name: str, adapter_name: str) -> str:
        """English docstring."""
        clean_name = name
        
        clean_name = clean_name.replace("._fsdp_wrapped_module.", ".")
        clean_name = clean_name.replace("_fsdp_wrapped_module.", "")
        if clean_name.startswith("_fsdp_wrapped_module."):
            clean_name = clean_name[len("_fsdp_wrapped_module."):]
        
        clean_name = clean_name.replace(f".lora_A.{adapter_name}.", ".lora_A.")
        clean_name = clean_name.replace(f".lora_B.{adapter_name}.", ".lora_B.")
        
        clean_name = clean_name.replace(".lora_A.default.", ".lora_A.")
        clean_name = clean_name.replace(".lora_B.default.", ".lora_B.")
        
        return clean_name
    
    def _sync_weights_to_vllm_sync(self):
        """English docstring."""
        import asyncio
        import concurrent.futures
        
        if self.rank == 0:
            print(f"[DualLoRA] Force syncing weights to vLLM for adapter: {self.current_adapter}")
        
        if self._is_offload_param:
            load_fsdp_model_to_gpu(self.actor_module_fsdp)
        
        peft_config = None
        peft_model = getattr(self.actor_module_fsdp, "_fsdp_wrapped_module", self.actor_module_fsdp)
        
        if hasattr(peft_model, "peft_config"):
            peft_config = peft_model.peft_config.get(self.current_adapter, None)
            
            params = self._collect_lora_params_with_adapter(
                module=self.actor_module_fsdp,
                adapter_name=self.current_adapter,
                layered_summon=self.layered_summon,
                base_sync_done=self.base_sync_done,
            )
            
            if len(params) == 0:
                if self.rank == 0:
                    print(f"[DualLoRA] ERROR: No LoRA params collected for adapter {self.current_adapter}!")
                    print(f"[DualLoRA] This will cause generation to produce garbage output!")
                raise ValueError(f"No LoRA params collected for adapter {self.current_adapter}")
            
            if not self.base_sync_done:
                params = {replace_lora_wrapper(k, peft_config): v for k, v in params.items()}
                params = convert_weight_keys(
                    params, getattr(self.actor_module_fsdp, "_fsdp_wrapped_module", self.actor_module_fsdp)
                )
        else:
            params = self.actor_module_fsdp.state_dict()
            params = convert_weight_keys(
                params, getattr(self.actor_module_fsdp, "_fsdp_wrapped_module", self.actor_module_fsdp)
            )
        
        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.actor_module_fsdp)
        
        set_expandable_segments(False)
        
        def convert_peft_key_to_vllm(key: str) -> str:
            """English docstring."""
            if key.startswith("base_model.model."):
                return key[len("base_model.model."):]
            if key.startswith("base_model."):
                return key[len("base_model."):]
            return key
        
        if peft_config is not None and self.base_sync_done:
            converted_params = {convert_peft_key_to_vllm(k): v for k, v in params.items()}
            per_tensor_param = list(converted_params.items())
            
            if self.rank == 0:
                invalid_params = []
                for key, param in converted_params.items():
                    param_float = param.float()
                    if torch.isnan(param_float).any().item() or torch.isinf(param_float).any().item():
                        invalid_params.append(key)
                if invalid_params:
                    print(f"[DualLoRA] ERROR: Found {len(invalid_params)} params with invalid values (nan/inf): {invalid_params[:5]}{'...' if len(invalid_params) > 5 else ''}")
        else:
            device = get_device_id()
            per_tensor_param = [
                (name, param.to(device, non_blocking=True).full_tensor() if isinstance(param, DTensor) else param)
                for name, param in params.items()
            ]
        
        async def _async_update():
            import time as time_module
            
            if self.config.rollout.free_cache_engine:
                await self.rollout.resume(tags=["weights"])
                await self.rollout.resume(tags=["kv_cache"])

            
            if peft_config is not None and self.base_sync_done:
                try:
                    inference_engine = self.rollout.inference_engine
                    
                    try:
                        if hasattr(inference_engine, 'llm_engine'):
                            existing_loras = list(inference_engine.llm_engine.list_loras())
                            if self.rank == 0:
                                print(f"[DualLoRA] Current LoRAs before cleanup: {existing_loras}")
                            
                            
                            for lora_id in existing_loras:
                                try:
                                    inference_engine.llm_engine.remove_lora(lora_id)
                                    if self.rank == 0:
                                        print(f"[DualLoRA] Removed LoRA {lora_id} for clean state")
                                except Exception as rm_e:
                                    if self.rank == 0:
                                        print(f"[DualLoRA] Warning: Failed to remove LoRA {lora_id}: {rm_e}")
                            
                            remaining_loras = list(inference_engine.llm_engine.list_loras())
                            if self.rank == 0:
                                print(f"[DualLoRA] LoRAs after cleanup: {remaining_loras}")
                            
                            from verl.workers.rollout.vllm_rollout.utils import ADAPTER_LORA_IDS
                            target_id = ADAPTER_LORA_IDS.get(self.current_adapter, "unknown")
                            if self.rank == 0:
                                print(f"[DualLoRA] Will add adapter '{self.current_adapter}' with ID={target_id}")
                    except Exception as e:
                        if self.rank == 0:
                            print(f"[DualLoRA] Failed to cleanup LoRAs: {e}")

                    if dist.is_initialized():
                        dist.barrier()
                    
                    await self.rollout.update_weights(
                        per_tensor_param,
                        peft_config=peft_config, 
                        base_sync_done=self.base_sync_done,
                        adapter_name=self.current_adapter
                    )
                    
                    if self.rank == 0:
                        try:
                            if hasattr(inference_engine, 'llm_engine'):
                                updated_loras = list(inference_engine.llm_engine.list_loras())
                                print(f"[DualLoRA] Current LoRAs after update: {updated_loras}")
                                from verl.workers.rollout.vllm_rollout.utils import ADAPTER_LORA_IDS
                                fixed_id = ADAPTER_LORA_IDS.get(self.current_adapter, "unknown")
                                print(f"[DualLoRA] Updated adapter '{self.current_adapter}' with fixed ID: {fixed_id}")
                                
                                if fixed_id not in updated_loras:
                                    print(f"[DualLoRA] ERROR: LoRA {fixed_id} was not loaded correctly!")
                                    print(f"[DualLoRA] Expected LoRA ID {fixed_id} but got {updated_loras}")
                                    raise RuntimeError(f"LoRA {fixed_id} was not loaded correctly after update!")
                                else:
                                    print(f"[DualLoRA] SUCCESS: LoRA {fixed_id} is now loaded and ready")
                        except RuntimeError:
                            raise
                        except Exception as e:
                            print(f"[DualLoRA] Failed to verify LoRAs: {e}")
                            import traceback
                            traceback.print_exc()
                except Exception as e:
                    if self.rank == 0:
                        print(f"[DualLoRA] LoRA update failed: {e}")
                    import traceback
                    traceback.print_exc()
                    raise
            else:
                await self.rollout.update_weights(
                    per_tensor_param, 
                    peft_config=peft_config, 
                    base_sync_done=self.base_sync_done
                )
        
        def _run_in_thread():
            """English docstring."""
            new_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(new_loop)
            try:
                return new_loop.run_until_complete(_async_update())
            finally:
                new_loop.close()
        
        try:
            loop = get_event_loop()
            if loop.is_running():
                if self.rank == 0:
                    print(f"[DualLoRA] Event loop is running, using thread pool for sync...")
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(_run_in_thread)
                    future.result(timeout=300)
            else:
                loop.run_until_complete(_async_update())
        except Exception as e:
            if self.rank == 0:
                print(f"[DualLoRA] Error during weight sync: {e}")
            raise
        
        del params, per_tensor_param
        
        self.base_sync_done = True
        self._last_synced_adapter = self.current_adapter
        self._force_sync_required = False
        
        if self.rank == 0:
            print(f"[DualLoRA] Successfully synced weights for adapter: {self.current_adapter}")
    
    async def rollout_mode(self):
        """English docstring."""
        
        if self._is_offload_param:
            load_fsdp_model_to_gpu(self.actor_module_fsdp)
        
        peft_config = None
        peft_model = getattr(self.actor_module_fsdp, "_fsdp_wrapped_module", self.actor_module_fsdp)
        
        if hasattr(peft_model, "peft_config"):
            peft_config = peft_model.peft_config.get(self.current_adapter, None)
            
            params = self._collect_lora_params_with_adapter(
                module=self.actor_module_fsdp,
                adapter_name=self.current_adapter,
                layered_summon=self.layered_summon,
                base_sync_done=self.base_sync_done,
            )
            
            if len(params) == 0:
                if self.rank == 0:
                    print(f"[DualLoRA] ERROR: No LoRA params collected for adapter {self.current_adapter}!")
                raise ValueError(f"No LoRA params collected for adapter {self.current_adapter}")
            
            if not self.base_sync_done:
                params = {replace_lora_wrapper(k, peft_config): v for k, v in params.items()}
                params = convert_weight_keys(
                    params, getattr(self.actor_module_fsdp, "_fsdp_wrapped_module", self.actor_module_fsdp)
                )
        else:
            params = self.actor_module_fsdp.state_dict()
            params = convert_weight_keys(
                params, getattr(self.actor_module_fsdp, "_fsdp_wrapped_module", self.actor_module_fsdp)
            )
        
        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.actor_module_fsdp)
        
        set_expandable_segments(False)
        
        def convert_peft_key_to_vllm(key: str) -> str:
            """English docstring."""
            if key.startswith("base_model.model."):
                return key[len("base_model.model."):]
            if key.startswith("base_model."):
                return key[len("base_model."):]
            return key
        
        if peft_config is not None and self.base_sync_done:
            converted_params = {convert_peft_key_to_vllm(k): v for k, v in params.items()}
            per_tensor_param = list(converted_params.items())
            
            if self.rank == 0:
                print(f"[DualLoRA] Async rollout_mode: {len(params)} params -> {len(converted_params)} converted")
        else:
            device = get_device_id()
            per_tensor_param = [
                (name, param.to(device, non_blocking=True).full_tensor() if isinstance(param, DTensor) else param)
                for name, param in params.items()
            ]
        
        if self.config.rollout.free_cache_engine:
            await self.rollout.resume(tags=["weights"])
        
        await self.rollout.update_weights(
            per_tensor_param, 
            peft_config=peft_config, 
            base_sync_done=self.base_sync_done,
            adapter_name=self.current_adapter
        )
        
        del params, per_tensor_param
        
        if self.config.rollout.free_cache_engine:
            await self.rollout.resume(tags=["kv_cache"])
        
        self.base_sync_done = True
        self._last_synced_adapter = self.current_adapter
        self._force_sync_required = False
        self.torch_random_states = get_torch_device().get_rng_state()
        get_torch_device().set_rng_state(self.gen_random_states)
    
    async def trainer_mode(self):
        """English docstring."""
        if self.config.rollout.free_cache_engine:
            await self.rollout.release()
        
        self.actor_module_fsdp.train()
        set_expandable_segments(True)
        
        self.gen_random_states = get_torch_device().get_rng_state()
        get_torch_device().set_rng_state(self.torch_random_states)
    
    async def _release_kv_cache_only(self):
        """English docstring."""
        try:
            if hasattr(self.rollout, 'inference_engine'):
                inference_engine = self.rollout.inference_engine
                
                if hasattr(inference_engine, 'sleep'):
                    try:
                        import inspect
                        sig = inspect.signature(inference_engine.sleep)
                        if 'level' in sig.parameters:
                            inference_engine.sleep(level=0)
                            if self.rank == 0:
                                print("[DualLoRA] Released KV cache only (level=0), weights preserved")
                            return
                    except Exception as e:
                        if self.rank == 0:
                            print(f"[DualLoRA] Warning: sleep(level=0) failed: {e}, trying alternative method")
                
                if hasattr(inference_engine, 'reset_prefix_cache'):
                    inference_engine.reset_prefix_cache()
                    if self.rank == 0:
                        print("[DualLoRA] Reset prefix cache to free KV cache")
                
                if hasattr(inference_engine, 'llm_engine'):
                    llm_engine = inference_engine.llm_engine
                    if hasattr(llm_engine, 'scheduler'):
                        scheduler = llm_engine.scheduler
                        if hasattr(scheduler, 'free_finished_seq_blocks'):
                            scheduler.free_finished_seq_blocks()
                            if self.rank == 0:
                                print("[DualLoRA] Freed finished sequence blocks")
                
                aggressive_empty_cache(force_sync=True)
                
            else:
                if self.rank == 0:
                    print("[DualLoRA] Warning: Cannot access inference_engine, skipping KV cache release")
        except Exception as e:
            if self.rank == 0:
                print(f"[DualLoRA] Warning: Failed to release KV cache only: {e}")
                import traceback
                traceback.print_exc()
    
    def _ensure_trainer_mode(self):
        """English docstring."""
        import concurrent.futures
        
        async def _release_vllm():
            await self._release_kv_cache_only()
            
            if self.config.rollout.free_cache_engine:
                await self.rollout.release()
            
            aggressive_empty_cache(force_sync=True)
        
        try:
            loop = get_event_loop()
            if loop.is_running():
                def _run_in_thread():
                    new_loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(new_loop)
                    try:
                        return new_loop.run_until_complete(_release_vllm())
                    finally:
                        new_loop.close()
                
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(_run_in_thread)
                    future.result(timeout=60)
                    if self.rank == 0:
                        print("[DualLoRA] Released vLLM KV Cache for training")
            else:
                loop.run_until_complete(_release_vllm())
        except Exception as e:
            if self.rank == 0:
                print(f"[DualLoRA] Warning: Failed to release vLLM cache: {e}")
        
        self.actor_module_fsdp.train()
        set_expandable_segments(True)
    
    def _sync_optimizer_state_device(self, optimizer):
        """English docstring."""
        if optimizer is None:
            return
        
        param_device = None
        for param in self.actor_module_fsdp.parameters():
            if param.requires_grad:
                param_device = param.device
                break
        
        if param_device is None:
            return
        
        moved_count = 0
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor) and v.device != param_device:
                    state[k] = v.to(param_device)
                    moved_count += 1
        
        if moved_count > 0 and self.rank == 0:
            print(f"[DualLoRA] Synced {moved_count} optimizer state tensors to {param_device}")
    
    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="actor"))
    def update_actor(self, data: DataProto):
        """English docstring."""
        assert self._is_actor
        
        self._ensure_trainer_mode()
        
        if self._is_offload_param:
            load_fsdp_model_to_gpu(self.actor_module_fsdp)
        
        current_optimizer = self.get_current_optimizer()
        if self._is_offload_optimizer and current_optimizer:
            load_fsdp_optimizer(optimizer=current_optimizer, device_id=get_device_id())
        
        if self.rank == 0:
            print("[message omitted]")
            print("[message omitted]")
            print("[message omitted]")
            print("[message omitted]")
        
        self.actor.actor_optimizer = current_optimizer
        
        with self.ulysses_sharding_manager:
            data = data.to("cpu")
            
            data_batch_size = len(data)
            ppo_mini_batch_size = getattr(self.actor.config, 'ppo_mini_batch_size', None)
            print(f"[DualLoRA update_actor] data_batch_size: {data_batch_size}, "
                  f"ppo_mini_batch_size: {ppo_mini_batch_size}")
            
            if ppo_mini_batch_size is None or ppo_mini_batch_size <= 0:
                print("[message omitted]"
                      "")
                if data_batch_size > 0:
                    self.actor.config.ppo_mini_batch_size = data_batch_size
                else:
                    print("[message omitted]")
                    return DataProto(meta_info={"metrics": {}})
            
            if data_batch_size == 0:
                print("[message omitted]")
                return DataProto(meta_info={"metrics": {}})
            
            if self._is_offload_param and current_optimizer is not None:
                self._sync_optimizer_state_device(current_optimizer)
            
            output = self.actor.update_policy(data=data)
            if isinstance(output, dict):
                output = DataProto(meta_info={"metrics": output})
        
        current_lr_scheduler = self.get_current_lr_scheduler()
        if current_lr_scheduler:
            current_lr_scheduler.step()
        
        if self.rank == 0:
            print("[message omitted]")
            print("[message omitted]")
        self.print_optimizer_stats()
        
        self.diagnose_lora_params(self.current_adapter)
        
        if self._is_offload_optimizer and current_optimizer:
            offload_fsdp_optimizer(optimizer=current_optimizer)
        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.actor_module_fsdp)
        
        output = output.to("cpu")
        return output
    
    async def _resume_kv_cache_only(self):
        """English docstring."""
        try:
            if self.config.rollout.free_cache_engine:
                await self.rollout.resume(tags=["kv_cache"])
            else:
                if hasattr(self.rollout, 'inference_engine'):
                    inference_engine = self.rollout.inference_engine
                    if hasattr(inference_engine, 'wake_up'):
                        try:
                            import inspect
                            sig = inspect.signature(inference_engine.wake_up)
                            if 'tags' in sig.parameters:
                                inference_engine.wake_up(tags=["kv_cache"])
                            else:
                                inference_engine.wake_up()
                            if self.rank == 0:
                                print("[DualLoRA] Resumed KV cache for generation")
                        except Exception as e:
                            if self.rank == 0:
                                print(f"[DualLoRA] Warning: Failed to resume KV cache: {e}")
        except Exception as e:
            if self.rank == 0:
                print(f"[DualLoRA] Warning: Failed to resume KV cache only: {e}")
    
    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="rollout"))
    def generate_sequences(self, prompts: DataProto):
        """English docstring."""
        assert self._is_rollout
        prompts = prompts.to(get_device_id())
        
        meta_info = {
            "eos_token_id": self.generation_config.eos_token_id
            if self.generation_config is not None
            else self.tokenizer.eos_token_id,
            "pad_token_id": self.generation_config.pad_token_id
            if self.generation_config is not None
            else self.tokenizer.pad_token_id,
        }
        prompts.meta_info.update(meta_info)
        
        if self._is_actor:
            self._sync_weights_to_vllm_sync()
        else:
            try:
                loop = get_event_loop()
                if loop.is_running():
                    if self.rank == 0:
                        print("[DualLoRA] Event loop running, using sync method for rollout-only worker")
                    self._sync_weights_to_vllm_sync()
                else:
                    loop.run_until_complete(self.rollout_mode())
            except RuntimeError as e:
                if "already running" in str(e) or "this event loop is already running" in str(e):
                    self._sync_weights_to_vllm_sync()
                else:
                    raise
        
        try:
            loop = get_event_loop()
            if loop.is_running():
                import concurrent.futures
                def _run_in_thread():
                    new_loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(new_loop)
                    try:
                        return new_loop.run_until_complete(self._resume_kv_cache_only())
                    finally:
                        new_loop.close()
                
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(_run_in_thread)
                    future.result(timeout=30)
            else:
                loop.run_until_complete(self._resume_kv_cache_only())
        except Exception as e:
            if self.rank == 0:
                print(f"[DualLoRA] Warning: Failed to resume KV cache: {e}")
        
        prompts.meta_info["adapter_name"] = self.current_adapter
        if self.rank == 0:
            print(f"[DualLoRA Generate] Setting adapter_name='{self.current_adapter}' in meta_info")
        
        with self.ulysses_sharding_manager:
            output = self.rollout.generate_sequences(prompts=prompts)
        
        if self._is_actor:
            try:
                loop = get_event_loop()
                if loop.is_running():
                    pass
                else:
                    loop.run_until_complete(self.trainer_mode())
            except RuntimeError as e:
                if "already running" in str(e) or "this event loop is already running" in str(e):
                    pass
                else:
                    raise
        
        return output
    
    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="actor"))
    def compute_log_prob(self, data: DataProto):
        """English docstring."""
        assert self._is_actor
        
        if self._is_offload_param:
            load_fsdp_model_to_gpu(self.actor_module_fsdp)
        
        from contextlib import nullcontext
        
        with self.ulysses_sharding_manager:
            data = data.to(get_device_id())
            if data.meta_info is None:
                data.meta_info = {}
            if "micro_batch_size" not in data.meta_info:
                data.meta_info["micro_batch_size"] = self.config.rollout.log_prob_micro_batch_size_per_gpu
            if "temperature" not in data.meta_info:
                data.meta_info["temperature"] = self.config.rollout.temperature
            if "use_dynamic_bsz" not in data.meta_info:
                data.meta_info["use_dynamic_bsz"] = self.config.rollout.log_prob_use_dynamic_bsz
            if "max_token_len" not in data.meta_info and self.config.rollout.log_prob_max_token_len_per_gpu is not None:
                data.meta_info["max_token_len"] = self.config.rollout.log_prob_max_token_len_per_gpu
            with nullcontext():
                output, entropys = self.actor.compute_log_prob(data=data, calculate_entropy=True)
            output = DataProto.from_dict(
                tensors={"old_log_probs": output, "entropys": entropys},
                meta_info={"temperature": self.config.rollout.temperature},
            )
        
        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.actor_module_fsdp)
        
        output = output.to("cpu")
        return output
    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="actor"))
    def compute_ref_log_prob(self, data: DataProto):
        """English docstring."""
        assert self._is_ref
        
        if self._is_offload_param:
            load_fsdp_model_to_gpu(self.actor_module_fsdp)
        
        peft_model = getattr(self.actor_module_fsdp, "_fsdp_wrapped_module", self.actor_module_fsdp)
        if hasattr(peft_model, "disable_adapter_layers"):
            peft_model.disable_adapter_layers()
        
        with self.ulysses_sharding_manager:
            data = data.to(get_device_id())
            if data.meta_info is None:
                data.meta_info = {}
            if "micro_batch_size" not in data.meta_info:
                data.meta_info["micro_batch_size"] = self.config.ref.log_prob_micro_batch_size_per_gpu
            if "temperature" not in data.meta_info:
                data.meta_info["temperature"] = self.config.rollout.temperature
            if "use_dynamic_bsz" not in data.meta_info:
                data.meta_info["use_dynamic_bsz"] = self.config.ref.log_prob_use_dynamic_bsz
            if "max_token_len" not in data.meta_info and self.config.ref.log_prob_max_token_len_per_gpu is not None:
                data.meta_info["max_token_len"] = self.config.ref.log_prob_max_token_len_per_gpu

            output, _ = self.ref_policy.compute_log_prob(data=data, calculate_entropy=False)
            output = DataProto.from_dict(tensors={"ref_log_prob": output})
        
        if hasattr(peft_model, "enable_adapter_layers"):
            peft_model.enable_adapter_layers()
            peft_model.set_adapter(self.current_adapter)
        
        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.actor_module_fsdp)
        
        output = output.to("cpu")
        return output
    
    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def save_checkpoint(self, local_path: str, adapter_name: str = None):
        """English docstring."""
        if adapter_name is None:
            adapter_name = self.current_adapter
        
        peft_model = getattr(self.actor_module_fsdp, "_fsdp_wrapped_module", self.actor_module_fsdp)
        
        lora_save_path = os.path.join(local_path, f"lora_adapter_{adapter_name}")
        
        if dist.get_rank() == 0:
            os.makedirs(lora_save_path, exist_ok=True)
            
            peft_config = {}
            if hasattr(peft_model, "peft_config"):
                config = peft_model.peft_config.get(adapter_name, None)
                if config:
                    peft_config = asdict(config)
                    if hasattr(peft_config.get("task_type"), "value"):
                        peft_config["task_type"] = peft_config["task_type"].value
                    if hasattr(peft_config.get("peft_type"), "value"):
                        peft_config["peft_type"] = peft_config["peft_type"].value
                    peft_config["target_modules"] = list(peft_config["target_modules"])
        
        try:
            if fsdp_version(self.actor_module_fsdp) > 0:
                self.actor_module_fsdp = self.actor_module_fsdp.to(get_device_name())
                
                lora_params = {}
                with FSDP.summon_full_params(self.actor_module_fsdp, writeback=False):
                    for name, param in peft_model.named_parameters():
                        is_lora_param = "lora" in name.lower()
                        is_target_adapter = (
                            f".lora_A.{adapter_name}." in name or 
                            f".lora_B.{adapter_name}." in name or
                            name.endswith(f".lora_A.{adapter_name}.weight") or
                            name.endswith(f".lora_B.{adapter_name}.weight")
                        )
                        
                        if is_lora_param and is_target_adapter:
                            clean_name = self._normalize_lora_key_for_save(name, adapter_name)
                            lora_params[clean_name] = self._lora_param_to_cpu_clone(param)
                
                if dist.get_rank() == 0:
                    if len(lora_params) == 0:
                        print(f"[DualLoRA] WARNING: No LoRA params collected for adapter {adapter_name}")
                        with FSDP.summon_full_params(self.actor_module_fsdp, writeback=False):
                            for name, param in peft_model.named_parameters():
                                if adapter_name in name and "lora" in name.lower():
                                    clean_name = self._normalize_lora_key_for_save(name, adapter_name)
                                    lora_params[clean_name] = self._lora_param_to_cpu_clone(param)
                        print(f"[DualLoRA] Fallback collected {len(lora_params)} params")
                    
                    save_file(lora_params, os.path.join(lora_save_path, "adapter_model.safetensors"))
                    with open(os.path.join(lora_save_path, "adapter_config.json"), "w", encoding="utf-8") as f:
                        json.dump(peft_config, f, ensure_ascii=False, indent=4)
                    
                    print(f"[DualLoRA] Saved {len(lora_params)} LoRA params for adapter {adapter_name}")
                    if lora_params:
                        sample_key = list(lora_params.keys())[0]
                        print(f"[DualLoRA] Sample key format: {sample_key}")
        except Exception as e:
            if self.rank == 0:
                print(f"[DualLoRA] Save adapter {adapter_name} error: {e}")
                import traceback
                traceback.print_exc()
        
        try:
            if dist.get_rank() == 0:
                optimizer_save_path = os.path.join(local_path, f"optimizer_{adapter_name}")
                os.makedirs(optimizer_save_path, exist_ok=True)
                
                if adapter_name == POLICY_LLM_ADAPTER:
                    optimizer = self.policy_optimizer
                    scheduler = self.policy_lr_scheduler
                else:
                    optimizer = self.rubrics_optimizer
                    scheduler = self.rubrics_lr_scheduler
                
                if optimizer is not None:
                    optimizer_state = {
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
                    }
                    torch.save(optimizer_state, os.path.join(optimizer_save_path, "optimizer_state.pt"))
                    print(f"[DualLoRA] Saved optimizer state for {adapter_name} to: {optimizer_save_path}")
        except Exception as e:
            if self.rank == 0:
                print(f"[DualLoRA] Save optimizer {adapter_name} error: {e}")
        
        dist.barrier()
        if self.rank == 0:
            print(f"[DualLoRA] Saved adapter {adapter_name} to: {lora_save_path}")
    
    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def load_optimizer_checkpoint(self, local_path: str, adapter_name: str = None):
        """English docstring."""
        if adapter_name is None:
            adapter_name = self.current_adapter
        
        optimizer_load_path = os.path.join(local_path, f"optimizer_{adapter_name}", "optimizer_state.pt")
        
        if not os.path.exists(optimizer_load_path):
            if self.rank == 0:
                print(f"[DualLoRA] No optimizer checkpoint found at: {optimizer_load_path}")
            return
        
        try:
            if adapter_name == POLICY_LLM_ADAPTER:
                optimizer = self.policy_optimizer
                scheduler = self.policy_lr_scheduler
            else:
                optimizer = self.rubrics_optimizer
                scheduler = self.rubrics_lr_scheduler
            
            if optimizer is not None:
                checkpoint = torch.load(optimizer_load_path, map_location="cpu")
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                if scheduler is not None and checkpoint.get("scheduler_state_dict"):
                    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
                
                if self.rank == 0:
                    print(f"[DualLoRA] Loaded optimizer state for {adapter_name} from: {optimizer_load_path}")
        except Exception as e:
            if self.rank == 0:
                print(f"[DualLoRA] Load optimizer {adapter_name} error: {e}")
