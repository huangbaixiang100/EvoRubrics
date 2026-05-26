"""
EvoRubric training dataset (AdversarialDataset).
Supports two HealthBench-style data modes:
- golden_rubrics: each sample carries prompt + per-sample golden rubrics + instruction.
- open_rubrics: each sample only carries instruction/query; rubrics are generated on the fly.
"""
import json
from typing import Dict, Any, Optional, List
from torch.utils.data import Dataset
from verl import DataProto

from models import (
    POLICY_LLM_SYSTEM_PROMPT,
    RUBRICS_GENERATOR_SYSTEM_PROMPT,
    UNIVERSAL_RUBRICS_GENERATOR_SYSTEM_PROMPT,
)


class AdversarialDataset(Dataset):
    
    def __init__(
        self,
        data_files: str,
        tokenizer,
        processor,
        config,
        max_samples: int = -1,
        mode: str = "policy_llm"  # "policy_llm" or "rubrics_generator"
    ):
        """English docstring."""
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config
        self.mode = mode
        # dataset_mode chooses between golden_rubrics (per-sample golden rubrics) and open_rubrics
        # (rubrics generated on the fly from instruction/query).
        self.dataset_mode = config.get("dataset_mode", "golden_rubrics") if config else "golden_rubrics"
        
        self.samples = self._load_data(data_files, max_samples)
        self.max_prompt_length = self.config.get("max_prompt_length", 512) if self.config else 512

        if self.config and self.config.get("filter_overlong_prompts", False):
            self.samples = self._filter_overlong_samples(self.samples)
        
        print("[message omitted]")
    
    def _load_data(self, data_files: str, max_samples: int):
        """English docstring."""
        with open(data_files, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if max_samples > 0:
            data = data[:max_samples]
        
        return data
    
    def __len__(self):
        return len(self.samples)

    def _parse_sample_fields(self, sample: Dict[str, Any]):
        """English docstring."""
        if self.dataset_mode == "open_rubrics":
            instruction = sample.get("instruction", "") or sample.get("question", "")
            question = sample.get("question", "") or instruction
            prompt_messages = sample.get("prompt", [{"role": "user", "content": question}])
            if not isinstance(prompt_messages, list) or len(prompt_messages) == 0:
                prompt_messages = [{"role": "user", "content": question}]
            if not str(question).strip():
                question = self._fallback_question_from_prompt_messages(prompt_messages)
            ground_truth_rubrics = sample.get("ground_truth_rubrics", "")
            if not ground_truth_rubrics:
                rubrics_list = sample.get("rubrics", [])
                if isinstance(rubrics_list, list) and rubrics_list:
                    ground_truth_rubrics = self._format_rubrics(rubrics_list)
        elif "prompt" in sample and isinstance(sample["prompt"], list):
            prompt_messages = sample["prompt"]
            instruction = sample.get("instruction", "")
            rubrics = sample.get("rubrics", [])
            ground_truth_rubrics = self._format_rubrics(rubrics)
            question = self._extract_question_from_messages(prompt_messages)
        else:
            if "input" in sample:
                input_text = sample.get("input", "")
            else:
                input_text = sample.get("prompt", "")
            question = self._extract_question_from_text(input_text)
            instruction = sample.get("instruction", "")
            ground_truth_rubrics = sample.get("output", "")
            prompt_messages = [{"role": "user", "content": input_text}]

        golden_answer = sample.get("golden_answer", "")
        return prompt_messages, instruction, question, ground_truth_rubrics, golden_answer

    def _build_prompt_for_sample(self, sample: Dict[str, Any]):
        """English docstring."""
        prompt_messages, instruction, question, _, _ = self._parse_sample_fields(sample)

        if self.mode == "policy_llm":
            messages = self._build_policy_llm_messages(prompt_messages)
        else:
            messages = self._build_rubrics_generator_messages(question, instruction)

        if hasattr(self.tokenizer, "apply_chat_template") and self.tokenizer.chat_template is not None:
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                extra_body={"enable_thinking": False}
            )

        return self._build_prompt_manually(messages)

    def _filter_overlong_samples(self, samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """English docstring."""
        kept_samples = []
        dropped = 0

        for sample in samples:
            prompt = self._build_prompt_for_sample(sample)
            encoded_no_pad = self.tokenizer(
                prompt,
                padding=False,
                truncation=False,
                return_tensors="pt"
            )
            prompt_len = int(encoded_no_pad["input_ids"].shape[-1])

            if prompt_len <= self.max_prompt_length:
                kept_samples.append(sample)
            else:
                dropped += 1

        if dropped > 0:
            print(
                "[message omitted]"
                f"kept={len(kept_samples)}, max_prompt_length={self.max_prompt_length}, mode={self.mode}"
            )

        if len(kept_samples) == 0:
            raise RuntimeError(
                ""
                ""
            )

        return kept_samples
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """English docstring."""
        import torch
        from verl.utils.model import compute_position_id_with_mask
        
        sample = self.samples[idx]
        prompt_messages, instruction, question, ground_truth_rubrics, golden_answer = self._parse_sample_fields(sample)
        
        if self.mode == "policy_llm":
            messages = self._build_policy_llm_messages(prompt_messages)
        else:
            messages = self._build_rubrics_generator_messages(question, instruction)
        
        if hasattr(self.tokenizer, 'apply_chat_template') and self.tokenizer.chat_template is not None:
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                extra_body={"enable_thinking": False}
            )
        else:
            prompt = self._build_prompt_manually(messages)
        
        max_prompt_length = self.max_prompt_length
        encoded_no_pad = self.tokenizer(
            prompt,
            padding=False,
            truncation=False,
            return_tensors="pt"
        )
        prompt_len = int(encoded_no_pad["input_ids"].shape[-1])
        if prompt_len > max_prompt_length:
            raise RuntimeError(
                ""
                f"max_prompt_length={max_prompt_length}. "
                ""
            )

        encoded = self.tokenizer(
            prompt,
            padding="max_length",
            truncation=False,
            max_length=max_prompt_length,
            return_tensors="pt"
        )
        
        input_ids = encoded["input_ids"].squeeze(0)  # (seq_len,)
        attention_mask = encoded["attention_mask"].squeeze(0)  # (seq_len,)
        
        position_ids = compute_position_id_with_mask(attention_mask.unsqueeze(0)).squeeze(0)  # (seq_len,)
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'position_ids': position_ids,
            'data_source': sample.get('data_source', 'healthbench'),
            'prompt': prompt,
            'ground_truth_rubrics': ground_truth_rubrics,
            'golden_answer': golden_answer,
            'question': question,
            'instruction': instruction,
            'prompt_messages_json': json.dumps(prompt_messages, ensure_ascii=False),
            'mode': self.mode
        }
    
    def _build_policy_llm_messages(self, prompt_messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """English docstring."""
        messages = []
        
        messages.append({"role": "system", "content": POLICY_LLM_SYSTEM_PROMPT})
        
        for msg in prompt_messages:
            messages.append(msg)
        
        return messages
    
    def _build_rubrics_generator_messages(self, question: str, instruction: str) -> List[Dict[str, str]]:
        """English docstring."""
        messages = []
        
        # Pick system prompt based on dataset mode.
        if self.dataset_mode == "open_rubrics":
            system_prompt = UNIVERSAL_RUBRICS_GENERATOR_SYSTEM_PROMPT
        else:
            system_prompt = RUBRICS_GENERATOR_SYSTEM_PROMPT
        
        messages.append({"role": "system", "content": system_prompt})
        
        if instruction:
            messages.append({"role": "system", "content": instruction})
        
        messages.append({"role": "user", "content": question})
        
        return messages
    
    def _build_prompt_manually(self, messages: List[Dict[str, str]]) -> str:
        """English docstring."""
        prompt_parts = []
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            prompt_parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")
        
        prompt_parts.append("<|im_start|>assistant\n")
        
        return "\n".join(prompt_parts)
    
    def _fallback_question_from_prompt_messages(self, prompt_messages: Any) -> str:
        """English docstring."""
        if not isinstance(prompt_messages, list) or len(prompt_messages) == 0:
            return ""
        for msg in reversed(prompt_messages):
            if not isinstance(msg, dict) or msg.get("role") != "user":
                continue
            content = msg.get("content", "")
            s = str(content).strip() if content is not None else ""
            if s:
                return s
        parts = []
        for msg in prompt_messages:
            if not isinstance(msg, dict) or msg.get("role") == "system":
                continue
            content = msg.get("content", "")
            s = str(content).strip() if content is not None else ""
            if s:
                parts.append(s)
        return "\n\n".join(parts)

    def _extract_question_from_messages(self, messages: List[Dict[str, str]]) -> str:
        """English docstring."""
        for msg in reversed(messages):
            if not isinstance(msg, dict):
                continue
            if msg.get("role") == "user":
                content = msg.get("content", "")
                return str(content).strip() if content is not None else ""
        return ""
    
    def _extract_question_from_text(self, input_text: str) -> str:
        """English docstring."""
        if input_text.startswith("User: "):
            return input_text[6:].strip()
        return input_text.strip()
    
    def _format_rubrics(self, rubrics: List[Dict[str, Any]]) -> str:
        """English docstring."""
        if not rubrics:
            return ""
        
        rubrics_lines = []
        for i, rubric in enumerate(rubrics, 1):
            criterion = rubric.get('criterion', '')
            points = rubric.get('points', 0)
            tags = rubric.get('tags', [])
            
            axis = "general"
            for tag in tags:
                if isinstance(tag, str) and tag.startswith("axis:"):
                    axis = tag.split(":", 1)[1]
                    break
            
            rubrics_lines.append(f"{i}. {criterion}")
            rubrics_lines.append(f"   points: {points}")
            rubrics_lines.append(f"   axis: {axis}")
            rubrics_lines.append("")
        
        return "\n".join(rubrics_lines)
    
    def _get_rubrics_generator_system_prompt(self) -> str:
        """Return the Rubrics Generator system prompt for the active dataset_mode."""
        if self.dataset_mode == "open_rubrics":
            return UNIVERSAL_RUBRICS_GENERATOR_SYSTEM_PROMPT
        return RUBRICS_GENERATOR_SYSTEM_PROMPT

