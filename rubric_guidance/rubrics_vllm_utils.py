import os
import subprocess
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from openai import OpenAI
from tqdm.auto import tqdm

try:
    from rubric_guidance.prompts import UNIVERSAL_RUBRICS_GENERATOR_SYSTEM_PROMPT
    from rubric_guidance import rubric_guidance as _rg
except ModuleNotFoundError:
    from prompts import UNIVERSAL_RUBRICS_GENERATOR_SYSTEM_PROMPT
    import rubric_guidance as _rg


def _snap_max_lora_rank_for_vllm(lora_rank: int) -> int:
    """English docstring."""
    ranks = [8, 16, 32, 64, 128, 256, 320, 512]
    for r in ranks:
        if lora_rank <= r:
            return r
    return ranks[-1]


@dataclass
class RubricsVLLMConfig:
    """English docstring."""

    model_path: str
    served_model_name: str
    host: str = "127.0.0.1"
    port: int = 18000
    gpu_memory_utilization: float = 0.75
    start_vllm: bool = True
    keep_user_site: bool = False
    trust_remote_code: bool = False
    api_key: str = "EMPTY"
    timeout_s: float = 600.0
    request_timeout_s: float = 180.0
    max_tokens: int = 1024
    temperature: float = 0.6
    max_workers: int = 16
    progress_bar: bool = True
    lora_adapter_path: Optional[str] = None
    lora_adapter_name: str = "rubrics"
    max_lora_rank: int = 32

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}/v1"

    @property
    def openai_model_id(self) -> str:
        """English docstring."""
        if self.lora_adapter_path:
            return self.lora_adapter_name
        return self.served_model_name


class RubricsVLLMRuntime:
    def __init__(self, cfg: RubricsVLLMConfig):
        self.cfg = cfg
        self.proc: Optional[subprocess.Popen] = None
        self.client: Optional[OpenAI] = None

    def start(self) -> None:
        if self.cfg.start_vllm:
            cmd = [
                sys.executable,
                "-m",
                "vllm.entrypoints.openai.api_server",
                "--model",
                self.cfg.model_path,
                "--served-model-name",
                self.cfg.served_model_name,
                "--host",
                self.cfg.host,
                "--port",
                str(self.cfg.port),
                "--gpu-memory-utilization",
                str(self.cfg.gpu_memory_utilization),
            ]
            if self.cfg.trust_remote_code:
                cmd.append("--trust-remote-code")
            if self.cfg.lora_adapter_path:
                lp = os.path.abspath(os.path.expanduser(str(self.cfg.lora_adapter_path)))
                if not os.path.isdir(lp):
                    raise FileNotFoundError("")
                snap_rank = _snap_max_lora_rank_for_vllm(max(8, int(self.cfg.max_lora_rank)))
                cmd.extend(
                    [
                        "--enable-lora",
                        "--max-loras",
                        "4",
                        "--max-lora-rank",
                        str(snap_rank),
                        "--lora-modules",
                        f"{self.cfg.lora_adapter_name}={lp}",
                    ]
                )
            env = os.environ.copy()
            if not self.cfg.keep_user_site:
                env["PYTHONNOUSERSITE"] = "1"
            self.proc = subprocess.Popen(cmd, env=env)
        self._wait_ready()
        self.client = OpenAI(
            api_key=self.cfg.api_key,
            base_url=self.cfg.base_url,
            timeout=self.cfg.request_timeout_s,
        )

    def close(self) -> None:
        if self.proc is not None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait(timeout=10)
            self.proc = None

    def _wait_ready(self) -> None:
        url = f"{self.cfg.base_url}/models"
        deadline = time.time() + self.cfg.timeout_s
        last_err: Optional[Exception] = None
        while time.time() < deadline:
            if self.proc is not None and self.proc.poll() is not None:
                raise RuntimeError("")
            try:
                req = urllib.request.Request(url, method="GET")
                with urllib.request.urlopen(req, timeout=5) as resp:
                    if resp.status == 200:
                        return
            except Exception as e:  # noqa: BLE001
                last_err = e
            time.sleep(2)
        raise RuntimeError("")

    @staticmethod
    def build_open_rubrics_chat_messages(
        prompt_messages: Optional[List[Dict[str, Any]]],
        instruction: str = "",
        fallback_question: str = "",
    ) -> List[Dict[str, str]]:
        return _rg.build_open_rubrics_chat_messages(
            prompt_messages, instruction=instruction, fallback_question=fallback_question
        )

    @staticmethod
    def build_rubrics_user_prompt(question: str) -> str:
        return _rg.build_rubrics_user_prompt(question)

    @staticmethod
    def build_rubric_guided_query(question: str, generated_rubrics: str) -> str:
        return _rg.build_rubric_guided_query(question, generated_rubrics)

    def generate_rubrics_from_messages(self, messages: List[Dict[str, str]]) -> str:
        """English docstring."""
        if self.client is None:
            raise RuntimeError("")
        comp = self.client.chat.completions.create(
            model=self.cfg.openai_model_id,
            messages=messages,
            temperature=self.cfg.temperature,
            max_tokens=self.cfg.max_tokens,
            extra_body={
                "chat_template_kwargs": {
                    "enable_thinking": False,
                    "thinking": False,
                },
            },
        )
        return (comp.choices[0].message.content or "").strip()

    def generate_rubrics(self, question: str) -> str:
        """English docstring."""
        messages = [
            {"role": "system", "content": UNIVERSAL_RUBRICS_GENERATOR_SYSTEM_PROMPT},
            {"role": "user", "content": self.build_rubrics_user_prompt(question)},
        ]
        return self.generate_rubrics_from_messages(messages)

    def generate_rubrics_batch(self, questions: List[str]) -> List[str]:
        if not questions:
            return []
        out = [""] * len(questions)
        workers = max(1, min(self.cfg.max_workers, len(questions)))
        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_to_idx = {
                executor.submit(self.generate_rubrics, q): i for i, q in enumerate(questions)
            }
            done_iter = as_completed(future_to_idx)
            if self.cfg.progress_bar:
                done_iter = tqdm(
                    done_iter,
                    total=len(future_to_idx),
                    desc="Rubrics batch",
                    unit="sample",
                    dynamic_ncols=True,
                    smoothing=0.05,
                )
            for future in done_iter:
                idx = future_to_idx[future]
                try:
                    out[idx] = future.result()
                except Exception:  # noqa: BLE001
                    out[idx] = ""
        return out

    def generate_rubrics_batch_messages(self, messages_list: List[List[Dict[str, str]]]) -> List[str]:
        """English docstring."""
        if not messages_list:
            return []
        out = [""] * len(messages_list)
        workers = max(1, min(self.cfg.max_workers, len(messages_list)))
        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_to_idx = {
                executor.submit(self.generate_rubrics_from_messages, m): i
                for i, m in enumerate(messages_list)
            }
            done_iter = as_completed(future_to_idx)
            if self.cfg.progress_bar:
                done_iter = tqdm(
                    done_iter,
                    total=len(future_to_idx),
                    desc="Rubrics batch",
                    unit="sample",
                    dynamic_ncols=True,
                    smoothing=0.05,
                )
            for future in done_iter:
                idx = future_to_idx[future]
                try:
                    out[idx] = future.result()
                except Exception:  # noqa: BLE001
                    out[idx] = ""
        return out
