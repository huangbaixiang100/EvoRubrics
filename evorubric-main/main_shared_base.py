"""Entry point for shared-base dual-LoRA adversarial training."""
from typing import Any, Optional
import os
import sys
from pathlib import Path
import hydra
import ray
from omegaconf import OmegaConf

package_root = Path(__file__).resolve().parent
project_root = package_root.parent
sys.path.insert(0, str(project_root / "third_party" / "verl"))
sys.path.insert(0, str(project_root / "rubric_guidance"))
sys.path.insert(0, str(package_root))
sys.path.insert(0, str(project_root))

from verl.trainer.constants_ppo import get_ppo_ray_runtime_env
from shared_base_trainer import SharedBaseAdversarialTrainer


def _strip_env_value(v: Optional[str]) -> Optional[str]:
    if v is None:
        return None
    s = str(v).strip()
    return s if s else None


def _redact_for_log(obj: Any) -> Any:
    """Redact API keys and secrets before logging Ray init kwargs."""
    secret_markers = ("API_KEY", "SECRET", "TOKEN", "PASSWORD")
    if isinstance(obj, dict):
        return {
            k: ("***" if any(m in k.upper() for m in secret_markers) else _redact_for_log(v))
            for k, v in obj.items()
        }
    if isinstance(obj, list):
        return [_redact_for_log(x) for x in obj]
    return obj


@hydra.main(config_path="config", config_name="shared_base_config", version_base=None)
def main(config):
    """Hydra entry point."""
    run_shared_base_adversarial_training(config)


def run_shared_base_adversarial_training(config) -> None:
    """Initialize Ray and run dual-LoRA adversarial training."""
    print("=" * 80)
    print("EvoRubrics shared-base dual-LoRA training")
    print("=" * 80)

    if not ray.is_initialized():
        default_runtime_env = get_ppo_ray_runtime_env()
        ray_init_kwargs = config.get("ray_kwargs", {}).get("ray_init", {})
        runtime_env_kwargs = ray_init_kwargs.get("runtime_env", {})

        runtime_env = OmegaConf.merge(default_runtime_env, runtime_env_kwargs)
        _judge_keys = (
            "LLM_JUDGE_DEBUG_LOG",
            "LLM_JUDGE_DEBUG_FULL",
            "DEEPSEEK_DEBUG_LOG",
            "LLM_EVALUATOR_TYPE",
            "DEEPSEEK_API_KEY",
            "DEEPSEEK_BASE_URL",
            "DEEPSEEK_MODEL",
        )
        _extra_env = {}
        for k in _judge_keys:
            v = _strip_env_value(os.environ.get(k))
            if v is not None:
                _extra_env[k] = v
        if _extra_env:
            base_env = OmegaConf.to_container(runtime_env.get("env_vars", {}), resolve=True) or {}
            base_env.update(_extra_env)
            runtime_env = OmegaConf.merge(runtime_env, OmegaConf.create({"env_vars": base_env}))
        ray_init_kwargs = OmegaConf.create({**ray_init_kwargs, "runtime_env": runtime_env})

        safe_log = _redact_for_log(OmegaConf.to_container(ray_init_kwargs, resolve=True))
        print(f"Ray init kwargs: {safe_log}")
        ray.init(**OmegaConf.to_container(ray_init_kwargs))

    task_runner_class = ray.remote(num_cpus=1)(SharedBaseAdversarialTrainer)
    runner = task_runner_class.remote()

    ray.get(runner.run.remote(config))

    print("\n" + "=" * 80)
    print("Training finished.")
    print("=" * 80)


if __name__ == "__main__":
    main()
