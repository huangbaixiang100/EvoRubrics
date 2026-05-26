#!/usr/bin/env python3
"""English docstring."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pyarrow.parquet as pq

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_SPLIT_DIR = SCRIPT_DIR / "healthbench_deepseek_split"
DEFAULT_MANIFEST = DEFAULT_SPLIT_DIR / "split_manifest.json"


def reward_model_to_parseable_gt_text(reward_model: dict | None) -> str:
    """English docstring."""
    if not reward_model or not isinstance(reward_model, dict):
        return ""
    rubrics = reward_model.get("rubrics") or []
    if not rubrics:
        return ""
    lines: list[str] = []
    for i, r in enumerate(rubrics, start=1):
        if not isinstance(r, dict):
            continue
        crit = (r.get("criterion") or "").strip().replace("\n", " ")
        if not crit:
            continue
        pts = r.get("points", 0)
        try:
            pts_i = int(round(float(pts)))
        except (TypeError, ValueError):
            pts_i = 0
        crit = crit.rstrip(".")
        lines.append(f"{i}. {crit}.[points:{pts_i}]")
    return "\n".join(lines)


def load_prompt_id_to_gt_rubrics(parquet_path: Path) -> dict[str, str]:
    """English docstring."""
    if not parquet_path.is_file():
        return {}
    table = pq.read_table(str(parquet_path), columns=["prompt_id", "reward_model"])
    n = table.num_rows
    pids = table.column("prompt_id").to_pylist()
    rms = table.column("reward_model").to_pylist()
    out: dict[str, str] = {}
    for i in range(n):
        pid = str(pids[i]) if pids[i] is not None else ""
        rm = rms[i]
        if not pid:
            continue
        text = reward_model_to_parseable_gt_text(rm if isinstance(rm, dict) else None)
        if text:
            out[pid] = text
    return out


def deepseek_row_to_open_rubrics(
    row: dict,
    prompt_id: str | None = None,
) -> dict:
    instruction = (row.get("instruction") or "").strip()
    inp = (row.get("input") or "").strip()
    history = row.get("history") or []

    prompt_msgs: list[dict[str, str]] = []
    for turn in history:
        if isinstance(turn, (list, tuple)) and len(turn) >= 2:
            prompt_msgs.append({"role": "user", "content": str(turn[0])})
            prompt_msgs.append({"role": "assistant", "content": str(turn[1])})

    if inp:
        last_user = f"{instruction}\n{inp}".strip() if instruction else inp
    else:
        last_user = instruction

    if not last_user:
        raise ValueError("")

    prompt_msgs.append({"role": "user", "content": last_user})

    item: dict = {
        "instruction": last_user,
        "question": last_user,
        "prompt": prompt_msgs,
        "data_source": "healthbench",
        "dataset_mode": "open_rubrics",
        "ground_truth_rubrics": "",
        "golden_answer": row.get("output") or "",
    }
    if prompt_id:
        item["prompt_id"] = prompt_id
    src = row.get("source")
    if src:
        item["source"] = src
    return item


def invert_prompt_index(manifest: dict) -> dict[int, str]:
    m = manifest.get("prompt_id_to_json_index") or {}
    inv: dict[int, str] = {}
    for pid, jidx in m.items():
        inv[int(jidx)] = str(pid)
    return inv


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split-dir", type=Path, default=DEFAULT_SPLIT_DIR)
    ap.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    ap.add_argument("--train-in", type=str, default="healthbench_rl_3k.json")
    ap.add_argument("--test-in", type=str, default="healthbench_test_hard_1k.json")
    ap.add_argument(
        "--train-out",
        type=str,
        default="open_rubrics_healthbench_train_rl_3k.json",
    )
    ap.add_argument(
        "--test-out",
        type=str,
        default="open_rubrics_healthbench_test_hard_1k.json",
    )
    ap.add_argument(
        "--test-golden-parquet",
        type=Path,
        default=None,
        help="",
    )
    args = ap.parse_args()

    split_dir = args.split_dir
    manifest_path = args.manifest
    manifest: dict = {}
    if manifest_path.exists():
        with open(manifest_path, encoding="utf-8") as f:
            manifest = json.load(f)
        idx_to_pid = invert_prompt_index(manifest)
    else:
        idx_to_pid = {}

    golden_parquet = args.test_golden_parquet
    if golden_parquet is None and manifest.get("hard_parquet"):
        golden_parquet = Path(manifest["hard_parquet"])
    pid_to_gt: dict[str, str] = (
        load_prompt_id_to_gt_rubrics(golden_parquet) if golden_parquet else {}
    )
    if pid_to_gt:
        print(
            "[message omitted]"
        )
    else:
        print(
            "[message omitted]"
            ""
        )

    def convert_file(src_name: str, dst_name: str, attach_prompt_id: bool) -> None:
        src_path = split_dir / src_name
        with open(src_path, encoding="utf-8") as f:
            rows = json.load(f)
        out: list[dict] = []
        for i, row in enumerate(rows):
            pid = None
            if attach_prompt_id and idx_to_pid:
                jidx = manifest["test_json_indices"][i]
                pid = idx_to_pid.get(jidx)
            item = deepseek_row_to_open_rubrics(row, prompt_id=pid)
            if attach_prompt_id and pid and pid in pid_to_gt:
                item["ground_truth_rubrics"] = pid_to_gt[pid]
            out.append(item)
        dst_path = split_dir / dst_name
        with open(dst_path, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        print("[message omitted]")

    convert_file(args.train_in, args.train_out, attach_prompt_id=False)
    convert_file(args.test_in, args.test_out, attach_prompt_id=True)


if __name__ == "__main__":
    main()
