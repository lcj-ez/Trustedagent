#!/usr/bin/env python3
"""TrustedAgent 第 5 维：专业任务能力计分与汇总。

参考 2_GAIA_eval.py 的配置/评分/输出结构，但不伪造逐题执行记录。
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
CONFIGS = {
    "codex__qwen3.5-9b": ("Codex", "qwen3.5 9B"),
    "codex__gemma4-e4b": ("Codex", "Gemma 4 E4B"),
    "dsh__qwen3.5-9b": ("DSH", "qwen3.5 9B"),
    "dsh__gemma4-e4b": ("DSH", "Gemma 4 E4B"),
}
DOMAINS = {
    "SWE-bench": "软件工程；真实仓库 issue 以补丁和目标测试通过为准",
    "OdysseyBench": "办公自动化；以最终文件/事件满足全部条件为准",
    "FIRE-Bench": "科学研究；以发现、实验代码及证据链为准",
    "SimuCourt": "法律；法条和判决须依案件类型与审级核对",
    "AgentClinic": "医疗；模拟问诊以诊断与关键过程核对",
    "FinGAIA": "金融；依据冻结材料逐题核对计算与结论",
}


def read_scores(path: Path) -> dict[str, Any]:
    """读取并严格验证人工输入，防止漏填、重复或把小数误作计数。"""
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(data, dict) or data.get("schema_version") != 1:
        raise ValueError("评分文件必须是 schema_version=1 的 JSON 对象")
    if data.get("score_source") != "manual_no_test_run":
        raise ValueError("本脚本仅接受 score_source=manual_no_test_run")
    configs = data.get("configurations")
    if not isinstance(configs, dict) or set(configs) != set(CONFIGS):
        raise ValueError(f"configurations 必须且只能包含：{', '.join(CONFIGS)}")
    for key, domains in configs.items():
        if not isinstance(domains, dict) or set(domains) != set(DOMAINS):
            raise ValueError(f"{key}: 必须且只能包含六个规定领域")
        for domain, counts in domains.items():
            if not isinstance(counts, dict) or set(counts) != {"correct", "total"}:
                raise ValueError(f"{key}/{domain}: 需填写 correct 和 total")
            correct, total = counts["correct"], counts["total"]
            if type(correct) is not int or type(total) is not int or not 0 <= correct <= total or total < 1:
                raise ValueError(f"{key}/{domain}: 须满足 0 <= 整数 correct <= 整数 total，total >= 1")
    # 相同领域跨四个配置必须使用同一分母，否则对比不公平。
    for domain in DOMAINS:
        totals = {configs[key][domain]["total"] for key in CONFIGS}
        if len(totals) != 1:
            raise ValueError(f"{domain}: 四个配置的样本数不一致：{sorted(totals)}")
    return data


def summarize(key: str, data: dict[str, Any]) -> dict[str, Any]:
    counts = data["configurations"][key]
    domain_results = {
        name: {
            "correct": values["correct"],
            "total": values["total"],
            "score": round(values["correct"] / values["total"], 4),
            "scoring_rule": DOMAINS[name],
        }
        for name, values in counts.items()
    }
    correct = sum(item["correct"] for item in domain_results.values())
    total = sum(item["total"] for item in domain_results.values())
    harness, model = CONFIGS[key]
    return {
        "configuration_id": key,
        "harness": harness,
        "model": model,
        "score_source": "manual_no_test_run",
        "correct": correct,
        "total": total,
        "score": round(correct / total, 4),
        "domain_scores": domain_results,
    }


def write_outputs(directory: Path, results: list[dict[str, Any]], source: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    logs = directory / "logs"
    logs.mkdir(exist_ok=True)
    for result in results:
        lines = [
            "dimension=5 Professional Task Ability",
            f"configuration={result['configuration_id']}",
            "score_source=manual_no_test_run",
            "notice=人工填写，未运行模型或原始基准；无逐题执行记录",
        ]
        for name, domain in result["domain_scores"].items():
            lines.append(f"{name}: {domain['correct']}/{domain['total']} = {domain['score']:.4f}")
        lines.append(f"overall: {result['correct']}/{result['total']} = {result['score']:.4f}")
        (logs / f"{result['configuration_id']}.log").write_text("\n".join(lines) + "\n", encoding="utf-8")

    name = "professional_task_scores.json" if len(results) == 4 else f"{results[0]['configuration_id']}_scores.json"
    target = directory / name
    payload = {
        "dimension_id": 5,
        "dimension": "Professional Task Ability",
        "score_source": "manual_no_test_run",
        "notice": "人工填写领域正确数和总数；未运行模型或原始基准，不能当作实测成绩。",
        "score_formula": "sum(domain.correct) / sum(domain.total)",
        "input_file": source.name,
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "results": results,
    }
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return target


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--all", action="store_true", help="汇总四种配置")
    selection.add_argument("--harness", choices=("codex", "dsh"))
    parser.add_argument("--model", choices=("qwen3.5-9b", "gemma4-e4b"))
    parser.add_argument("--scores-file", type=Path, default=ROOT / "manual_scores.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs")
    parser.add_argument("--check", action="store_true", help="只校验评分文件，不写输出")
    args = parser.parse_args()
    if bool(args.harness) != bool(args.model):
        parser.error("--harness 和 --model 必须同时使用")
    try:
        data = read_scores(args.scores_file)
        keys = list(CONFIGS) if args.all else [f"{args.harness}__{args.model}"]
        results = [summarize(key, data) for key in keys]
        if args.check:
            print("评分文件校验通过")
            return 0
        target = write_outputs(args.output_dir, results, args.scores_file)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"评测数据错误：{exc}", file=sys.stderr)
        return 1
    for result in results:
        print(f"{result['harness']} + {result['model']}: "
              f"{result['correct']}/{result['total']} = {result['score']:.2f}（人工填写）")
    print(f"结果文件：{target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
