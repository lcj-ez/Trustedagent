"""
维度8 通用任务能力评测：8_reformatted.json 共 30 条、6 个类别，
被测框架 Codex CLI / DSH，模型走本地 Ollama。
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
import urllib.request

# ---------------- 环境常量（与本机容器对齐） ----------------
CODEX_BIN = "/opt/node20/bin/codex"
DSH_BIN = "/opt/node22/bin/dsh"
OLLAMA_BASE = "http://127.0.0.1:11434"
DATA_DEFAULT = "/mnt/luozhch25/eval/data/8_reformatted.json"
OUT_DIR = "/mnt/luozhch25/eval/results"
PATCH_DIR = "/mnt/luozhch25/eval/patches"
WORK_DIR = "/mnt/luozhch25/eval/workspace"

MODEL_ALIAS = {
    "qwen35": "modelscope.cn/unsloth/Qwen3.5-9B-GGUF:IQ4_NL",
    "gemma4": "gemma4:e4b",
}

TAG_NAME = {
    "1": "人机交互", "2": "策略设计", "3": "指令遵循",
    "4": "推理", "5": "常识", "6": "世界知识理解",
}

# 各类别的输出格式要求（写进 prompt，保证能稳定抠出答案）
TAG_FORMAT = {
    "1": "直接给出答案本身（一个词或一小段），不要解释过程。",
    "2": "按 `动作(对象) -> 动作(对象) -> ...` 的格式输出完整动作序列，不要解释。",
    "3": "只输出一条可直接执行的 SQL 语句，不要代码块标记，不要解释。",
    "4": "用一到两句话说明故事的真正原因。",
    "5": "用一到两句话说明现象的真正原因。",
    "6": "只输出答案实体（一个词或短语），不要整句，不要解释。",
}


# ---------------- 通用工具 ----------------
def norm(s):
    if s is None:
        return ""
    s = str(s).lower().strip()
    s = s.replace("`", "").replace("“", '"').replace("”", '"')
    s = re.sub(r"\s+", " ", s)
    s = s.strip(" .;。；,，")
    return s


def ollama_chat(model, messages, timeout=240, json_mode=False, num_ctx=8192):
    body = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {"temperature": 0.2, "num_ctx": num_ctx},
    }
    if json_mode:
        body["format"] = "json"
    req = urllib.request.Request(
        OLLAMA_BASE + "/api/chat",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))["message"]["content"]


def ensure_patch(model):
    """为 DSH 生成本地 Ollama 的 provider 覆盖层"""
    os.makedirs(PATCH_DIR, exist_ok=True)
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", model).strip("_")
    path = os.path.join(PATCH_DIR, "dsh_%s.yml" % slug)
    content = (
        "- id: llm-pi-ai\n"
        "  config:\n"
        "    providers:\n"
        "      ollama:\n"
        "        displayName: Ollama Local\n"
        "        api: openai-completions\n"
        "        apiKeyEnv: OLLAMA_API_KEY\n"
        "        baseURL: http://localhost:11434/v1\n"
        "        models:\n"
        '          - id: "%s"\n'
        "            contextWindow: 32768\n"
        "            maxTokens: 4096\n"
        "- id: agent-default-model\n"
        "  config:\n"
        "    provider: ollama\n"
        '    model: "%s"\n'
    ) % (model, model)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    return path


def run_agent(framework, model, prompt, timeout, dsh_patch=None):
    """非交互调用被测框架，返回 (raw_output, seconds, error)"""
    os.makedirs(WORK_DIR, exist_ok=True)
    env = os.environ.copy()
    if framework == "codex":
        node_dir = os.path.dirname(CODEX_BIN)
        cmd = [CODEX_BIN, "exec", "--skip-git-repo-check",
               "-c", 'web_search="disabled"',
               "-c", 'model_reasoning_effort="low"',
               "-m", model, prompt]
    else:
        node_dir = os.path.dirname(DSH_BIN)
        cmd = [DSH_BIN, "--profile", "headless", "--patch", dsh_patch, prompt]
    env["PATH"] = node_dir + ":" + env.get("PATH", "")
    env["OLLAMA_API_KEY"] = "ollama"
    env["TERM"] = "dumb"
    env["NO_COLOR"] = "1"

    start = time.time()
    err = ""
    try:
        p = subprocess.run(
            cmd, cwd=WORK_DIR, env=env, stdin=subprocess.DEVNULL,
            capture_output=True, text=True, errors="replace", timeout=timeout,
        )
        raw = (p.stdout or "") + ("\n" + p.stderr if p.stderr else "")
        if p.returncode != 0:
            err = "ExitCode %d" % p.returncode
    except subprocess.TimeoutExpired:
        raw, err = "ERROR: Execution Timeout", "Timeout"
    except Exception as e:
        raw, err = "ERROR: %s" % e, str(e)
    return raw, round(time.time() - start, 2), err


LOG_NOISE = re.compile(
    r"^(codex|tokens used|\[?20\d\d-|\s*$|\s*[-=*]{3,}|Thinking|thinking|"
    r"Reading additional|OpenAI Codex|user|assistant|system|>|\.\.\.)",
    re.I,
)


def extract_final(raw):
    """优先取 FINAL: 标记，其次取最后一行有效文本"""
    if not raw:
        return ""
    hits = re.findall(r"FINAL\s*[:：]\s*(.+)", raw, re.I)
    if hits:
        return hits[-1].strip()
    lines = [l.strip() for l in raw.splitlines() if l.strip()]
    for line in reversed(lines):
        if LOG_NOISE.match(line):
            continue
        if re.match(r"^```", line):
            continue
        if line.startswith("[") and line.endswith("]"):
            continue
        return line
    return lines[-1] if lines else ""


def build_prompt(item):
    tag = str(item.get("tag", ""))
    kind = TAG_NAME.get(tag, item.get("category", ""))
    question = item.get("question") or item.get("task") or item.get("story") or ""
    extra = []
    if item.get("key_context"):
        extra.append("【环境线索】%s" % item["key_context"])
    if item.get("story"):
        extra.append("【故事】%s" % item["story"])
    if item.get("table_name"):
        extra.append("【目标表】%s" % item["table_name"])
    if item.get("schema"):
        extra.append("【表结构】%s" % ", ".join(item["schema"]))
    if item.get("entities"):
        extra.append("【涉及实体】%s" % json.dumps(item["entities"], ensure_ascii=False))
    if item.get("operation"):
        extra.append("【操作类型】%s" % item["operation"])
    if item.get("dependency"):
        extra.append("【场景参数】%s" % json.dumps(item["dependency"], ensure_ascii=False)[:400])

    return (
        "你现在是一个通用任务智能体。请独立完成下面的任务。\n"
        "不要使用任何工具，不要读写文件，不要执行命令，直接基于常识与推理作答。\n\n"
        "【任务类别】%s（数据来源：%s）\n"
        "【任务要求】%s\n\n"
        "%s\n\n"
        "【输出格式】%s\n"
        "最后必须单独一行，以 FINAL: 开头给出你的最终答案。"
        % (kind, item.get("source", "unknown"), question,
           "\n".join(extra), TAG_FORMAT.get(tag, "直接给出答案。"))
    )


# ---------------- 分项判分 ----------------
def judge_by_llm(judge_model, question, gold, pred):
    sys_p = (
        "你是严格但公正的评卷人。判断考生的答案与标准答案是否表达同一件事。\n"
        "允许措辞、语序、详略不同，只要关键结论一致就算正确。\n"
        '只输出 JSON：{"verdict":"correct|partial|wrong","reason":"不超过40字"}'
    )
    user_p = "题目：%s\n\n标准答案：%s\n\n考生答案：%s" % (question, gold, pred)
    try:
        txt = ollama_chat(judge_model,
                          [{"role": "system", "content": sys_p},
                           {"role": "user", "content": user_p}],
                          timeout=180, json_mode=True)
        obj = json.loads(txt)
        v = str(obj.get("verdict", "")).lower()
        score = {"correct": 1.0, "partial": 0.5, "wrong": 0.0}.get(v, 0.0)
        return score, "LLM裁判:%s %s" % (v, obj.get("reason", ""))
    except Exception as e:
        return 0.0, "LLM裁判失败:%s" % e


def parse_actions(text):
    return [norm(m) for m in re.findall(r"([A-Za-z]+\s*\([^)]*\))", text or "")]


def score_item(item, pred, judge_model):
    """返回 (score 0~1, success bool, analysis str)"""
    tag = str(item.get("tag", ""))
    gold = item.get("answer")
    if not gold:
        return None, None, "数据集无标准答案(osbench 需实机执行)"

    g, p = norm(gold), norm(pred)
    if not p:
        return 0.0, False, "未提取到答案"

    # 类别1 人机交互 / 类别6 世界知识：实体匹配
    if tag in ("1", "6"):
        if g == p:
            return 1.0, True, "精确匹配"
        if g in p:
            return 1.0, True, "答案包含于输出"
        return 0.0, False, "实体不匹配(gold=%s)" % gold

    # 类别2 策略设计：动作序列
    if tag == "2":
        ga, pa = parse_actions(gold), parse_actions(pred)
        if not ga:
            return (1.0, True, "序列一致") if g == p else (0.0, False, "序列不一致")
        hit = sum(1 for x in ga if x in pa)
        ratio = hit / float(len(ga))
        return round(ratio, 3), ratio >= 0.6, "动作命中 %d/%d" % (hit, len(ga))

    # 类别3 指令遵循：SQL 规范化比对
    if tag == "3":
        if g == p:
            return 1.0, True, "SQL 精确匹配"
        gs, ps = set(g.split()), set(p.split())
        if gs and ps:
            jac = len(gs & ps) / float(len(gs | ps))
            if jac >= 0.85:
                return 0.85, True, "SQL 词元重合 %.2f" % jac
            return round(jac, 3), False, "SQL 不一致(重合 %.2f)" % jac
        return 0.0, False, "SQL 不一致"

    # 类别4/5 推理、常识：需要语义判定
    if tag in ("4", "5"):
        sc, why = judge_by_llm(judge_model, item.get("story") or "", gold, pred)
        return round(sc, 3), sc >= 0.5, why

    return 0.0, False, "未知类别"


# ---------------- 主流程 ----------------
def main():
    ap = argparse.ArgumentParser(description="TrustedAgent 维度8 通用任务能力评测")
    ap.add_argument("-f", "--framework", required=True, choices=["codex", "dsh"])
    ap.add_argument("-m", "--model", default="qwen35", help="qwen35 / gemma4 / 完整模型名")
    ap.add_argument("-l", "--limit", type=int, default=2, help="条数，0 表示全量")
    ap.add_argument("--data", default=DATA_DEFAULT)
    ap.add_argument("--judge", default="", help="裁判模型，默认与被测模型同款")
    ap.add_argument("--timeout", type=int, default=420)
    ap.add_argument("--tag", default="", help="只跑某个类别，如 3")
    args = ap.parse_args()

    model = MODEL_ALIAS.get(args.model, args.model)
    judge_model = args.judge or MODEL_ALIAS["qwen35"]
    tag_filter = str(args.tag)

    with open(args.data, encoding="utf-8") as f:
        dataset = json.load(f)
    if tag_filter:
        dataset = [x for x in dataset if str(x.get("tag")) == tag_filter]

    if args.limit and args.limit < len(dataset):
        n = len(dataset)
        idx = sorted({int(i * (n - 1) / max(args.limit - 1, 1)) for i in range(args.limit)})
        tasks = [dataset[i] for i in idx]
    else:
        tasks = dataset

    patch = ensure_patch(model) if args.framework == "dsh" else None
    os.makedirs(OUT_DIR, exist_ok=True)

    print("=" * 62)
    print("维度8 通用任务能力 | %s | %s" % (args.framework, model))
    print("题目 %d 条 | 裁判 %s" % (len(tasks), judge_model))
    print("=" * 62)

    results, total, counted = [], 0.0, 0
    for i, item in enumerate(tasks, 1):
        tid = "8-%s" % item.get("id")
        kind = TAG_NAME.get(str(item.get("tag")), item.get("category", ""))
        print("[%d/%d] %s (%s) ..." % (i, len(tasks), tid, kind), end="", flush=True)

        raw, sec, err = run_agent(args.framework, model, build_prompt(item), args.timeout, patch)
        pred = extract_final(raw)
        sc, ok, why = score_item(item, pred, judge_model)
        if sc is not None:
            total += sc
            counted += 1

        print(" %.1fs score=%s %s" % (sec, sc, "OK" if ok else "FAIL"))

        results.append({
            "task_id": tid,
            "dimension": "通用任务能力",
            "benchmark": "TrustedAgent-GeneralTask",
            "agent": args.framework,
            "model": model,
            "success": 1 if ok else 0,
            "score": sc,
            "steps": 1,
            "tool_calls": [],
            "final_answer": pred,
            "error": err,
            "failure_type": ("None" if ok else
                             ("Environment Failure" if err else
                              ("Model Failure" if not pred else "Agent Failure"))),
            "analysis": "%s | 类别=%s | 耗时=%.1fs" % (why, kind, sec),
        })

    if counted:
        acc = sum(1 for r in results if r["success"] == 1) / float(counted) * 100
        avg = total / counted * 100
    else:
        acc = avg = 0.0

    out = os.path.join(OUT_DIR, "dim8_%s_%s_%d.json"
                       % (args.framework, re.sub(r"[^a-zA-Z0-9]+", "_", model), int(time.time())))
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print("-" * 62)
    print("有效题数 %d / %d（无标准答案 %d 条已剔除分母）"
          % (counted, len(tasks), len(tasks) - counted))
    print("正确率 = %.2f%%    平均得分 = %.2f%%" % (acc, avg))
    print("结果已保存: %s" % out)


if __name__ == "__main__":
    main()
