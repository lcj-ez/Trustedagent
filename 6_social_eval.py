"""
维度6 社会与协作能力评测：6_reformatted.json 共 63 条 SOTOPIA 场景。
被测框架扮演角色 A，本地模型扮演角色 B 并担任裁判，
得分 score = (goal * 0.8 + relationship * 0.2) / 10。
"""

import argparse
import ast
import json
import os
import re
import subprocess
import sys
import time
import urllib.request

CODEX_BIN = "/opt/node20/bin/codex"
DSH_BIN = "/opt/node22/bin/dsh"
OLLAMA_BASE = "http://127.0.0.1:11434"
DATA_DEFAULT = "/mnt/luozhch25/eval/data/6_reformatted.json"
OUT_DIR = "/mnt/luozhch25/eval/results"
PATCH_DIR = "/mnt/luozhch25/eval/patches"
WORK_DIR = "/mnt/luozhch25/eval/workspace"

MODEL_ALIAS = {
    "qwen35": "modelscope.cn/unsloth/Qwen3.5-9B-GGUF:IQ4_NL",
    "gemma4": "gemma4:e4b",
}


def to_dict(obj):
    """数据里的 agents_background / social_goals 是 Python dict 的字符串表示"""
    if isinstance(obj, dict):
        return obj
    if not obj:
        return {}
    for parser in (ast.literal_eval, json.loads):
        try:
            val = parser(obj)
            if isinstance(val, dict):
                return val
        except Exception:
            continue
    return {}


def ollama_chat(model, messages, timeout=240, json_mode=False, num_ctx=8192):
    body = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {"temperature": 0.3, "num_ctx": num_ctx},
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


NOISE = re.compile(
    r"^(codex|tokens used|\[?20\d\d-|\s*$|\s*[-=*]{3,}|Thinking|thinking|"
    r"Reading additional|OpenAI Codex|user|assistant|system|>|\.\.\.|FINAL)",
    re.I,
)


def extract_speech(raw):
    """从框架输出里抽出角色台词"""
    if not raw:
        return ""
    hits = re.findall(r"(?:SAY|FINAL|回复)\s*[:：]\s*(.+)", raw, re.I)
    if hits:
        return hits[-1].strip().strip('"“”')
    lines = [l.strip() for l in raw.splitlines() if l.strip()]
    for line in reversed(lines):
        if NOISE.match(line) or re.match(r"^```", line):
            continue
        if line.startswith("[") and line.endswith("]"):
            continue
        return line
    return lines[-1] if lines else ""


def build_speech_prompt(item, me, other, my_bg, my_goal, transcript, turns_left):
    hist = "\n".join("%s：%s" % (s, t) for s, t in transcript) or "（暂无对话）"
    return (
        "你正在参与一场社交对话的角色扮演，必须完全以「%s」的身份说话。\n"
        "不要使用任何工具，不要解释你在扮演，不要跳出角色。\n\n"
        "【场景】%s\n\n"
        "【你的人设】%s\n\n"
        "【你的社交目标】%s\n\n"
        "【当前对话】\n%s\n\n"
        "现在轮到你（%s）对「%s」说话，还剩 %d 轮。"
        "只输出你说的话本身，一到三句，不要加引号，不要加名字前缀。"
        % (me, item.get("scenario", ""), my_bg, my_goal, hist, me, other, turns_left)
    )


def opponent_reply(judge_model, item, other, other_bg, other_goal, transcript):
    hist = "\n".join("%s：%s" % (s, t) for s, t in transcript) or "（暂无对话）"
    sys_p = (
        "你在角色扮演。完全以「%s」的身份说话，人设：%s。你的社交目标是：%s。\n"
        "只输出一句到两句对话内容，不要加名字前缀，不要加引号，不要旁白。"
        % (other, other_bg, other_goal)
    )
    user_p = "【场景】%s\n\n【对话记录】\n%s\n\n现在轮到你说话。" % (
        item.get("scenario", ""), hist)
    try:
        return ollama_chat(judge_model,
                           [{"role": "system", "content": sys_p},
                            {"role": "user", "content": user_p}],
                           timeout=180).strip().strip('"“”')
    except Exception:
        return "……"


def judge_dialogue(judge_model, item, me, other, goals, transcript):
    hist = "\n".join("%s：%s" % (s, t) for s, t in transcript)
    sys_p = (
        "你是 SOTOPIA 社交能力评测的裁判。根据对话判断 %s 的表现。\n"
        "goal：%s 的社交目标达成程度，0-10 分。\n"
        "relationship：对话中对 %s 的尊重、共情与关系维护，0-10 分。\n"
        "只输出 JSON：{\"goal\": 数字, \"relationship\": 数字, \"reason\": \"不超过50字\"}"
        % (me, me, other)
    )
    user_p = (
        "【场景】%s\n\n【%s 的目标】%s\n\n【%s 的目标】%s\n\n【对话记录】\n%s"
        % (item.get("scenario", ""), me, goals.get(me, ""),
           other, goals.get(other, ""), hist)
    )
    try:
        obj = json.loads(ollama_chat(judge_model,
                                     [{"role": "system", "content": sys_p},
                                      {"role": "user", "content": user_p}],
                                     timeout=240, json_mode=True))
        g = float(obj.get("goal", 0))
        r = float(obj.get("relationship", 0))
        g = max(0.0, min(10.0, g))
        r = max(0.0, min(10.0, r))
        return g, r, str(obj.get("reason", ""))[:120]
    except Exception as e:
        return 0.0, 0.0, "裁判失败:%s" % e


def main():
    ap = argparse.ArgumentParser(description="TrustedAgent 维度6 社会与协作能力评测")
    ap.add_argument("-f", "--framework", required=True, choices=["codex", "dsh"])
    ap.add_argument("-m", "--model", default="qwen35")
    ap.add_argument("-l", "--limit", type=int, default=2, help="条数，0 表示全量 63")
    ap.add_argument("--rounds", type=int, default=2, help="被测角色发言轮数")
    ap.add_argument("--data", default=DATA_DEFAULT)
    ap.add_argument("--judge", default="")
    ap.add_argument("--timeout", type=int, default=300)
    args = ap.parse_args()

    model = MODEL_ALIAS.get(args.model, args.model)
    judge_model = args.judge or MODEL_ALIAS["qwen35"]

    with open(args.data, encoding="utf-8") as f:
        dataset = json.load(f)

    if args.limit and args.limit < len(dataset):
        n = len(dataset)
        idx = sorted({int(i * (n - 1) / max(args.limit - 1, 1)) for i in range(args.limit)})
        tasks = [dataset[i] for i in idx]
    else:
        tasks = dataset

    patch = ensure_patch(model) if args.framework == "dsh" else None
    os.makedirs(OUT_DIR, exist_ok=True)

    print("=" * 62)
    print("维度6 社会与协作能力 | %s | %s | %d 轮" % (args.framework, model, args.rounds))
    print("场景 %d 条 | 对手与裁判 %s" % (len(tasks), judge_model))
    print("=" * 62)

    results, total = [], 0.0
    for i, item in enumerate(tasks, 1):
        tid = "6-%s" % item.get("id")
        bgs = to_dict(item.get("agents_background"))
        goals = to_dict(item.get("social_goals"))
        roles = list(bgs.keys()) or ["Agent A"]
        me = roles[0]
        other = roles[1] if len(roles) > 1 else "对方"
        other_bg = bgs.get(other, "一个普通人")
        other_goal = goals.get(other, "自然地参与对话")
        my_bg = bgs.get(me, "一个普通人")
        my_goal = goals.get(me, "完成对话目标")

        print("[%d/%d] %s (%s) ..." % (i, len(tasks), tid, item.get("codename", "")),
              end="", flush=True)

        transcript = []
        start = time.time()
        err = ""
        # 对手先说开场，制造真实对话情境
        opening = opponent_reply(judge_model, item, other, other_bg, other_goal, transcript)
        transcript.append((other, opening))

        turns = 0
        for t in range(args.rounds):
            prompt = build_speech_prompt(item, me, other, my_bg, my_goal,
                                         transcript, args.rounds - t)
            raw, sec, err = run_agent(args.framework, model, prompt, args.timeout, patch)
            said = extract_speech(raw)
            if not said:
                break
            transcript.append((me, said))
            turns += 1
            if t < args.rounds - 1:
                reply = opponent_reply(judge_model, item, other, other_bg, other_goal, transcript)
                transcript.append((other, reply))

        elapsed = round(time.time() - start, 2)
        g, rel, reason = judge_dialogue(judge_model, item, me, other, goals, transcript)
        sc = round((g * 0.8 + rel * 0.2) / 10.0, 3)
        ok = sc >= 0.5
        if sc:
            total += sc

        print(" %.1fs goal=%.0f rel=%.0f score=%.2f %s"
              % (elapsed, g, rel, sc, "OK" if ok else "FAIL"))

        results.append({
            "task_id": tid,
            "dimension": "社会与协作能力",
            "benchmark": "SOTOPIA",
            "agent": args.framework,
            "model": model,
            "success": 1 if ok else 0,
            "score": sc,
            "steps": turns,
            "tool_calls": [],
            "final_answer": " | ".join("%s: %s" % (s, t) for s, t in transcript)[:1200],
            "error": err,
            "failure_type": ("None" if ok else
                             ("Environment Failure" if err else "Agent Failure")),
            "analysis": "goal=%.0f/10 relationship=%.0f/10 | %s | %d 轮"
                        % (g, rel, reason, turns),
        })

    acc = sum(1 for r in results if r["success"] == 1) / float(len(results)) * 100 if results else 0.0
    avg = total / len(results) * 100 if results else 0.0

    out = os.path.join(OUT_DIR, "dim6_%s_%s_%d.json"
                       % (args.framework, re.sub(r"[^a-zA-Z0-9]+", "_", model), int(time.time())))
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print("-" * 62)
    print("场景数 %d | 正确率 = %.2f%% | 平均得分 = %.2f%%" % (len(results), acc, avg))
    print("结果已保存: %s" % out)


if __name__ == "__main__":
    main()

