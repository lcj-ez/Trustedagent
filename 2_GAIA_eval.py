import os
import json
import subprocess
import shutil
import re
import argparse
from datetime import datetime

# ================= 配置区域 =================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "gaia_data/2023/validation")
METADATA_FILE = os.path.join(DATA_DIR, "metadata.jsonl")
EVAL_OUTPUT_BASE_DIR = os.path.join(BASE_DIR, "gaia_eval_outputs")
WORKSPACE_DIR = os.path.join(BASE_DIR, "gaia_workspace_final")

# 确保输出和工作目录存在
os.makedirs(EVAL_OUTPUT_BASE_DIR, exist_ok=True)
os.makedirs(WORKSPACE_DIR, exist_ok=True)

INVALID_KWS = (
    "[your answer]", "<answer>", "your answer", "tokens used",
    "error", "invalid", "rate limit", "failed", "reconnecting",
)

# 🚨 终极混合 Prompt：同时教它本地读文件、MCP搜索、无头浏览器抓取
PROMPT_TEMPLATE = """You are an autonomous AI agent.
Current working directory: {task_dir}
{file_info}

CRITICAL INSTRUCTIONS:
1. You do NOT have a built-in web_search tool.
2. To search the internet, execute: `agent-web-search "<query>"`
3. To read a specific webpage, execute: `python web_fetch.py <URL>`
4. To read local attachments, write Python script using `pandas`, `openpyxl`, or `PyPDF2`.
5. DO NOT describe what you will do. IMMEDIATELY issue the bash tool call.
6. If a tool times out or fails, make your best educated guess immediately.

TASK:
Answer this question: {question}

Once you know the answer, you MUST end your response exactly with:
FINAL ANSWER: <your answer>
"""

# ================= 工具函数 =================
def extract_clean_answer(reply: str):
    if not reply:
        return None
    matches = re.findall(r'FINAL ANSWER:\s*(.*)', reply, re.IGNORECASE)
    if not matches:
        return None
    ans = re.sub(r'^[\*\_`\'\"\s]+|[\*\_`\'\"\s]+$', '', matches[-1].strip())
    ans = ans.split("\n")[0].strip()
    if ans and not any(kw in ans.lower() for kw in INVALID_KWS):
        return ans
    return None

def load_gaia_data(limit=None, only_files=False, start_idx=0, end_idx=None):
    with open(METADATA_FILE, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f if line.strip()]

    # 筛选带附件的题
    if only_files:
        data = [item for item in data if item.get("file_name")]
                
    # 切片逻辑
    if end_idx is not None:
        data = data[start_idx:end_idx]
    elif start_idx > 0:
        data = data[start_idx:]

    if limit:
        data = data[:limit]
        
    for i, item in enumerate(data):
        if "task_id" not in item:
            item["task_id"] = item.get("id", f"task_{i}")
    return data

# ================= Evaluator 框架层 =================
class BaseEvaluator:
    def __init__(self, data, model_name, output_file):
        self.data = data
        self.model_name = model_name
        self.output_file = output_file
        self.env = os.environ.copy()
        
        # 🚨 强行注入防扣费变量和代理
        self.env["OLLAMA_LOCAL_API_KEY"] = "ollama"
        self.env["OLLAMA_HOST"] = "http://127.0.0.1:11434"
        self.env["DEEPSEEK_API_KEY"] = "" # 强制置空，走本地

    def build_cmd(self, prompt):
        raise NotImplementedError

    def evaluate(self):
        results = []
        for idx, task in enumerate(self.data, 1):
            task_id = task["task_id"]
            question = task["Question"]
            file_name = task.get("file_name", "")
            
            print(f"\n[{idx}/{len(self.data)}] 开始测试 Task: {task_id}")
            task_dir = os.path.abspath(os.path.join(WORKSPACE_DIR, str(task_id)))
            os.makedirs(task_dir, exist_ok=True)

            # 注入 web_fetch 武器库
            fetch_tool_src = os.path.join(BASE_DIR, "web_fetch.py")
            if os.path.exists(fetch_tool_src):
                shutil.copy(fetch_tool_src, os.path.join(task_dir, "web_fetch.py"))

            file_info = ""
            if file_name:
                src = os.path.join(DATA_DIR, file_name)
                if os.path.exists(src):
                    shutil.copy(src, os.path.join(task_dir, file_name))
                    file_info = f"Attached file: {file_name} (in current dir)"
                    print(f"  📎 已复制附件: {file_name}")

            prompt = PROMPT_TEMPLATE.format(file_info=file_info, task_dir=task_dir, question=question)
            cmd = self.build_cmd(prompt)

            print(f"  🚀 调用框架求解...")
            try:
                res = subprocess.run(
                    cmd, cwd=task_dir, capture_output=True, env=self.env,
                    text=True, timeout=1800
                )
                reply = res.stdout + "\n" + res.stderr
            except subprocess.TimeoutExpired:
                print(f"  ❌ 超时。")
                reply = "TIMEOUT"
            except Exception as e:
                print(f"  ❌ 异常: {e}")
                reply = str(e)

            answer = extract_clean_answer(reply)
            if answer:
                print(f"  ✅ 提取成功: {answer}")
            else:
                print(f"  ❌ 未能提取到有效答案。")

            results.append({
                "task_id": task_id,
                "question": question,
                "model_answer": answer,
                "ground_truth": task.get("Final answer", ""),
                "raw_output": reply
            })

        with open(self.output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\n🏆 评测完成！结果已保存至: {self.output_file}")

class CodexEvaluator(BaseEvaluator):
    def build_cmd(self, prompt):
        # 注意：不加 --search，依靠 env 继承和本地 mcp
        return [
            "codex", "exec",
            "--skip-git-repo-check",
            "--dangerously-bypass-approvals-and-sandbox",
            "-m", self.model_name,
            "-c", 'model_reasoning_effort="low"',
            prompt
        ]

class DSHEvaluator(BaseEvaluator):
    def build_cmd(self, prompt):
        return ["npx", "--yes", "@deepseek-ai/dsh", "headless", prompt]

# ================= 主程序 =================
def main():
    parser = argparse.ArgumentParser(description="GAIA Ultimate Evaluator")
    parser.add_argument("--agent", type=str, default="codex", choices=["codex", "dsh"], help="Framework: codex or dsh")
    parser.add_argument("--model", type=str, required=True, help="Model name e.g., qwen3.5-9b or gemma3:4b")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of tasks")
    parser.add_argument("--only-files", action="store_true", help="Only run tasks with attachments")
    parser.add_argument("--start", type=int, default=0, help="Start index (0-based)")
    parser.add_argument("--end", type=int, default=None, help="End index (exclusive)")
    args = parser.parse_args()

    data = load_gaia_data(
        limit=args.limit, 
        only_files=args.only_files,
        start_idx=args.start,
        end_idx=args.end
    )
    print(f"✅ 成功加载 {len(data)} 题。")

    safe_model_name = args.model.split("/")[-1].replace(":", "_")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(EVAL_OUTPUT_BASE_DIR, f"{args.agent}_{safe_model_name}_{timestamp}.json")

    if args.agent == "codex":
        evaluator = CodexEvaluator(data, args.model, output_file)
    elif args.agent == "dsh":
        evaluator = DSHEvaluator(data, args.model, output_file)
    else:
        raise ValueError("Unsupported agent")

    evaluator.evaluate()

if __name__ == "__main__":
    main()
