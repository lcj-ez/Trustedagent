import os
import json
import subprocess
import shutil
import re
import argparse
import requests
from datetime import datetime

# ================= 配置区域 =================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(BASE_DIR, "benchmark data", "1_GTA_reformatted.json")
if not os.path.exists(DATA_FILE):
    DATA_FILE = os.path.join(BASE_DIR, "benchmark data", "benchmark data", "1_GTA_reformatted.json")

WORK_DIR = os.path.join(BASE_DIR, "gta_workspace")
RESULTS_FILE = os.path.join(BASE_DIR, "gta_results.jsonl")
EVAL_OUTPUT_BASE_DIR = os.path.join(BASE_DIR, "gta_eval_outputs")

os.makedirs(EVAL_OUTPUT_BASE_DIR, exist_ok=True)

INVALID_KWS = (
    "[your answer]", "<answer>", "your answer", "tokens used",
    "error", "invalid", "rate limit", "failed", "reconnecting",
)

# 🚨 终极防呆 Prompt：强迫纯文本模型写代码去处理图片
PROMPT_TEMPLATE = """Task: {intent}

Current directory: {task_dir}
Available image files: {files_str}

CRITICAL INSTRUCTIONS:
1. You DO NOT have built-in vision capabilities. You CANNOT see the images directly.
2. If the task involves analyzing an image, YOU MUST write a Python script to extract text or information using `pytesseract` or `PIL`.
3. You can use bash and python tools to execute your scripts.
4. If the task involves calculations (like counting objects or prices), write Python code to compute the result.
5. If you cannot figure out the answer from OCR or code, make your best educated guess immediately.
6. DO NOT describe what you will do. IMMEDIATELY issue the bash tool call.

Your final response MUST strictly end with:
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

def download_file(url, dest_path):
    """使用 requests + 伪装请求头下载图片，防止 403 和超时"""
    if not url: return False
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        }
        print(f"  📥 正在下载: {url[:60]}...")
        response = requests.get(url, headers=headers, timeout=15, stream=True)
        response.raise_for_status()
        with open(dest_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        return True
    except Exception as e:
        print(f"  ❌ 下载失败: {e}")
        return False

# ================= Evaluator 框架层 =================
class BaseGTAEvaluator:
    def __init__(self, data, model_name, output_file):
        self.data = data
        self.model_name = model_name
        self.output_file = output_file
        self.env = os.environ.copy()
        
        # 强行注入防扣费变量和代理（代理需在终端 export）
        self.env["OLLAMA_LOCAL_API_KEY"] = "ollama"
        self.env["OLLAMA_HOST"] = "http://127.0.0.1:11434"
        self.env["DEEPSEEK_API_KEY"] = "" # 强制置空，走本地

    def build_cmd(self, prompt):
        raise NotImplementedError

    def evaluate(self):
        results = []
        for idx, task in enumerate(self.data, 1):
            task_id = str(task.get("id", "0"))
            intent = task.get("intent") or task.get("question") or task.get("task") or "请根据提供的图片回答问题"
            
            print(f"\n[{idx}/{len(self.data)}] 开始测试 GTA Task: {task_id}")
            print(f"  ❓ 意图: {intent[:80]}...")
            
            task_dir = os.path.abspath(os.path.join(WORK_DIR, task_id))
            os.makedirs(task_dir, exist_ok=True)

            # 下载/复制附件
            files = task.get("files", [])
            local_files = []
            for f in files:
                file_url = f.get("url")
                original_path = f.get("path", "file.jpg")
                filename = os.path.basename(original_path)
                dest_path = os.path.join(task_dir, filename)
                
                if file_url and file_url.startswith("http"):
                    if download_file(file_url, dest_path):
                        local_files.append(filename)
                else:
                    local_src = os.path.join(BASE_DIR, "image", filename)
                    if os.path.exists(local_src):
                        shutil.copy(local_src, dest_path)
                        local_files.append(filename)

            files_str = ", ".join(local_files) if local_files else "No local images available."
            
            prompt = PROMPT_TEMPLATE.format(intent=intent, task_dir=task_dir, files_str=files_str)
            cmd = self.build_cmd(prompt)

            print(f"  🚀 调用框架求解...")
            reply = ""
            try:
                # 注入环境变量并执行
                res = subprocess.run(
                    cmd, cwd=task_dir, capture_output=True, 
                    env=self.env, text=True, timeout=600
                )
                reply = res.stdout + "\n" + res.stderr
            except subprocess.TimeoutExpired:
                print(f"  ❌ 超时。")
                reply = "TIMEOUT"
            except Exception as e:
                print(f"  ❌ 异常: {e}")
                reply = str(e)

            print(f"  📄 日志片段:\n{reply[-800:]}\n")
            
            answer = extract_clean_answer(reply)
            if answer:
                print(f"  ✅ 提取成功: {answer}")
            else:
                print(f"  ❌ 未能提取到有效答案。")

            record = {
                "task_id": task_id,
                "intent": intent,
                "ground_truth": task.get("gt_answer", None),
                "predicted_answer": answer,
                "raw_output": reply
            }
            results.append(record)
            
            # 实时写入磁盘，防止中断
            with open(RESULTS_FILE, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        with open(self.output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\n🏆 GTA 评测完成！结果已保存至: {self.output_file}")

class CodexEvaluator(BaseGTAEvaluator):
    def build_cmd(self, prompt):
        return [
            "codex", "exec",
            "--skip-git-repo-check",
            "--dangerously-bypass-approvals-and-sandbox",
            "-m", self.model_name,
            "-c", 'model_reasoning_effort="low"',
            prompt
        ]

class DSHEvaluator(BaseGTAEvaluator):
    def build_cmd(self, prompt):
        return ["npx", "--yes", "@deepseek-ai/dsh", "headless", prompt]

# ================= 主程序 =================
def main():
    parser = argparse.ArgumentParser(description="GTA Ultimate Evaluator")
    parser.add_argument("--agent", type=str, default="codex", choices=["codex", "dsh"], help="Framework: codex or dsh")
    parser.add_argument("--model", type=str, required=True, help="Model name e.g., qwen3.5-9b or gemma")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of tasks")
    parser.add_argument("--start", type=int, default=0, help="Start index (0-based)")
    parser.add_argument("--end", type=int, default=None, help="End index (exclusive)")
    args = parser.parse_args()

    # 清空历史工作区
    if os.path.exists(WORK_DIR):
        shutil.rmtree(WORK_DIR)
    os.makedirs(WORK_DIR)
    
    # 清理历史结果文件（可选）
    if os.path.exists(RESULTS_FILE):
        os.remove(RESULTS_FILE)

    if not os.path.exists(DATA_FILE):
        print(f"❌ 找不到数据文件: {DATA_FILE}")
        return

    with open(DATA_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    if args.end is not None:
        data = data[args.start:args.end]
    elif args.start > 0:
        data = data[args.start:]
        
    if args.limit:
        data = data[:args.limit]

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
