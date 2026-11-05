"""TrustedAgent ToolBench entry point, structured like the supplied AgentHarm template.

Uses the previously tested native runner and official local API classes in this
package. This is an entry point, not a standalone copy of all dependencies.
"""
import argparse
import collections
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys

import run_eval as scorer
from launch import CONDITIONS
from run_matrix import read_events
from summarize import aggregate

ROOT = Path(__file__).resolve().parent
FIELDS = ('task_id', 'dimension', 'benchmark', 'agent', 'model', 'success',
          'score', 'steps', 'tool_calls', 'final_answer', 'error', 'failure_type', 'analysis')


# 1. 原生工具调用与失败分类：仅接受真实事件，不从普通回复猜测调用。
def extract_detailed_tool_calls(events, framework):
    if framework == 'codex':
        return [event['item'] for event in events
                if event.get('type') == 'item.completed'
                and event.get('item', {}).get('type') == 'command_execution']
    return [event for event in events if event.get('type') == 'tool_call']


def classify_failure_type(row):
    if row['success']:
        return None
    # Preserve the original, provisional diagnosis, including environment timeouts.
    return row.get('failure_type') or 'Evaluation Failure'


# 2. 评分：复用官方状态F1与执行证据，不使用AgentHarm的0/0.5/1规则。
def grade_toolbench_score(row, reference, events):
    score = row['score']
    if row['success'] not in (0, 1) or not math.isfinite(score) or not 0 <= score <= 1:
        raise ValueError('Invalid score/success: ' + row['task_id'])
    calls = extract_detailed_tool_calls(events, row['agent'])
    if calls != row['tool_calls'] or row['steps'] != len(calls):
        raise ValueError('Native tool evidence mismatch: ' + row['task_id'])
    if row.get('predicted_state') is not None or row['success']:
        work = Path(row['workspace'])
        receipt = json.loads((work/'execution_receipt.json').read_text(encoding='utf-8'))
        state = scorer.execute(receipt['program'], row['tag'])
        result = scorer.score(state, scorer.labels(reference), row['tag'])
        if state != receipt['state'] or state != row.get('predicted_state'):
            raise ValueError('API replay mismatch: ' + row['task_id'])
        if not math.isclose(result['f1'], score, rel_tol=0, abs_tol=1e-12):
            raise ValueError('F1 mismatch: ' + row['task_id'])
        if bool(result['success']) != bool(row['success']):
            raise ValueError('Success mismatch: ' + row['task_id'])
        if row['success']:
            if not calls or row.get('exit_code') != 0:
                raise ValueError('Missing successful execution: ' + row['task_id'])
            if (work/'solution.py').read_text(encoding='utf-8') != receipt['program']:
                raise ValueError('Solution differs from receipt: ' + row['task_id'])
            if row['agent'] == 'dsh' and (not row.get('turn_completed') or not row.get('final_event')):
                raise ValueError('DSH turn incomplete: ' + row['task_id'])
        return score, '官方状态F1已独立回放核验；' + ('完全成功' if row['success'] else '部分匹配，未完全完成'), calls
    if score != 0 or row['success']:
        raise ValueError('Unscored execution must retain score=0 and success=0')
    return 0, '无有效可评分状态，按本轮端到端约定记0分；保留原始失败原因', calls


# 3. 框架配置与执行：沿用已验证的四组启动方式，真实运行需要Linux。
def get_framework_config(framework, provider, model):
    if provider != 'local':
        raise ValueError('Only the verified local Ollama route is supported; API route is not implemented')
    for name, (agent, configured_model, port) in CONDITIONS.items():
        if agent == framework and configured_model == model:
            return dict(condition=name, port=port)
    raise ValueError('Use qwen35-tools:iq4nl or gemma4:e4b-it-q4_K_M with codex/dsh')


def run_framework(args, model):
    if sys.platform != 'linux':
        raise ValueError('Native runs require Linux; --from-results supports offline export on Windows')
    config = get_framework_config(args.framework, args.provider, model)
    cmd = [sys.executable, str(ROOT/'launch.py'), '--condition', config['condition'],
           '--limit', str(args.limit), '--timeout', str(args.timeout),
           '--dataset', str(args.dataset.resolve()), '--output', str(args.output.resolve()/'native')]
    if args.port is not None:
        cmd += ['--port', str(args.port)]
    # launch.py warms the model before scoring, run_matrix.py enforces each task's timeout.
    subprocess.run(cmd, check=True, env=os.environ.copy())
    return args.output.resolve()/'native'/config['condition']


# 4. 主流程：输出与组内模板一致的13字段JSON，附独立指标汇总及证据索引。
def export_results(folder, dataset, framework, model, output):
    manifest = json.loads((folder/'manifest.json').read_text(encoding='utf-8'))
    if manifest['agent'] != framework or manifest['model'] != model:
        raise ValueError('Requested framework/model differs from source manifest')
    digest = hashlib.sha256((dataset/'inputs/tasks.jsonl').read_bytes()).hexdigest()
    if digest != manifest['source_sha256']:
        raise ValueError('Dataset hash differs from source run')
    refs = {r['sample_id']: r['reference'] for r in json.loads(
        (dataset/'evaluation/references.json').read_text(encoding='utf-8'))}
    rows = [json.loads(line) for line in (folder/'results.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
    ids = [r['task_id'] for r in rows]
    if len(ids) != len(set(ids)) or ids != manifest['selected'][:len(ids)]:
        raise ValueError('Duplicate tasks or different sample order')
    result = aggregate(rows, len(manifest['selected']))
    records = []
    for row in rows:
        if row['agent'] != framework or row['model'] != model:
            raise ValueError('Task framework/model differs from manifest')
        sid = row['task_id']
        reference = refs[sid]
        tag = str(reference['tag'])
        if row['tag'] != tag or tag not in ('2', '11') or (tag == '11' and int(reference['id']) >= 291):
            raise ValueError('Task outside supported ToolBench local API scope')
        event_path = folder/(sid+'.events.jsonl')
        if not event_path.exists():
            if row.get('tool_calls') or row.get('success') or row.get('predicted_state') is not None:
                raise ValueError('Missing event log: ' + sid)
            events = []
        else:
            events = read_events(event_path)
        score, reason, calls = grade_toolbench_score(row, reference, events)
        record = dict(task_id=sid, dimension='工具交互', benchmark='ToolBench',
                      agent=framework, model=model, success=int(row['success']), score=score,
                      steps=len(calls), tool_calls=calls, final_answer=row.get('final_answer', ''),
                      error=row.get('error'), failure_type=classify_failure_type(row),
                      analysis=reason + ('；' + str(row['error']) if row.get('error') else ''))
        if tuple(record) != FIELDS:
            raise ValueError('Unexpected output schema')
        records.append(record)
    result.update(dimension='工具交互', benchmark='ToolBench', agent=framework, model=model,
                  score_metric='mean_state_f1', higher_is_better=True,
                  failures=dict(collections.Counter(r['failure_type'] for r in records if r['failure_type'])),
                  raw_results=str((folder/'results.jsonl').resolve()),
                  note='执行失败无可评分状态时记0；未运行题不补0；steps不混入F1；失败归因为初步判断')
    output.mkdir(parents=True, exist_ok=True)
    prefix = 'toolbench_' + manifest.get('agent', framework) + '_' + datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    target = output/(prefix+'.json')
    target.write_text(json.dumps(records, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    (output/(prefix+'_summary.json')).write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print('RESULT_JSON:', target)
    return target


def main():
    parser = argparse.ArgumentParser(description='TrustedAgent - ToolBench 工具交互评测（组内模板版）')
    parser.add_argument('-f', '--framework', required=True, choices=['codex', 'dsh'])
    parser.add_argument('-p', '--provider', default='local', choices=['local'], help='当前已验证本地Ollama路径')
    parser.add_argument('-m', '--model', help='省略时：新运行使用Qwen；离线导出从manifest读取')
    parser.add_argument('-l', '--limit', type=int, default=10, help='新运行1..220；0表示全部已适配220条')
    parser.add_argument('--timeout', type=int, default=240)
    parser.add_argument('--port', type=int)
    parser.add_argument('--dataset', type=Path, default=ROOT/'data')
    parser.add_argument('--output', type=Path, default=ROOT/'outputs_paper')
    parser.add_argument('--from-results', type=Path, help='已有单组目录，含manifest.json、results.jsonl和证据；不调用模型')
    args = parser.parse_args()
    if args.limit == 0:
        args.limit = 220
    if not 1 <= args.limit <= 220 or args.timeout <= 0:
        parser.error('limit必须为0..220，timeout必须大于0')
    try:
        if args.from_results:
            folder = args.from_results.resolve()
            source = json.loads((folder/'manifest.json').read_text(encoding='utf-8'))
            model = args.model or source['model']
        else:
            model = args.model or 'qwen35-tools:iq4nl'
            folder = run_framework(args, model)
        target = export_results(folder, args.dataset.resolve(), args.framework, model, args.output.resolve())
    except (OSError, ValueError, KeyError, AssertionError, subprocess.CalledProcessError) as exc:
        parser.exit(1, 'EVALUATION_ERROR: ' + str(exc) + '\n')
    return target


if __name__ == '__main__':
    main()
