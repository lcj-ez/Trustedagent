# Trustedagent
测试agent框架 + 小模型

我们方法的特点和独特贡献
八维评测框架
TrustedAgent 基于全球 AI 法规、EU AI Act、NIST AI RMF、专家建议以及学术界和产业界推荐，定义了 8 个维度：
1. 感知能力 Perception Ability  
2. 规划能力 Planning Ability  
3. 记忆能力 Memory Ability  
4. 工具使用能力 Tool Use Ability  
5. 专业任务能力 Professional Task Ability  
6. 社会与协作能力 Social & Collaborative Ability  
7. 风险防御能力 Risk Defense Ability  
8. 通用任务能力 General Task Ability  
这些维度覆盖 Agent 从输入感知、规划推理、工具调用到结果交付的完整生命周期

实验介绍和现阶段结论
实验设置
论文选取 2 个 Agent 框架：
- Codex
- Deepseek Hareness

每个框架搭配 2 个前沿 LLM：
- qwen3.5 9B
- Gemma 4 E4B

分别找了8个维度的测试数据，然后进行2 x 2跑测试集来评分
评分细则： 得分 =得分 = 正确的测试用例数 / 总测试用例数
