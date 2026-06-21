# Megatron-Note

## Topics

### Context Parallel

- 2026-06-21 [第二章：MindSpeed Ring Attention上下文并行方案详解](features/context_parallel/chapter-02-mindspeed-ring-attention-context-parallel.md) - 系统梳理 MindSpeed Ring Attention CP 的 zigzag 切分、KV 环形轮转、causal case 分发、Case 1 mask 与 pre_tokens/next_tokens 接入、在线 softmax 合并和双层 Ring 通信隐藏机制。
  - Reason: 该笔记聚焦 MindSpeed Ring Attention 风格的 Context Parallel 实现，和现有 Context Parallel 专题中的 Ulysses 方案形成互补，适合作为 Ring/P2P CP 章节归档。
  - Visual: [summary image](summary_images/chapter-02-mindspeed-ring-attention-context-parallel.png)

- 2026-06-14 [第一章：MindSpeed与Megatron Ulysses上下文并行全链路](features/context_parallel/chapter-01-mindspeed-and-megatron-ulysses-context-parallel-a2a.md) - 系统梳理 MindSpeed Ulysses CP 与 Megatron/TransformerEngine A2A CP 从 GPTDataset、TP/CP batch 分发、Q/K/V all-to-all、DualChunkSwap/reorder 到 backward 与调试约束的完整实现差异。
  - Reason: 该笔记聚焦 MindSpeed、Megatron 与 TransformerEngine 的 Ulysses 上下文并行方案，内容主题直接对应 features/context_parallel 目录，适合作为 Context Parallel 特性专题的首篇归档笔记。
  - Visual: [summary image](summary_images/chapter-01-mindspeed-and-megatron-ulysses-context-parallel-a2a.png)

### Megatron Lm Architecture

- 2026-06-11 [第三章：Megatron-LM初始化流程与并行通信组创建详解](megatron-lm-architecture/chapter-03-initialization-and-parallel-groups.md) - 系统梳理 Megatron-LM 从 pretrain 启动、torch.distributed 全局初始化、RankGenerator rank 切分，到 TP/PP/CP/DP/EP 通信组、随机种子、JIT 预热和通信缓冲区初始化的完整链路。
  - Reason: 该笔记聚焦 Megatron-LM 初始化主链路与并行通信组构造，和现有 Megatron-LM 架构主题高度匹配，适合作为该主题的初始化与并行组专题章节。
  - Visual: [summary image](summary_images/chapter-03-initialization-and-parallel-groups.png)

- 2026-06-08 [第二章：Megatron数据处理与并行分发全流程](megatron-lm-architecture/chapter-02-megatron-data-processing-and-flow.md) - 系统梳理 Megatron-LM GPT 预训练从离线 tokenize、indexed dataset、GPTDataset 索引构建、SFT/预训练 packing 差异、reset mask 到 DP/TP/PP/CP batch 分发和模型训练入口的数据全链路。
  - Reason: 该笔记聚焦 Megatron-LM 数据处理、GPTDataset 与并行 rank 分发，和现有 Megatron-LM 架构主题高度匹配，适合作为该主题的数据专题章节。
  - Visual: [summary image](summary_images/chapter-02-megatron-data-processing-and-flow.png)

- 2026-06-07 [第一章：Megatron-LM架构与源码学习路线](megatron-lm-architecture/chapter-01-megatron-lm-architecture-source-learning-route.md) - 系统梳理 Megatron-LM 从训练入口、分布式初始化、数据流到 GPTModel、TransformerLayer、并行 Linear、Attention、Pipeline、DDP 与优化器的源码学习路径。
  - Reason: 该笔记聚焦 Megatron-LM 整体架构、核心模块和源码阅读顺序，适合作为 Megatron-LM 架构主题的第一篇归档笔记。
  - Visual: [summary image](summary_images/chapter-01-megatron-lm-architecture-source-learning-route.png)
