# Megatron-Note

## Topics

### Megatron Lm Architecture

- 2026-06-11 [第三章：Megatron-LM初始化流程与并行通信组创建详解](megatron-lm-architecture/chapter-03-initialization-and-parallel-groups.md) - 系统梳理 Megatron-LM 从 pretrain 启动、torch.distributed 全局初始化、RankGenerator rank 切分，到 TP/PP/CP/DP/EP 通信组、随机种子、JIT 预热和通信缓冲区初始化的完整链路。
  - Reason: 该笔记聚焦 Megatron-LM 初始化主链路与并行通信组构造，和现有 Megatron-LM 架构主题高度匹配，适合作为该主题的初始化与并行组专题章节。
  - Visual: [summary image](megatron-lm-architecture/summary-visual-v3.png)

- 2026-06-08 [第二章：Megatron数据处理与并行分发全流程](megatron-lm-architecture/chapter-02-megatron-data-processing-and-flow.md) - 系统梳理 Megatron-LM GPT 预训练从离线 tokenize、indexed dataset、GPTDataset 索引构建、reset mask 到 DP/TP/PP/CP batch 分发和模型训练入口的数据全链路。
  - Reason: 该笔记聚焦 Megatron-LM 数据处理、GPTDataset 与并行 rank 分发，和现有 Megatron-LM 架构主题高度匹配，适合作为该主题的数据专题章节。
  - Visual: [summary image](megatron-lm-architecture/summary-visual-v2.png)

- 2026-06-07 [第一章：Megatron-LM架构与源码学习路线](megatron-lm-architecture/chapter-01-megatron-lm-architecture-source-learning-route.md) - 系统梳理 Megatron-LM 从训练入口、分布式初始化、数据流到 GPTModel、TransformerLayer、并行 Linear、Attention、Pipeline、DDP 与优化器的源码学习路径。
  - Reason: 该笔记聚焦 Megatron-LM 整体架构、核心模块和源码阅读顺序，适合作为 Megatron-LM 架构主题的第一篇归档笔记。
  - Visual: [summary image](megatron-lm-architecture/summary-visual.png)
