---
title: "Hermes Agent v0.16 Kanban Swarm 功能深度解析"
date: 2026-08-29
tags: [hermes-agent, kanban, swarm, multi-agent]
draft: false
---

# Hermes Agent v0.16 Kanban Swarm 功能深度解析

假设你有五个 researcher 并行调研同一个主题，谁来决定它们何时可以开始汇总？再假设某个 agent 进程在凌晨三点崩溃，它的任务凭什么能在第二天早上被另一个 worker 捡起来，而不是永远卡死？多 agent 协作真正的难点从来不是"让多个 agent 同时干活"，而是让协调本身**可持久、可审计、可恢复**。Hermes Agent 的 Kanban Swarm 用一张看板回答了这个问题：它不是给子代理加一层进程内调度，而是把"谁该干什么、干到哪一步"写进一个 SQLite 文件，让任何 worker 都能在任意时刻接手。

> 版本说明：本文标题沿用 v0.16 的命名，但正文所有机制名与默认值均以当前源码实现（0.20.x）为准，并经本地源码逐条核实。

## 一、从 v1 swarm 到 Kanban Swarm：演进动机

最早的 `delegate_task` 是进程内 RPC：父 agent fork 出子代理，阻塞等待其 summary 返回。问题很直接——子代理**匿名、无持久记忆、失败即失败**，审计信息随上下文压缩而丢失。当一个 handoff 需要活过一个 API loop、需要被别的 agent 看见时，进程内 RPC 就不够了。

于是有了 Kanban Swarm。它的核心不变量是 **"no in-process subagent swarms"**：协调必须发生在系统可控的层面（持久化看板），而不是子代理的生命周期里——这是从 NanoClaw 的教训里提炼出的设计原则。仓库里的 `hermes kanban swarm` 自述为 "thin swarm topology helpers on top of Kanban"，它**不引入第二个调度器**，只是把一个固定形状的小任务图写进既有的 Kanban 内核。

## 二、核心架构：board / card / task graph

Kanban Swarm 的设计分三个平面。**控制面**（control plane）：用户通过 `hermes kanban …`、`/kanban` 斜杠命令或 dashboard 操作看板，agent 则通过专用 `kanban_*` 工具集操作。**状态面**（state plane）：`~/.hermes/kanban.db` 这个单文件 SQLite（WAL 模式）是唯一事实源。**执行面**（execution plane）：每个 worker 是一个完整 OS 进程（`hermes -p <profile> chat -q …`），拥有独立的 HERMES_HOME、记忆、技能与 workspace。

**为什么三平面分离重要**：worker 之间**永不直接通信**，一切通过 board。这带来两个收益——状态是持久且可审计的（`task_events` 表逐条追加 claimed/completed/crashed 等事件），且 worker 进程崩溃不会丢状态，任务回 `ready` 后另一个进程就能接手。

一个 board 就是一个独立 SQLite DB，加上自己的 `workspaces/`、`logs/` 目录与调度循环。**板间隔离是绝对的**：dispatcher 在 spawn 时注入 `HERMES_KANBAN_BOARD`，worker 只看得到自己板的任务，跨板 link 被禁止。一个 task（card）是 `tasks` 表的一行：title、body、**一个 assignee（profile 名）**、status、priority、workspace_kind/path，以及可选的 idempotency_key（自动化去重）。

依赖关系落在 `task_links` 表里，形成一张**通用 DAG**。child 只有在其**所有 parent 都 `done`** 时才被 promote 到 `ready`——这就是 fan-in 门控；N 个无依赖的兄弟任务则并行执行（fan-out）。"3 researchers → 1 reviewer" 正是 P3 Voting/Quorum 模式的实现。**为什么用图而不是链**：`parents` 是任意组合，无 parent 即并行，这让 orchestrator 能表达任意深度、任意形状的协作，而不是被钉死在固定流水线上。

## 三、调度机制

dispatcher 是一个每 60 秒 tick 一次的"哑"循环，只做四件事：回收 stale claim → 回收崩溃 worker → promote ready → 原子 claim 并 spawn。默认 `dispatch_in_gateway: true`，dispatcher 直接内嵌在 gateway 进程里，无需单独起服务；独立 daemon 已 deprecated。claim 用 SQLite `BEGIN IMMEDIATE` + 行级 CAS 保证并发时最多一个赢家。

**`auto_decompose`（默认 true）** 是核心创新：任务落入 Triage 列时，dispatcher 每 tick 自动调用 `auxiliary.kanban_decomposer` 这个辅助 LLM，读入 profile 名册与任务标题/正文，产出一个 JSON 任务图：

```json
{"fanout": true, "tasks": [
  {"title": "调研竞品实现", "assignee": "researcher", "parents": []},
  {"title": "撰写技术初稿", "assignee": "writer",     "parents": [0]},
  {"title": "语法与事实核查", "assignee": "reviewer", "parents": [1]}
]}
```

`parents` 是同一列表内 0-based 索引，无 parent 即并行。`auto_decompose_per_tick`（默认 3）限制每 tick 分解数量，防止批量灌入时突发消耗辅助 LLM。

**failure_limit（默认 2）是熔断器**：任务连续 N 次非成功尝试（如 spawn 失败）后，dispatcher 自动 block 并记 `gave_up` 事件——防止对不存在的 profile、挂不上的 workspace 无限抖动。`rate_limited`（provider 限流）不算失败、不 tick 计数，任务回 `ready` 轻量重试。protocol violation（进程干净退出但任务仍 running）有单独预算，上限 3 次。

block 有四种 kind：`dependency`（只等另一任务，路由回 `todo`，自动恢复）、`needs_input` / `capability` / `transient`（需要人或能力，表面化到 `blocked`）。unblock 恢复到安全来源相位，**绝不直接送 triage**。更关键的是 `BLOCK_RECURRENCE_LIMIT = 2`：任务 block→unblock→同因再 block 达两次后，会触发 `block_loop_detected` 把任务路由到 triage 交给人类——这是**确定性 DB 守卫，不是 LLM 判断**，否则 cron 会一直 unblock 它形成死循环。

长任务用 `kanban_heartbeat` 保活：若任务可能跑超一小时，至少每小时打一次心跳。满足"已运行超时 + 最后一小时无心跳"双条件，dispatcher 会 SIGTERM 掉 worker、任务回 `ready`（stale 不算 worker 过错，不 tick 失败计数）。一个写作时须留意的**文档-实现偏差**：`dispatch_stale_timeout_seconds` 文档称默认 4 小时，但当前实现默认 `0`（stale 检测默认关闭），需显式配置才启用。

## 四、角色拓扑与生命周期

Kanban **没有 role 字段**——"角色"是用户空间约定：profile（工具集限制）+ skill（行为引导）。dispatcher 不关心 profile 是否叫 orchestrator。博客流水线就是这套拓扑的实例：

| 角色 | 特征 | 职责 |
|---|---|---|
| orchestrator | 禁用 terminal/file/web/code | 拆解目标、link、指派，然后抽身；决策在 fan-out 前定死并写进每张 child body |
| researcher / writer | 正常工具集 + kanban 工具 | 执行单卡，`kanban_show()` → 干活 → `kanban_complete/block` |
| reviewer | 加载 sdlc-review skill | 同卡评审：complete / request_changes / block |
| publisher | 预创建 release child | 实现卡 complete 后由依赖门控放行 |

orchestrator 被物理上剥夺执行工具，这**强制了"只编排不干活"的纪律**——因为 worker 看不到 sibling 卡，所有命名/schema/文件格式等决策必须由 orchestrator 提前定死并写进每张 child body。

状态机有 **8 态**：`triage | todo | ready | running | blocked | review | done | archived`（其中 triage、review 是 v1 spec 6 态之后新增的）。review lane 是独立于 block 的流程：`kanban_request_review` 把卡送进 `review`，reviewer 批准走 `kanban_complete`、退回走 `kanban_request_changes`（不占 block-loop 计数），只有真外部阻塞才用 `kanban_block`。

## 五、Agent 间通信协议

worker 之间不直接通信，一切通过三类载体：

**结构化 handoff**：`kanban_complete(summary=…, metadata=…)` 里，summary 是人类可读收尾，metadata 是机器可读交接。child 的 worker context 里含 "Parent task results" 段，**verbatim 携带每个 parent 完成时的 summary + metadata**——这就是本文写作时读到的调研笔记来源。红线是 summary/metadata 里不放 secrets/tokens/raw PII，因为 run 行是持久的。

**comments 线程**：agent 与人都在任务线程追加；worker（重）生成时读**完整**评论线程作为上下文。

**artifacts / attachments**：上限 25MB，`kanban_attach`（base64 内联）或 `kanban_attach_url`（服务端抓取）上传。scratch workspace **任务完成即删除**，所以 `kanban_complete(artifacts=[绝对路径])` 是唯一可靠的跨任务文件传递通道——缺失的声明 artifact 会让任务保持 in-flight。

## 六、v1 swarm 与通用图调度对比

| 维度 | Swarm v1（固定拓扑 helper） | 通用看板图调度 |
|---|---|---|
| 编排方式 | 预定义三阶段：workers 并行 → verifier 门控 → synthesizer，一条 CLI 原子建图 | 任意 DAG：auto_decompose 一键产出，或 orchestrator 用 `kanban_create` + `parents` 手搓 |
| 依赖表达 | 固定链，worker 必须等齐 | `parents` 任意组合，fan-in/fan-out 无形状限制 |
| 失败处理 | 完全复用内核：claim TTL / crash 检测 / stale / 熔断 / protocol violation | 同一套内核，无区别 |
| 通信 | 显式 blackboard（root 卡 `[swarm:blackboard]` JSON 评论） | 结构化 handoff + 评论线程，parent link 即上下文通道 |
| 扩展性 | 形状固定，加阶段要手动改图 | 任意层级、角色数、依赖深度 |

一条命令即可起一个固定研发生命周期：

```bash
hermes kanban swarm "实现并评审用户登录模块" \
  --workers backend:后端实现,security:安全审查 \
  --verifier reviewer --synthesizer writer
```

固定拓扑适合"并行产出 → 质检 → 汇总"的标准流程；而通用图调度能表达一切 DAG——从 research triage、工程管线到 fleet farming。

## 七、适用场景与最佳实践

官方给出的五类场景：Research triage（并行 researcher + analyst + writer，人在环中）、Scheduled ops（每日简报等循环任务）、Digital twins（持久命名助手，跨会话累积记忆）、Engineering pipelines（decompose → 并行 worktree → review → PR）、Fleet work（一个 specialist 管 N 个对象）。

几条实战经验，每条背后都是一个踩过的坑：

- **assignee 必须真实存在**：dispatcher 对未知 assignee **静默失败**，卡会永远停在 ready。fan-out 前先 `hermes profile list` 对齐真实 profile。
- **成本分层**：frontier 模型跑 orchestrator/dispatcher，便宜模型跑 worker（token 大头），质量敏感卡用 per-task `--model` 钉回强模型。
- **handoff 证据四问**：改了什么 / 怎么验证 / 失败谁能解 / 故意留了什么风险，落到 metadata 约定键。
- **后续工作开新卡，不 reopen done 卡**：done 卡是不可变历史，上下文通过 parent link 前向流动。
- **hotspot 标注**：某文件被 ≥2 张卡 `hotspot:` 点名，先拆文件再排队更多活。

## 八、结语

Kanban Swarm 的价值不在"能并行跑几个 agent"，而在它把**协调本身变成了持久状态**：依赖是数据库里的 link，交接是任务行上的 handoff，失败是熔断计数与 block kind。正因如此，一个凌晨崩溃的 worker 才不会拖垮整条流水线——任务安静地回到 `ready`，等下一个 tick、下一个进程来接手。这，才是多 agent 系统真正稀缺的可靠性。

---

*延伸阅读：官方文档 https://hermes-agent.nousresearch.com/docs（user-guide/features/kanban 与 kanban-worker-lanes）。*
