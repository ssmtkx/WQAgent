# WQ Agent

一个**工作区受限**的工具调用型编码 Agent：你给它一句自然语言任务，它自己规划、写代码、跑命令验证，最后给出结论。命令行入口是 `wq`，所有文件读写与命令执行都被限制在 `--workspace` 指定的目录里。

```powershell
wq "创建 hello.py，打印 Hello, World，然后运行它" --workspace .\demo
```

- **版本**：0.1.0 ｜ **许可**：MIT ｜ **Python**：>= 3.10
- **技术栈**：LangGraph（图编排）· LangChain（模型与工具协议）· Typer + rich（CLI）· Textual（可选 TUI）
- **模型**：任何 OpenAI 兼容端点（OpenAI / DeepSeek / Ollama / vLLM …）

---

## 目录

1. [环境配置](#环境配置)
2. [架构](#架构)
3. [使用](#使用)


---

## 环境配置

### 1. 安装

项目用 `uv` 管理依赖（推荐），也支持纯 `pip + venv`。两种方式二选一，结果等价。

**方式 A：uv（推荐，装包更快、版本锁定在 `uv.lock`）**

```powershell
git clone <repo> ; cd WQAgent
uv sync --extra tui          # 自动建 .venv 并装齐依赖（含 TUI 的 textual）
uv run wq --version
```

**方式 B：pip + venv（不想装 uv 也能跑）**

```powershell
git clone <repo> ; cd WQAgent
python -m venv .venv
.\.venv\Scripts\Activate.ps1          # Linux/macOS: source .venv/bin/activate
pip install -e .                      # 安装 wq 命令（可编辑模式）
pip install -e ".[tui]"               # 需要交互式 TUI 时再加 Textual
wq --version
```

> 无 uv 且想锁死版本：见仓库根目录 `requirements.txt`，可执行 `pip install -r requirements.txt`。

运行时依赖（`pyproject.toml` 声明，本仓库当前实际使用括号内版本）：

| 依赖 | 版本要求 | 用途 |
| --- | --- | --- |
| `langchain-core` | >= 0.3（1.6.6） | 消息与工具协议 |
| `langchain-openai` | >= 0.2（1.6.7） | `ChatOpenAI` |
| `langgraph` | >= 0.2（1.2.12） | 状态图与流式事件 |
| `tavily-python` | >= 0.5（0.8.4） | Web 搜索工具 |
| `typer` | >= 0.12（0.27.2） | `wq` 命令行 |
| `rich` | >= 13.0（15.0.0） | 终端渲染 |
| `python-dotenv` | >= 1.0 | 读取 `.env` |
| `textual`（可选，`[tui]`） | >= 0.80（8.2.8） | 交互式 TUI |

### 2. 模型配置（`.env`）

复制模板并填入自己的值：

```powershell
Copy-Item .env.example .env
```

```ini
# --- 必填 ---
API_KEY=sk-your-api-key-here
MODEL=deepseek-flash
BASE_URL=https://api.deepseek.com        # 留空则用 api.openai.com

# --- 可选 ---
TAVILY_API_KEY=tvly-xxxxxxxx             # 留空则 web_search 返回 {"ok": false}，不会崩
# OPENAI_API_KEY / OPENAI_MODEL / OPENAI_BASE_URL 可作为上面三项的兜底名称
```

### 3. 环境变量总表

| 变量 | 作用 | 默认 |
| --- | --- | --- |
| `WQ_WORKSPACE` | `--workspace` 的默认值 | `./workspace` |
| `WQ_MODEL` / `WQ_BASE_URL` / `WQ_API_KEY` | 覆盖 `.env` 的模型配置 | — |
| `WQ_APPROVAL_MODE` | 危险命令策略：`inline` / `auto` / `deny` | `inline` |
| `WQ_APPROVAL_TIMEOUT` | 审批等待秒数（`0` = 一直等） | `300` |
| `WQ_CHECKPOINT_MODE` | 断点保存：`light` / `strict` / `off` | `light` |
| `WQ_TRACE_MODE` | 执行追踪：`on`(=summary) / `summary` / `full` / `off` | `summary` |
| `WQ_REQUEST_TIMEOUT` | 单次模型响应超时秒数（重试 2 次） | `120` |
| `WQ_CONTEXT_TOKEN_LIMIT` | 上下文 token 预算，超出触发压缩 | `400000` |
| `WQ_ENV_FILE` | 指定 `.env` 路径 | `.env` |
| `WQ_DISABLE_RIPGREP` | 置 `1` 强制用内置扫描器（不调用 `rg`） | 未设置（有 `rg` 就用） |
| `WQ_NO_LOGO` / `WQ_NO_ANIMATION` | TUI 启动 Logo：不打 / 不播动画 | 未设置 |
| `TAVILY_API_KEY` | Web 搜索；为空则禁用该工具 | — |

### 4. 工作区里会生成什么

所有运行痕迹都在工作区内的 `.wqagent/` 下，另有三个可读的记忆文件在工作区根：

```
<workspace>/
├── .wqagent/
│   ├── checkpoints/
│   │   ├── checkpoint.json     每次节点的元数据快照（任务、计划、验收标准、状态）
│   │   ├── RECOVERY.md         人读的恢复指南（含恢复命令）
│   │   ├── state.json          仅 strict：完整图状态（含消息）
│   │   ├── events.jsonl        仅 strict：每个事件一行
│   │   └── git/                内嵌 git 仓库，保存每次快照的工作区状态
│   ├── traces/<trace-id>/      trace-id 形如 wq-20260101-120000-ab12
│   │   ├── trace.json          计数与时间线（节点、工具调用、审批、检查点）
│   │   ├── timeline.md         人读时间线
│   │   └── events.jsonl        仅 full：逐事件记录
│   └── session/
│       ├── session.json        多轮对话记录（turn、角色、route、摘要）
│       └── SESSION_SUMMARY.md  同一段对话的人读版本
├── TODO.md                     当前计划（由 planner 维护）
├── NOTEPAD.md                  压缩时沉淀的长期笔记
└── HISTORY_SUMMARY.md          被压缩掉的历史摘要
```

`checkpoint_mode=off` / `trace_mode=off` 只是不写对应目录，`.wqagent/` 本身仍会用于会话与临时文件。

---

## 架构

### 分层

```
                    ┌──────────────────────────────────────────────┐
   wq 命令 ────────▶│ cli/app.py        typer 入口、参数解析、      │
   （人）            │                  事件渲染、审批交互、Ctrl+C   │
                    └───────────────┬──────────────────────────────┘
                                    │ stream_session_events()  一轮对话
                                    │ stream_agent_events()    一次运行
                    ┌───────────────▼──────────────────────────────┐
                    │ core/agent.py    LangGraph 驱动 + 事件翻译    │
                    │                  （检查点、追踪、收尾事件）    │
                    └───────────────┬──────────────────────────────┘
                                    │
                    ┌───────────────▼──────────────────────────────┐
                    │ graph/           workflow.py  图与路由        │
                    │                  nodes.py     节点实现        │
                    │                  state.py     31 个状态通道   │
                    │                  memory.py    三层记忆 + 压缩 │
                    └───────┬───────────────────────┬──────────────┘
                            │                       │
              ┌─────────────▼─────────┐   ┌─────────▼──────────────┐
              │ agents/               │   │ tools/                 │
              │  code_agent.py  写码  │   │  paths.py  路径围栏     │
              │  search_agent.py 检索 │   │  file_tools / grep /   │
              │                       │   │  bash / web_search     │
              └───────────────────────┘   └────────────────────────┘
                            │
                    ┌───────▼──────────────┐   ┌────────────────────┐
                    │ providers/           │   │ prompts/           │
                    │  ChatOpenAI 构造     │   │  stage1..4, entry  │
                    └──────────────────────┘   └────────────────────┘
```

### 两张图

`core/agent.py` 先跑**入口图**判断意图，再决定是否进入**复杂图**：

```
入口图（build_entry_workflow）
        intent_router ──chat──▶ chat_responder ──▶ END
              │
              └──workflow──▶ （交给复杂图）

复杂图（build_complex_workflow）
        planner ─▶ context_monitor ─▶ verifier ─▶ context_monitor ─▶ final
           ▲                │              │
           │                └─超预算─▶ context_compressor ─┐
           └──────────────────未通过重规划──────────────────┘
```

| 节点 | 职责 | 可用工具 |
| --- | --- | --- |
| `intent_router` | 判断这句输入是闲聊还是任务（输出 JSON，含置信度） | 无（一次便宜调用） |
| `chat_responder` | 直接回答闲聊/提问，不碰工具 | 无 |
| `planner` | 主管：出计划、派活、决定重规划 | `todo_write`、`call_search_agent`、`call_code_agent` |
| `context_monitor` | 测量上下文 token 数，判断是否需要压缩 | 无 |
| `context_compressor` | 把历史压缩成摘要，写入 `HISTORY_SUMMARY.md` 与 `NOTEPAD.md` | 无（一次模型调用） |
| `verifier` | 真跑验收命令并裁决通过与否 | `verification_report`、`bash`、`grep`、`file_read` |
| `final` | 汇总事实，输出最终答案 | 无 |
| `codeAgent`（专家） | 写文件、改代码、跑测试的 ReAct 循环（默认 10 轮） | `file_read`、`file_write`、`file_edit`、`grep`、`bash`、`todo_update` |
| `searchAgent`（专家） | 检索并沉淀结论与来源（默认 4 轮） | `web_search` |

路由表在 `graph/workflow.py`：`ENTRY_ROUTE_TARGETS`（chat/workflow）、`ROUTE_TARGETS`（final/planner）、`CONTEXT_ROUTE_TARGETS`、`COMPRESSOR_ROUTE_TARGETS`。默认重规划上限 3 轮（`--max-attempts`），planner 每次最多 10 个工具轮（`--max-steps`）。

### 状态与记忆

`WQGraphState` 有 31 个通道，关键几组：

- **任务与计划**：`task`、`plan_summary`、`todos`、`acceptance_criteria`、`verification_commands`
- **执行**：`attempts`、`max_attempts`、`messages`（`add_messages` 归约）、`final_answer`、`passed`
- **专家交接**：`agent_handoffs`、`code_agent_summary`、`verifier_summary`、`research_notes`、`sources`
- **上下文管理**：`context_token_limit`、`context_token_count`、`context_should_compress`、`compression_events`、`history_summary`、`context_summary`
- **入口与会话**：`intent_route`、`intent_reason`、`intent_confidence`、`chat_response`、`session_id`、`session_turn`

**三层记忆**（`graph/memory.py`，每个节点调用模型前都会拿到快照）：

1. `RulesLayer`：工作区约束与行为准则；
2. `WorkingMemory`：当前任务状态（计划、todo、验收标准、错误、尝试次数……所有自由文本都有长度上限）；
3. `HistorySummaryStore`：磁盘上的持久层（`HISTORY_SUMMARY.md`、`NOTEPAD.md`、`TODO.md`）加上本次运行的压缩摘要与会话上下文。

超过 `WQ_CONTEXT_TOKEN_LIMIT` 时 `context_compressor` 会把历史压成结构化摘要（目标、已完成、待办、重要文件、工具发现、来源、下一步、风险），同时写进 `HISTORY_SUMMARY.md`（历史）与 `NOTEPAD.md`（长期笔记），并把消息列表裁到摘要 + 最近若干条。

### 安全与可追溯（`core/`）

| 模块 | 作用 |
| --- | --- |
| `paths.py` | 所有工具路径先过 `resolve_in_workspace()`：`../`、绝对路径越界、符号链接逃逸一律拒绝 |
| `approval.py` | 10 条风险命令正则（`pip/npm install`、`curl`、`uvicorn` …）+ 审批请求/决策；`inline` 询问、`auto` 放行并标记、`deny` 拒绝 |
| `checkpoint.py` | 每个节点后写快照；内嵌 git 仓库记录工作区差异；`RECOVERY.md` 给出恢复命令；`--resume` 载入计划继续跑 |
| `trace.py` | 每次运行一个 trace 目录：节点访问、工具调用、失败调用、审批次数、检查点次数、交接次数与时间线；节点级增量落盘，运行中途也能看到文件 |
| `session.py` | 多轮会话：turn 编号、用户/助手记录、`SESSION_SUMMARY.md`，以及注入提示词的上下文块（会话 id、轮次、最近文件、最近 10 轮） |
| `state.py` | `RuntimeState`：工作区、超时、输出预算、命令黑名单、审批/检查点/追踪策略与回调 |

### 模块清单

```
src/wqagent/
├── __init__.py            # 版本号与公开入口
├── __main__.py            # python -m wqagent
├── core/
│   ├── agent.py           # 驱动 LangGraph；stream_agent_events / stream_session_events
│   ├── state.py           # RuntimeState + create_runtime()
│   ├── paths.py           # 工作区路径围栏
│   ├── approval.py        # 危险命令分类与审批
│   ├── checkpoint.py      # 断点保存/恢复（含内嵌 git 快照）
│   ├── trace.py           # 执行追踪
│   └── session.py         # 多轮会话与上下文块
├── graph/
│   ├── workflow.py        # build_workflow / build_complex_workflow / build_entry_workflow
│   ├── nodes.py           # 各节点实现与节点专用工具
│   ├── state.py           # WQGraphState（31 通道）
│   └── memory.py          # 三层记忆、压缩、TODO/NOTEPAD/HISTORY
├── agents/
│   ├── code_agent.py      # codeAgent：写码 ReAct 循环
│   └── search_agent.py    # searchAgent：检索循环
├── tools/
│   ├── registry.py        # build_tools(state) 注册 5 个工具
│   ├── file_tools.py      # file_read / file_write / file_edit
│   ├── grep_tool.py       # grep（优先 rg，可退回内置扫描器）
│   ├── bash_tool.py       # bash（超时、进程树清理、实时输出）
│   ├── web_search_tool.py # Tavily 搜索
│   └── graph_tools.py     # 图内工具（todo / 交接 / 验收报告）
├── providers/
│   ├── __init__.py        # create_model / DEFAULT_MODEL_TIMEOUT_SECONDS
│   └── openai_provider.py # .env 解析与 ChatOpenAI 构造
├── prompts/
│   ├── stage1.py … stage4.py  # 各阶段提示词
│   └── entry.py           # 入口图提示词（意图路由、闲聊回答）
└── cli/
    ├── app.py             # wq 命令
    └── tui/
        ├── app.py         # Textual 界面（Plan 面板 / 事件流 / 状态栏）
        ├── approval.py    # 审批弹窗 + 工具线程 ↔ UI 线程的 gate
        └── logo.py        # 启动 Logo（rich 渐变 + 揭幕动画）
```

---

## 使用

### 快速开始

```powershell
wq "创建 hello.py，打印 Hello, World，然后运行它" -w .\demo
wq "给这个项目加一个 README，说明如何运行" -w .\demo -v      # 展开每一步
wq "继续，给刚才的接口加上参数校验" -w .\demo                  # 同一工作区=同一会话，接着上一轮
```

运行时会实时打印：计划快照、工具调用与结果、专家交接、检查点与追踪落盘、审批请求，最后是结论面板。

### 命令行选项

| 选项 | 说明 | 默认 |
| --- | --- | --- |
| `TASK` | 自然语言任务（`--tui` / `--resume` 时可省略） | — |
| `-w, --workspace` | 工作区目录，不存在会自动创建（`WQ_WORKSPACE`） | `./workspace` |
| `--resume` | 恢复该工作区最近一次断点并继续（会替代 `--workspace`） | — |
| `-m, --model` | 模型名（覆盖 `$MODEL`，`WQ_MODEL`） | `.env` |
| `--base-url` / `--api-key` | 端点与密钥（`WQ_BASE_URL` / `WQ_API_KEY`） | `.env` |
| `--env-file` | 指定 `.env` 文件 | `.env` |
| `--max-steps` | planner 每次尝试的工具轮上限（1–200） | `10` |
| `--max-attempts` | 计划/执行/验证的重试轮数（1–20） | `3` |
| `--approval-mode` | `inline` / `auto` / `deny` | `inline` |
| `--checkpoint-mode` | `light` / `strict` / `off` | `light` |
| `--trace-mode` | `on` / `summary` / `full` / `off` | `on` |
| `-t, --temperature` | 采样温度（0–2） | `0.0` |
| `--timeout` | 单条 shell 命令的秒数上限（1–3600） | `120` |
| `--request-timeout` | 单次模型响应秒数（1–3600） | `120` |
| `-q, --quiet` | 只打印最终答案 | 关 |
| `--no-router` | 跳过意图路由，任何输入都当任务跑工作流 | 关 |
| `--tui` | 打开 Textual 交互界面（无需 TASK） | 关 |
| `-v, --verbose` | 打印每一步的参数与完整输出 | 关 |
| `--show-config` | 打印生效配置（含取值来源 `cli/env/default`）后退出 | — |
| `--list-tools` | 列出注册的工具后退出 | — |
| `-V, --version` | 版本号 | — |

退出码：`0` 成功 ｜ `1` 运行失败 ｜ `2` 用法错误（含缺少任务、依赖缺失）｜ `130` 被 Ctrl+C 中断。

### 危险命令审批

`pip install`、`npm install`、`curl | sh`、`uvicorn` 之类的命令会被拦下来问一句（问题画在 **stderr**，不污染管道输出）：

```
⚠  approval required
tool      : BashTool
risk      : Python package installation
$ pip install fastapi
[y/N]
```

`--approval-mode auto` 直接放行并在事件里标记，`deny` 一律拒绝；等待超过 `WQ_APPROVAL_TIMEOUT`（默认 300 秒，`0` 表示一直等）按拒绝处理。运行中按 **Ctrl+C** 第一次会杀掉正在跑的子进程（约 0.1 秒内响应）并收尾，第二次立即以 130 退出。

### 交互式 TUI

```powershell
pip install -e ".[tui]"
wq --tui -w .\demo                 # 不需要 TASK，进界面后输入
wq --tui -w .\demo --approval-mode auto --trace-mode full
```

```
┌──────────────────────────────────────────────────────────────────────┐
│ 🐺 WQ Agent   ● ready                     session: 8f3c1d2e · turn 2 │
├──────────────────────────────────────────────────────────────────────┤
│ ╭ Plan ────────────────────────────────────────────────────────────╮ │
│ │ ▰▰▰▱▱▱▱▱▱▱  1/3 done  ·  build the FastAPI app                    │ │
│ │ ✅ scaffold   🔄 auth-router   ⬜ tests                            │ │
│ ╰──────────────────────────────────────────────────────────────────╯ │
│ ╭ Activity ────────────────────────────────────────────────────────╮ │
│ │  💬 帮我搭一个 FastAPI 后台管理系统，包含用户认证                   │ │
│ │  🔧 codeAgent · file_write → file_path=app/main.py                │ │
│ │  ✔ codeAgent · file_write: Created app/main.py                    │ │
│ │  🔄 handoff · planner → codeAgent: add the JWT auth router        │ │
│ │  💾 checkpoint · running/verifier (4 files, git 9ad3b835)         │ │
│ ╰──────────────────────────────────────────────────────────────────╯ │
│ ╭──────────────────────────────────────────────────────────────────╮ │
│ │ ›  ask WQ Agent…                                                 │ │
│ ╰──────────────────────────────────────────────────────────────────╯ │
│ Enter send   ·   Ctrl+L clear   ·   Ctrl+C quit                      │
└──────────────────────────────────────────────────────────────────────┘
```

- 每次回车 = 一个会话轮次，后台线程跑 `stream_session_events()`，事件回到 UI 线程更新 Plan 面板 / 事件流 / 状态栏；
- 危险命令弹审批框：`Y`/Enter 批准、`N`/Esc 拒绝，按钮等价；
- 启动前用 rich 画一块 ASCII Art Logo（青→紫渐变 + 揭幕动画，约 0.5 秒）；非终端只画一次，`WQ_NO_LOGO=1` 完全不画，`WQ_NO_ANIMATION=1` 只用静态版；控制台编码装不下方块字或 🐺 时自动换纯 ASCII 版；
- Textual 未安装时 `wq --tui` 会提示 `pip install "wqagent[tui]"` 并以 2 退出，普通 CLI 不受影响。

### Python API

```python
from wqagent.core.agent import stream_agent_events, stream_session_events

# 一次运行：拿到扁平的流式事件
for event in stream_agent_events("创建 hello.py 并运行", workspace="./demo"):
    print(event["type"], event.get("node", ""))

# 一轮对话：内部完成「读会话 → 记录用户轮 → 构建上下文 → 路由 → 记录助手轮 → 落盘」
for event in stream_session_events("继续，加个登录接口", session_workspace="./demo"):
    if event["type"] == "final_answer":
        print(event["route"], event["content"])
    if event["type"] == "session_saved":
        print("turn", event["turn"], "->", event["reply_turn"], event["path"])
```

```python
from wqagent.core.session import load_or_create_session, build_session_context, save_session
from wqagent.core.checkpoint import CheckpointManager
from wqagent.graph import build_complex_workflow, new_graph_state
from wqagent.core.state import create_runtime

runtime = create_runtime("./demo", approval_mode="auto", trace_mode="full")
graph = build_complex_workflow()                     # 或 build_entry_workflow() / build_workflow()
state = new_graph_state("重构 cli/app.py", runtime)
session = load_or_create_session("./demo")
print(build_session_context("./demo", session))
```

`create_runtime(workspace, approval_mode=…, approval_handler=…, checkpoint_mode=…, trace_mode=…, resume_from=…, session=…)` 是推荐入口；`RuntimeState` 直接构造也支持。
