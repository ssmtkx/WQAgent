# WQ Agent

一个基于 LangChain + Typer 的**工作区受限（workspace-scoped）**工具调用型编码 Agent。
命令行入口是 `wq`，所有文件操作都被限制在 `--workspace` 指定的目录内。

```powershell
wq "创建 hello.py，打印 Hello, World，然后运行它" --workspace ./demo
```

---

## 特性

- **Typer CLI**：`wq "<task>" --workspace <path>`，workspace 默认 `./workspace` 且自动创建。
- **5 个内置工具**：`file_read` / `file_write` / `file_edit` / `grep` / `bash`，全部通过 `build_tools(state)` 注册为 LangChain `StructuredTool`，可直接 `model.bind_tools(tools)`。
- **LangGraph 主管 + 专家**：`planner`（主管，只用 `todo_write` / `call_search_agent` / `call_code_agent` 三个工具）按需把研究交给 `searchAgent`、把实现交给 `codeAgent`，再由 `verifier` 真跑验收命令裁决，最后由 `final` 出结论。短流程是 `planner → verifier ⇄ planner → final`；`build_complex_workflow()` 在此之上接入 `context_monitor` / `context_compressor`（上下文测量与压缩）。
- **硬性路径围栏**：任何工具路径都先经过 `core/paths.py` 的 `resolve_in_workspace()`，`../`、绝对路径越界、符号链接逃逸都会被拒绝。
- **OpenAI 兼容**：`providers/openai_provider.py` 用 `.env` 里的 `API_KEY` / `MODEL` / `BASE_URL` 构造 `ChatOpenAI`，因此 OpenAI、Ollama、vLLM、DeepSeek 等都能用。
- **输出预算与超时**：工具输出按字符数截断；`bash` 有超时控制，超时会杀掉整个进程树。
- **无网络可测**：`tests/` 用脚本化的假模型跑完整 Agent 循环，不需要 API Key。

## 环境要求

Python 3.10+（依赖 `langchain-core`、`langchain-openai`、`langgraph`、`typer`、`python-dotenv`、`rich`）。

已通过的验证环境（`pytest`：**726 passed, 1 skipped**，跳过项是 Windows 下需要管理员权限的符号链接用例）：

| 组件 | 版本 |
| --- | --- |
| Python | 3.12.14 (Windows) |
| langchain-core | 1.6.6 |
| langchain-openai | 1.6.7 |
| langgraph | 1.2.12 |
| tavily-python | 0.8.4 |
| openai | 3.24.0 |
| typer | 0.27.2 |
| rich | 15.0.0 |
| pytest | 9.1.1 |

## 安装

```powershell
# Windows (PowerShell)
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
.\.venv\Scripts\wq.exe --version
```

```bash
# macOS / Linux
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
wq --version
```

## 配置

在项目根目录创建 `.env`（或用 `--env-file` 指定）并填写：

```ini
API_KEY=sk-xxxxxxxx
MODEL=gpt-4o-mini
BASE_URL=https://api.openai.com/v1
```

三种配置方式的优先级：**命令行参数 > 环境变量 > `.env`**。
`.env` 查找顺序：`--env-file` → `$WQ_ENV_FILE` → 从当前目录向上找到的第一个 `.env` → 项目根目录的 `.env`。

本地模型示例：

```ini
# Ollama
BASE_URL=http://localhost:11434/v1
MODEL=qwen2.5-coder:7b
API_KEY=not-needed
```

搜索专家（[`WebSearchTool`](src/wqagent/tools/web_search_tool.py)）可选配置：

```ini
# Tavily API key（https://tavily.com）。没有它时工具返回
# {"ok": false, "error": "missing TAVILY_API_KEY"}，搜索 Agent 报 ok=false 而不是崩。
TAVILY_API_KEY=tvly-xxxxxxxx
```

## 使用

```powershell
# 基本用法
wq "在 src/ 下创建一个 FastAPI 的最小示例并写一个冒烟测试"

# 在当前目录工作
wq "找出所有 TODO 注释并整理成 TODO.md" -w .

# 长任务：放宽步数与单条命令超时
wq "重构 utils.py 并跑通 pytest" -w . --max-steps 60 --timeout 600

# 只看最终答案（适合脚本调用）
wq "统计仓库里有多少个 Python 文件" -w . --quiet

# 排查配置 / 查看工具
wq --show-config
wq --list-tools --verbose

# 等价入口
python -m wqagent "同一个任务"
```

### 常用参数

| 参数 | 说明 | 默认值 |
| --- | --- | --- |
| `TASK` | 任务描述（自然语言）；`--resume` 时可省略（用检查点里的任务） | 必填 |
| `-w, --workspace` | 工作区目录，不存在会自动创建 | `./workspace`（`$WQ_WORKSPACE`） |
| `--resume PATH` | 从该工作区的检查点恢复并继续（**替代 `-w`**，同时传入会提示；命令见 `RECOVERY.md`；跳过意图路由） | — |
| `--no-router` | 跳过意图路由，任何输入都当任务跑主工作流 | 关（默认先做意图路由） |
| `--tui` | 打开 Textual 交互界面（Plan 面板 / 事件流 / 审批弹窗；**无需 TASK**） | 关（需 `pip install "wqagent[tui]"`） |
| `WQ_CONTEXT_TOKEN_LIMIT`（环境变量） | 上下文 token 预算，超出即触发压缩 | `400000` |
| `--approval-mode` | 风险命令策略：`inline`（先问）/ `auto`（放行并标记）/ `deny`（拒绝） | `$WQ_APPROVAL_MODE` 或 `inline` |
| `WQ_APPROVAL_TIMEOUT`（环境变量） | `inline` 问话等待秒数，超时按"否"；`0` 表示一直等 | `300` |
| `--checkpoint-mode` | 断点保存：`light`（每个节点存元数据+恢复指南+git 快照）/ `strict`（另存全量 state 与每个事件）/ `off` | `$WQ_CHECKPOINT_MODE` 或 `light` |
| `--trace-mode` | 执行追踪：`on`（=`summary`，统计+时间线）/ `full`（另存 `events.jsonl`）/ `off` | `$WQ_TRACE_MODE` 或 `on` |
| `-m, --model` | 模型名，覆盖 `$MODEL` | `.env` 的 `MODEL` |
| `--base-url` | OpenAI 兼容端点，覆盖 `$BASE_URL` | `.env` 的 `BASE_URL` |
| `--api-key` | API Key，覆盖 `$API_KEY` | `.env` 的 `API_KEY` |
| `--env-file` | 指定 `.env` 文件 | 自动查找 |
| `--max-attempts` | plan/delegate/verify 轮数上限 | `3` |
| `--max-steps` | **每次 attempt 内** planner 的工具轮数（含专家交接） | `10` |
| `-t, --temperature` | 采样温度 | `0.0` |
| `--timeout` | 单条 shell 命令的默认超时（秒） | `120` |
| `--request-timeout` | 等**一次模型响应**的上限（秒；失败重试 2 次） | `$WQ_REQUEST_TIMEOUT` 或 `120` |
| `-v, --verbose` | 打印每一步、工具参数与完整输出 | 关 |
| `-q, --quiet` | 只输出最终答案 | 关 |
| `--list-tools` | 列出已注册工具后退出 | — |
| `--show-config` | 打印解析后的配置后退出 | — |
| `-V, --version` | 版本号 | — |

退出码：`0` 成功、`1` 运行失败（含触发步数上限）、`2` 用法错误、`130` 用户中断。

### Ctrl+C 中断（为什么以前像"没反应"，现在怎么做的）

langgraph 把每个同步节点丢进**工作线程**执行，主线程用 `concurrent.futures.wait` 等它（`langgraph/pregel/_executor.py`：`if pending := {...}: concurrent.futures.wait(pending)`）。在 Windows 上这个锁等待**不会执行 Python 层的信号处理器**，于是 Ctrl+C 只能"排队"到当前节点返回为止——实测（12 秒的 shell 命令，第 3 秒发 SIGINT）：

```text
[t+ 0.06] register pid=34436            # 命令已登记在跑的清单里
[t+ 3.02s] sending SIGINT               # 按下 Ctrl+C
[t+12.53] UNREGISTER pid=34436 rc=0     # 命令自己跑完了，处理器这时才运行
[t+12.55s] KeyboardInterrupt reached the caller    # 反应延迟 9.55s，且子进程活到了最后
```

修好之后同一实验：

```text
[t+ 3.02s] sending SIGINT (running shell commands: 1)
interrupt received - stopped 1 running command(s): finishing the current step, ...
[t+ 3.11] UNREGISTER pid=13980 rc=1     # 命令被杀掉
[t+ 3.11s] KeyboardInterrupt reached the caller    # 反应延迟 0.11s，子进程没活下来
```

具体做法（`cli/app.py::_InterruptGuard` + `tools/bash_tool.py`）：

- bash 工具把每个在跑的子进程登记到一张进程级清单里，跑完就注销；`kill_running_commands()` 可以随时杀掉它们（`taskkill /F /T` / `killpg`）。
- CLI 运行期间安装 `_InterruptGuard`：它用 `signal.set_wakeup_fd`（C 层处理器**立刻**写一个字节）+ 一个守护线程 `select` 这个 fd，所以**不等主线程**就能反应——打印提示、杀掉在跑的命令，节点于是马上返回。
- 第一次 Ctrl+C：提示 + 杀命令，随后主线程的 `KeyboardInterrupt` 照常走完（检查点与 trace 记为 `interrupted`，退出码 130）。第二次 Ctrl+C：直接 `os._exit(130)`，不再等线程或子进程（长模型调用杀不掉，这是逃生口）。
- 局限（如实说明）：正在进行的**模型 HTTP 调用**无法中断，第一次 Ctrl+C 后仍需等它返回；那时按第二次即可立刻退出。

三个模式参数用 `Annotated[Literal[...], typer.Option(..., envvar=...)]` 声明（`ApprovalMode` / `CheckpointMode` / `TraceMode`）：`--help` 里直接列出可选值（`<inline|auto|deny>` 等），**非法值会被 click 当场拒绝**（退出码 2，例如 `--approval-mode junk` 报 `'junk' is not one of …`），环境变量写错也同样拒绝（报错里会带上 `env var: 'WQ_APPROVAL_MODE'`）；大小写不敏感（`--trace-mode ON` = `--approval-mode DENY` 都接受，随后由 `normalize_*()` 归一化）。库层面仍保留「非法值回落默认」的宽容语义（`normalize_approval_mode()` 等）。`--show-config` 的 `approval/checkpoints/traces/workspace` 行会附上取值来源 `(cli|env|default)`（由 `ctx.get_parameter_source()` 得出），例如：

```text
workspace   : D:\ai-python-prompt\WQAgent (exists) (cli)
approval    : inline (default)
traces      : full (env)
```

## 项目结构

```
src/wqagent/
├── __init__.py            # 版本号与公开接口
├── __main__.py            # python -m wqagent
├── core/
│   ├── __init__.py
│   ├── state.py           # RuntimeState 数据类（workspace、超时、输出预算、命令黑名单、审批/检查点/追踪策略）与 create_runtime()
│   ├── approval.py        # 风险命令分类（RISK_PATTERNS）与审批请求/决策
│   ├── checkpoint.py      # 断点保存/恢复：checkpoint.json、RECOVERY.md、git 快照
│   ├── session.py         # 会话：session.json + SESSION_SUMMARY.md + 上下文块
│   ├── trace.py           # 执行追踪：trace.json、events.jsonl、timeline.md
│   ├── paths.py           # 工作区路径工具（resolve_in_workspace 等）
│   └── agent.py           # 入口：驱动 LangGraph 工作流并翻译事件；stream_session_events() 负责一轮对话
├── prompts/
│   ├── __init__.py
│   ├── entry.py           # INTENT_ROUTER_PROMPT + CHAT_RESPONDER_PROMPT（入口图）
│   ├── stage1.py          # ACTOR_PROMPT（早期 actor 节点的系统提示词，保留）
│   ├── stage2.py          # FINAL_PROMPT（并转出 PLANNER_PROMPT / VERIFIER_PROMPT / ACTOR_PROMPT）
│   ├── stage3.py          # PLANNER_PROMPT（主管）+ VERIFIER_PROMPT + SEARCH_AGENT_PROMPT / CODE_AGENT_PROMPT
│   └── stage4.py          # CONTEXT_COMPRESSION_PROMPT（上下文压缩）
├── agents/
│   ├── __init__.py
│   ├── search_agent.py    # run_search_agent()：只带 web_search 的 ReAct 研究专家
│   └── code_agent.py      # run_code_agent()：工作区工具 + todo_update 的实现专家
├── graph/
│   ├── __init__.py
│   ├── state.py           # LangGraph 共享状态：WQGraphState / TodoItem / SourceItem / AgentHandoff
│   ├── memory.py          # 三层 Memory：RULES_LAYER / build_layered_memory / NOTEPAD.md 读取
│   ├── nodes.py           # planner / context_monitor / context_compressor / verifier / final + 路由
│   └── workflow.py        # build_entry_workflow()（意图路由）、build_workflow()（短流程）与 build_complex_workflow()（带上下文管理）
├── tools/
│   ├── __init__.py
│   ├── registry.py        # build_tools(state) / build_read_only_tools(state)
│   ├── file_tools.py      # FileReadTool / FileWriteTool / FileEditTool
│   ├── grep_tool.py       # GrepTool（优先 ripgrep，缺失时内置扫描器）
│   ├── bash_tool.py       # BashTool（超时 + 进程树清理 + 危险命令过滤 + 结构化 ShellOutcome）
│   ├── web_search_tool.py # WebSearchTool（Tavily；缺 key/缺包/网络错误都返回 ok=false）
│   └── graph_tools.py     # todo_write / todo_update / verification_report（结构化输出工具）
├── providers/
│   ├── __init__.py
│   └── openai_provider.py # create_model() / load_env() / describe_config()
└── cli/
    ├── __init__.py
    ├── app.py             # typer 入口，wq 命令（消费事件流并实时渲染；--tui 交给 cli/tui）
    └── tui/
        ├── __init__.py    # run_tui()：可选依赖的懒加载与友好报错
        ├── logo.py        # 启动 Logo：ASCII Art + rich 渐变 + 揭幕动画（自动降级）
        ├── app.py         # WQAgentTuiApp：Plan 面板 / 事件流 / 状态栏 / 输入框
        └── approval.py    # ApprovalModal（弹窗）+ ApprovalGate（工具线程 ↔ UI 线程握手）

tests/                     # pytest：路径围栏、5 个工具、ReAct 事件流、图状态、两个专家、CLI 端到端（含真实 HTTP）、命令审批、断点恢复、执行追踪、会话、TUI
```

### 多轮入口：`stream_session_events()`

`stream_agent_events()` 是"一次运行"，`stream_session_events()` 是"**一轮对话**"：把会话读写、意图路由和收尾打包在一起（[core/agent.py](src/wqagent/core/agent.py)）。

```python
from wqagent.core.agent import stream_session_events

for event in stream_session_events("继续加个登录接口", session_workspace="./demo"):
    print(event["type"], event.get("node", ""), event.get("route", ""))
```

它依次做六件事（对应实现里的六步）：

1. `load_or_create_session(session_workspace)`——读 `.wqagent/session/session.json`，没有就新建；
2. `append_user_turn(session, task)` 记录用户这一轮，**并立刻落盘**（这样第 4 步里入口节点与 Memory 从磁盘读到的就是本轮编号）；
3. `build_session_context()` 构建上下文块（会话 id、轮次、最近 30 个文件、最近 10 轮）；
4. 跑入口图：`chat` → `chat_responder` 直接答（不碰工具）；`workflow` → `build_complex_workflow()` 全流程；
5. `append_assistant_turn(route=最终事件的 route, summary=回答首句)` + `save_session()`（写 `session.json` 与 `SESSION_SUMMARY.md`）；
6. 收尾发一条 `session_saved` 事件：

```python
{"type": "session_saved", "session_id": "…", "turn": 1, "reply_turn": 2,
 "turn_index": 2, "route": "chat", "path": ".../session.json",
 "summary_path": ".../SESSION_SUMMARY.md", "turns": 2, "saved": True}
```

**上下文对两边都生效**：入口节点从磁盘读（所以"继续"能被理解成对上一轮工作的跟进）；工作流的每个节点则通过 **Memory 第三层**拿到它——`build_layered_memory()` 里新增了 `read_session()`/`session` 字段（与 `notepad`、`history_summary` 并列，上限 `SESSION_CHARS = 1600`），所以 planner / verifier / codeAgent 都知道"之前聊过什么"。

关键参数：`session_workspace=`（必填，也是运行的工作区；`workspace=` 是别名）、`session=`（复用内存里的会话，省一次读盘）、`save=False`（只更新内存、不写盘）、`entry_router=False`（跳过路由、一律走工作流）、以及 `model` / `max_attempts` / `max_loops` / `approval_mode` / `approval_handler` / `checkpoint_mode` / `trace_mode` / `graph` / `on_event`。

实测（同一工作区两轮）：

```
turn 1: chat     → session_saved{turn: 1, reply_turn: 2, turns: 2, route: chat}
turn 2: workflow → final_answer{route: workflow, passed: True} / nodes: intent_router → planner → context_monitor → verifier → context_monitor → final
         session.json: [(1,'user'),(2,'assistant','chat'),(3,'user'),(4,'assistant','workflow')]
```

运行被 Ctrl-C 或异常打断时：用户那一轮已经落盘，不会记助手轮（对话不会被悄悄重置）。

## TUI（`wq --tui`，Textual）

### 启动 Logo（`cli/tui/logo.py`）

启动时先由 **rich** 画一块 ASCII Art 横幅（带颜色；终端上还有"逐行揭幕 + 色块扫过"的动画，约 0.5 秒）：

```
  🐺 WQ Agent
  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   Stage 6 · MultiAgent + Context/Harness
  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  ██╗    ██╗ ██████╗
  ██║    ██║██╔═══██╗
  ██║ █╗ ██║██║   ██║
  ██║███╗██║██║▄▄ ██║
  ╚███╔███╔╝╚██████╔╝
   ╚══╝╚══╝  ╚══▀▀═╝
```

字标用方块字符（`██╗` 一类），从左到右走一条 cyan→magenta 渐变；揭幕期间未出现的列是暗色、正被扫过的那一列高亮。控制方式（环境变量，不用改代码）：

| 变量 | 作用 |
| --- | --- |
| `WQ_NO_LOGO=1` | 完全不打印 logo |
| `WQ_NO_ANIMATION=1` | 只打印静态横幅（不播动画） |

其余自动降级：**非终端**（重定向到文件/管道）只画一次、不播动画；控制台编码装不下 `━`/`🐺`（例如 GBK）时自动换成**纯 ASCII 版**（`WQ Agent` + `-` 规则行 + `##` 字标）；动画预算超过 1.5 秒自动转静态；任何异常都被吞掉（logo 永远不会拖垮启动）。

```python
from rich.console import Console
from wqagent.cli.tui.logo import print_logo, logo_lines, logo_text, frames_of

print_logo(Console())                     # 终端上带动画，否则静态
print_logo(Console(), animate=False)      # 强制静态
print_logo(Console(), ascii_only=True)    # 强制纯 ASCII
logo_lines()                              # 纯文本行（无颜色）
len(frames_of(Console()))                 # 动画帧数（可预览/测试）
```

### 启动 TUI

三步启动（`--tui` **不需要 TASK**，进界面后再输入）：

```powershell
# 1) 装可选依赖（本仓库的 .venv 里已经装好 textual 8.2.8）
pip install "wqagent[tui]"           # 或 pip install textual

# 2) 启动（-w 指定工作区；不写就用 ./workspace）
wq --tui -w .\demo

# 等价入口
python -m wqagent --tui
```

进界面后：底部输入框打字 → **Enter** 提交，每次提交就是一个 session turn；`Ctrl+L` 清屏，`Ctrl+C` 退出。其余参数照常在命令行给（`--approval-mode auto`、`--trace-mode full`、`--no-router`、`--max-attempts` 等都会带进 TUI）：

```powershell
wq --tui -w .\demo --approval-mode auto --trace-mode full
```

常见情况：

- **必须在真实终端里跑**（Windows Terminal / PowerShell 窗口 / VS Code 终端都行）：Textual 需要 TTY，重定向到管道或文件时会报终端相关错误 ✗；
- 没装 textual 时不会崩，会打印 `the TUI needs the optional 'textual' dependency: pip install "wqagent[tui]"` 并退出码 2；
- 会话仍然落在工作区里（`.wqagent/session/`），所以 `--tui` 与普通 `wq "<task>"` 交替使用是**同一条会话**；
- 危险命令会弹出审批弹窗，`Y`/Enter 批准、`N`/Esc 拒绝，超时（`WQ_APPROVAL_TIMEOUT`，默认 300 秒）按拒绝处理。

```
┌──────────────────────────────────────────────────────────────────────┐
│ 🐺 WQ Agent   ● ready                     session: 8f3c1d2e · turn 2 │  brand bar
├──────────────────────────────────────────────────────────────────────┤
│ ╭ Plan ────────────────────────────────────────────────────────────╮ │
│ │ ▰▰▰▱▱▱▱▱▱▱  1/3 done  ·  build the FastAPI app                    │ │
│ │ ✅ scaffold   🔄 auth-router   ⬜ tests                            │ │
│ ╰──────────────────────────────────────────────────────────────────╯ │
│ ╭ Activity ────────────────────────────────────────────────────────╮ │
│ │  💬 帮我搭一个 FastAPI 后台管理系统，包含用户认证                   │ │
│ │  🧭 route · workflow [0.92] needs files                           │ │
│ │  🔧 codeAgent · file_write → file_path=app/main.py                │ │
│ │  ✔ codeAgent · file_write: Created app/main.py                    │ │
│ │  🔄 handoff · planner → codeAgent: add the JWT auth router        │ │
│ │  ✖ verifier · bash: exit 1: 2 failed                              │ │
│ │  💾 checkpoint · running/verifier (4 files, git 9ad3b835)         │ │
│ │  📝 Done — FastAPI app with JWT auth scaffolded in app/main.py.   │ │
│ ╰──────────────────────────────────────────────────────────────────╯ │
│ ╭──────────────────────────────────────────────────────────────────╮ │
│ │ ›  ask WQ Agent…                                                 │ │  input bar
│ ╰──────────────────────────────────────────────────────────────────╯ │
│ Enter send   ·   Ctrl+L clear   ·   Ctrl+C quit                      │  hints
└──────────────────────────────────────────────────────────────────────┘
```

样式要点：深蓝灰底 + 青/紫强调色（与启动 Logo 同一套配色）；品牌栏左侧是 `🐺 WQ Agent` 加状态点（`● ready` / 转圈 `⠋ working` / `● needs approval`），右侧右对齐显示 `session · turn`；**Plan 卡片**用圆角边框 + border title，内含 `▰▰▱▱` 进度条、`done/total`、计划摘要与 `✅/🔄/⛔/⬜` 芯片，没有计划时整卡隐藏（不占空间）；**Activity 卡片**按事件类型着色（用户白色加粗、工具调用青、成功绿、失败红、交接紫、检查点暗灰、最终答复绿底加粗），每条带 `node · tool` 归因，轮次之间画一条暗色分隔线，超过 400 行自动丢弃最旧的；输入区是圆角条 + 青色 `›` 提示符（聚焦时边框变青）；底部一行快捷键提示（不再占用 Footer 空间）。

**机制**（[cli/tui/app.py](src/wqagent/cli/tui/app.py)）：

1. 每次回车就是**一个 session turn**：起一条后台线程跑 `stream_session_events()`（`session=` 在内存里连续传递，所以多轮对话与 `wq` 命令行走的是同一条会话）；
2. 线程里每个事件都 `post_message(AgentEventMessage(...))` 送到 UI 线程（Textual 的 `Message` 支持跨线程投递）；
3. UI 按类型分发：`todo_update` / `node_end`(带 todos) / `plan_snapshot` → **Plan 面板**；`tool_call` / `tool_result` / `handoff` / `checkpoint_saved` / `context_compressed` / `search_results` / `intent` → **事件流**；`final_answer` → 结果行 + 状态栏 `✔ passed`；`session_saved` → 更新 session id 与轮次；
4. 上一轮没跑完时再提交会被挡下（提示 `still working`），`Ctrl+L` 清屏，`Ctrl+C` 退出。

**审批弹窗**（[cli/tui/approval.py](src/wqagent/cli/tui/approval.py)）：

```
⚠  approval required
tool      : BashTool
risk      : Python package installation
approval  : approval-1a2b3c4d
workspace : D:\demo
$ pip install fastapi
[ Approve (Y) ]  [ Deny (N) ]
Y / Enter approves · N / Esc denies · no answer within 300s = denied
```

线程同步靠 `ApprovalGate`：BashTool 在自己的线程里调用 handler → 建 gate → `post_message(ApprovalRequestedMessage)` → app 弹出 `ApprovalModal` → 用户按键后 `gate.resolve(approved)` → 工具线程从 `gate.wait()` 返回（`WQ_APPROVAL_TIMEOUT` 兜底，超时按"拒绝"，绝不会把工具线程挂死）。按钮与快捷键等价（`Y`/Enter 批准，`N`/Esc 拒绝）。

Textual 缺失时 `--tui` 不会崩：打印 `the TUI needs the optional 'textual' dependency: pip install "wqagent[tui]"` 并以退出码 2 结束（普通 CLI 完全不依赖它）。

## 会话（`core/session.py`）

同一个工作区里的多次对话算**一个会话**，落在工作区内：

```
<workspace>/.wqagent/session/
    session.json         机器可读记录
    SESSION_SUMMARY.md   人读的同一段故事
```

```json
{
  "session_id": "uuid", "turn_index": 3,
  "recent_turns": [
    {"turn": 1, "role": "user", "content": "...", "timestamp": "..."},
    {"turn": 2, "role": "assistant", "route": "workflow", "content": "...", "summary": "...", "timestamp": "..."},
    {"turn": 3, "role": "user", "content": "...", "timestamp": "..."}
  ],
  "created_at": "...", "updated_at": "..."
}
```

```python
from wqagent.core.session import (
    load_or_create_session, append_user_turn, append_assistant_turn,
    save_session, build_session_context,
)

session = load_or_create_session(workspace)          # 缺文件/坏文件都安全新建
turn = append_user_turn(session, "帮我搭个 FastAPI 后台")   # -> 1
append_assistant_turn(session, turn=turn, route="workflow",
                      content="Done: app/main.py", summary="built the skeleton")  # -> turn 2
save_session(workspace, session)                     # 写 session.json + SESSION_SUMMARY.md
print(build_session_context(workspace, session))     # 给 intent_router / chat_responder 的上下文块
```

- **轮次编号**：`append_user_turn` 自增并返回编号；`append_assistant_turn(turn=用户轮次)` 记在**下一号**（与上面示例一致），传入更大的号则按传入的记。`turn_index` 始终是已发出的最大号。
- **上限**：单条内容 `MAX_TURN_CONTENT = 4000` 字符；磁盘上保留最近 `MAX_RECENT_TURNS = 40` 轮；**上下文块**只取最近 `CONTEXT_TURNS = 10` 轮 + 最近 `CONTEXT_FILES = 30` 个文件（按 mtime 倒序，跳过 `.wqagent/`），总长硬性 ≤ `MAX_SESSION_CONTEXT = 7000`（先丢最老的轮次，再砍文件清单，最后才截断开头并留 `...[earlier session context omitted]...`）。
- **不抛异常**：JSON 坏了 / 字段类型不对 → 归一化成可用会话；写盘失败只写进 `last_error`（并出现在 SUMMARY 末尾），不会打断运行。
- **接到哪里**：CLI 每次运行前 `load_or_create_session` + `append_user_turn`（并写进 `RuntimeState.session_id/session_turn`，再随 `new_graph_state` 进入图状态与 Memory 快照），运行结束后 `append_assistant_turn`（`route` 取自 `final_answer`，`summary` 取回答首句）+ `save_session`；被 Ctrl-C/异常打断时只保存用户那一轮。入口图的 `intent_router_node`/`chat_responder_node` 通过 `_session_context()` 把上面这个上下文块（以及本轮压缩摘要、上一轮 todo、最近 6 条对话）一起注入提示词——所以"继续/修一下"这类跟进能接上上文。

实测（连跑两次 `wq` 后）：

```
$ wq "你好" -w .\demo          → session.json: turn_index 2, [user, assistant] (chat)
$ wq "继续" -w .\demo          → session.json: turn_index 4, 四条记录，router 的提示词里带着上一轮
```

## 入口图：意图路由（`graph/workflow.py::build_entry_workflow`）

在主工作流之前先跑一张两节点的小图，决定这句话是**聊天**还是**任务**：

```
START ──▶ intent_router ──┬─(chat)─────▶ chat_responder ──▶ END
                          │
                          └─(workflow)─────────────────────▶ END
                                                            （调用方接着跑主工作流）
```

- `intent_router_node`：一次模型调用，要求返回 `{"route":"chat"|"workflow","reason":"...","confidence":0.0-1.0}`；提示词是 `INTENT_ROUTER_PROMPT`（含"`继续`/`修一下`/`运行测试` 若指向前序工作区工作 → workflow"的规则），输入是**用户输入 + session 上下文**（session id / 压缩摘要 / 上一轮计划 / 最近 6 条对话，绝不含工作区事实）。
- **不确定就干活**：JSON 解析失败、route 不是这两个值、或 `confidence < 0.55`（`INTENT_CONFIDENCE_THRESHOLD`）→ 一律走 `workflow`；路由器自身抛异常也只降级成 workflow，不打断运行。
- `chat_responder_node`：**不绑定任何工具**（测试断言 `model.bound == []`），用 `CHAT_RESPONDER_PROMPT` 直接回答，返回 `{chat_response, final_answer}`；提示词明确禁止声称读过文件/跑过命令/搜过网。
- `intent_route_fn(state)`：`"chat" → "chat_responder"`，否则 `"planner"`（在入口图里就是 `END`）。

```python
from wqagent.graph import build_entry_workflow, new_graph_state

app = build_entry_workflow(model=chat_model)
result = app.invoke(new_graph_state("你好", runtime))
result["intent_route"]      # "chat" | "workflow"
result["intent_confidence"] # 0.0 - 1.0
result["intent_reason"]     # 一句话理由
```

**接入方式**：`stream_agent_events(..., entry_router=True)`（默认 **off**，库行为不变）；CLI 默认打开，`--no-router` 关闭，`--resume` 时自动跳过（续跑显然是任务）。事件流多两个事件：`intent`（带 route/reason/confidence）与 `chat_response`；`final_answer` 多一个 `route` 字段：

```python
{"type": "final_answer", "content": "...", "passed": True, "attempts": 0,
 "stopped_early": False, "route": "chat", "intent_route": "chat",
 "intent_reason": "greeting", "intent_confidence": 0.95}
```

CLI 表现（`wq "你好"`）：

```
   🧭 chat (answer directly)
📝 Final
┌──────────────────────────────── chat ────────────────────────────────┐
│ 你好！我是 WQ Agent，可以帮你写代码、跑命令、查资料。                │
└──────────────────────────────────────────────────────────────────────┘
route  chat (no tools, no workspace changes)
why    greeting [0.95]
give it a task ("...") if you wanted work done - or pass --no-router to always run the workflow
```

即使走 chat 分支，检查点与 trace 照样收尾写入（`status: passed`，`task`/`intent_route` 都在 `checkpoint.json` 里），所以"这次只是聊了两句"同样可追溯。

## 事件流与 CLI 输出

[`stream_agent_events()`](src/wqagent/core/agent.py) 是 agent 的入口：它**不再自己跑裸的 ReAct 循环**，而是用 `build_complex_workflow().stream(inputs, stream_mode=["updates", "custom"])` 驱动整张图，把两种 chunk 翻译成一条扁平事件流：

- `custom` chunk = planner 与两个专家的 ReAct 事件（带 `node` 标签）→ 原样转发；
- `updates` chunk = 每个节点的状态更新 → 转成 `node_end` 事件；
- 每个事件同时喂给 `TraceRecorder` 记统计；`CheckpointManager` 在**开始时、每个节点边界、以及 `handoff`/`context_compressed` 这类里程碑事件之后、结束时**落检查点；
- 结束时先落最终检查点与 `trace_saved`，再汇总出一条 `final_answer`（取自 `final` 节点的文本 + verifier 的裁决）。

```python
from wqagent.core.agent import stream_agent_events

for event in stream_agent_events(
    "给 utils.py 加一个 slugify 函数并测试",
    workspace="./demo",
    approval_mode="inline",     # 风险命令策略（None → $WQ_APPROVAL_MODE）
    approval_handler=ask_human, # inline 模式下问谁；None 表示拒绝
    checkpoint_mode="light",    # light / strict / off
    trace_mode="on",            # summary（=on）/ full / off
):
    print(event["type"], event.get("node", ""), event)

# 从检查点继续（等价于 wq --resume <workspace>）：
for event in stream_agent_events("", resume_workspace="./demo", model=chat):
    ...
```

签名（其余参数见下）：`stream_agent_events(task, *, workspace=None, max_attempts=None, max_loops=None, model=None, state=None, inputs=None, graph=None, approval_mode="inline", approval_handler=None, checkpoint_mode="light", resume_workspace=None, trace_mode="on", resume_event=None, on_event=None)`。

- **`state=` 优先**：给了现成的 `RuntimeState`（CLI 就是这么做的）时，它与它的模式设置为准，上面那几个模式参数只在本函数**自己创建** runtime 时生效（`create_runtime()`，见 `core/state.py`）。
- **状态词汇**：检查点的 `status` 依次是 `started`（开局）→ `running`（每次节点切换）→ 终局 `passed` / `failed`；`Ctrl-C` 或异常中断时写 `interrupted` / `failed` 并**把异常继续抛出去**（CLI 自己决定退出码），所以中断的运行也有完整检查点与 trace。
- **事件保持扁平**（`event["type"]`），没有 `{"type": "custom_event", "event": ...}` 包装——CLI 渲染器、`_EventSummary` 与既有测试都按扁平协议消费。
- `events.jsonl`（`strict` 检查点 / `full` trace）里 `node_end` 的 `update` 会省略 `messages`（对话在 `state.json` 里；否则每个节点都要重抄一遍对话）：事件流给调用方的仍是**完整** update。

| 事件 | 字段 | 含义 | CLI 表现 |
| --- | --- | --- | --- |
| `ai_message` | `node`, `content`, `tool_calls`, `loop` | 某个 agent 的一轮输出 | 默认不打印；`-v` 时显示推理文本与 `calling: ...` |
| `tool_call` | `node`, `name`, `args`, `loop` | 即将执行的工具调用 | 青色 `-> 工具名 参数摘要` |
| `tool_result` | `node`, `name`, `result`, `loop` | 工具结果 `{ok, tool, result/error}` | 结果预览；失败显示红色 `x 工具名 failed: ...` |
| `handoff` | `node`, `from`, `to`, `instruction` | planner 把任务交给专家 | 品红 `↪ planner → 🛠 codeAgent` + 指令 |
| `memory` | `node`, `digest`, `chars`, `memory` | 节点把分层 Memory 注入提示词 | 灰色 `↳ memory[planner]: 2 todos, 0 sources, … (1517 chars)` |
| `search_results` | `node`, `query`, `results`, `ok` | searchAgent 一次搜索的命中 | 灰色 `↳ N result(s) for '...'` |
| `todo_update` | `node`, `todos`, `notes` | codeAgent 同步了 todo 状态 | 灰色 `↳ plan: t1=completed` |
| `node_end` | `node`, `update` | 某个图节点（planner/verifier/final）产出 | 📋 Planner / ✅❌ Verifier / 📝 Final 分段展示 |
| `checkpoint_saved` | `status`, `latest_node`, `files`, `git_commit`, `resume_command` | 落了一个检查点 | 默认不打印；失败时黄字，`-v` 时灰字一行 |
| `trace_saved` | `trace_id`, `path`, `tool_calls`, `node_visits`, … | 本次运行的追踪已写完 | 默认不打印；`-v` 时灰字统计 + 路径 |
| `final_answer` | `content`, `passed`, `attempts`, `stopped_early` | 最终回答（**最后一个事件**） | 绿色（通过）/ 红色（未通过）面板 |

`wq` 的实际输出长这样：

```
📋 Planner
write hello.py, then prove it runs
   [ ] t1: write hello.py
   $ exit 1
-> todo_write {"plan_summary": "write hello.py, then run it", ...
Plan recorded: 1 todos (t1=pending); 1 acceptance criteria; 1 verification commands.
-> call_code_agent {"instruction": "write hello.py"}
↪ planner → 🛠 codeAgent
   write hello.py
🛠 codeAgent
-> file_write hello.py
Created hello.py (30 bytes, 1 lines).
↳ plan: t1=completed
Delegated the implementation.        ← planner 的收尾总结
❌ Verifier
   ✗ exit 1 -> exit code 1
Recommended next instruction: run python hello.py instead
📋 Planner                    ← 失败后回到 planner 修订计划并重新交接
↪ planner → 🛠 codeAgent
   make it exit 0
✅ Verifier
   ✓ python hello.py -> exit code 0
📝 Final
┌────────── final answer ──────────┐
...
verdict  passed | attempts 2 | tool calls 4 | nodes planner -> verifier -> planner -> verifier -> final
```

参数：`max_attempts`（CLI `--max-attempts`，默认 3）控制 plan/delegate/verify 轮数；`max_loops`（CLI `--max-steps`，默认 10）控制**每次 attempt 内** planner 的工具轮数（含两个专家交接）。还可以注入 `model=` / `state=` / `inputs=`（现成的 `WQGraphState`）/ `graph=`（预先编译好的图）。

要点：

- 工具结果统一序列化成 `{"ok": true, "tool": "bash", "result": "..."}`；失败是 `{"ok": false, "tool": ..., "error": "..."}`（工具返回的 `Error:` 文本、未知工具、工具抛异常都归到这一类），模型据此自我纠正。
- planner 每轮的 `final_answer` 事件会被丢掉，只有 `final` 节点关闭整轮运行——所以事件流里 `final_answer` 恰好出现一次。
- 两个专家不是图节点：它们在 planner 节点内部同步执行，事件通过 `node` 标签（`searchAgent` / `codeAgent`）区分，交接本身用一条 `handoff` 事件表达。
- [`react_loop`](src/wqagent/core/agent.py)（以及对象封装 `Agent` / `AgentResult`）是**图内部的 ReAct 引擎**：planner、verifier、两个专家都跑在它上面；图相关的导入在 `stream_agent_events` 内部延迟进行，避免 `core.agent` 与 `graph.*` 循环导入。
- 表情与箭头会在**输出流无法编码时自动降级成 ASCII**（GBK 控制台/重定向管道下不会因为 emoji 崩掉）。


## LangGraph 共享状态

[`graph/state.py`](src/wqagent/graph/state.py) 定义整张图共用的状态 dict，所有节点读写同一个 `WQGraphState`：

| 字段 | 类型 | 主要写入方 | 说明 |
| --- | --- | --- | --- |
| `task` | `str` | 入口 | 用户原始请求 |
| `runtime` | `RuntimeState` | 入口 | 工作区 / 超时 / 输出预算，所有工具共用 |
| `messages` | `Annotated[list[BaseMessage], add_messages]` | 各节点 | **自动合并**：节点返回 `{"messages": [msg]}` 是追加而不是覆盖 |
| `plan_summary` | `str` | planner | 计划摘要 |
| `todos` | `list[TodoItem]` | planner / codeAgent | 待办项 `id` / `content` / `status` / `note` |
| `acceptance_criteria` | `list[str]` | planner | 「做完了」的判定标准 |
| `verification_commands` | `list[str]` | planner | 验收要执行的命令 |
| `verification_results` | `list[VerificationResult]` | verifier | `command` / `ok` / `exit_code` / `stdout` / `stderr` |
| `verification_checks` | `list[VerificationCheck]` | verifier | 逐条判定：`name` / `passed` / `detail` |
| `passed` | `bool` | verifier | 验收是否通过 |
| `attempts` / `max_attempts` | `int` | verifier / decide | 已重试轮数与预算（默认 3） |
| `last_actor_summary` | `str` | planner | 本轮主管（早期是 actor）给自己的收尾总结 |
| `last_error` | `str` | verifier | 上次验收失败的原因与建议（通过时清空） |
| `research_notes` | `str` | planner（searchAgent 交接） | 外部资料汇总，含来源 URL |
| `sources` | `list[SourceItem]` | planner（searchAgent 交接） | 去重后的来源（`title` / `url` / `content` / `score`） |
| `agent_handoffs` | `list[AgentHandoff]` | planner | 每次交接：`from_agent` / `to_agent` / `instruction` / `result` |
| `code_agent_summary` | `str` | planner（codeAgent 交接） | 实现专家本轮汇报 |
| `context_summary` | `str` | 压缩步骤 | 本轮历史的压缩版（替换早期对话） |
| `context_token_count` / `context_token_limit` | `int` | `context_monitor_node` | 当前上下文的估算 token 数与预算（默认 400000） |
| `context_should_compress` | `bool` | `context_monitor_node` | 是否已超预算 |
| `context_next_node` | `str` | planner（`"verifier"`）/ verifier 失败（`"planner"`）| 下一个该跑的节点；`context_monitor_route` 用它决定去向 |
| `compression_events` | `list[CompressionEvent]` | 压缩步骤 | 每次压缩一条记录（`node` / `attempt` / `tokens_before` / `tokens_after` / `token_limit` / `summary`） |
| `memory_snapshot` | `LayeredMemory` | 运行时 | 最近一次组装的三层 Memory（见下节）；未组装时为 `{}` |
| `history_summary` | `str` | 入口 / 压缩步骤 | 状态里携带的压缩历史；为空时快照回退读 `HISTORY_SUMMARY.md` |
| `final_answer` | `str` | 出口 | 返回给用户的答案 |

要点：

- `total=False`：27 个键**全部可选**，节点只返回自己改动的键，没返回的键保持原值；图也能从部分状态启动。
- **只有 `messages` 带 reducer**。其余键是「后写覆盖」，而且 LangGraph 把**同一个对象**交给下一个节点——所以 `todos` / `verification_results` 这类列表应当构造新列表再返回，不要就地修改（该行为由 `tests/test_graph_state.py::test_no_reducer_channels_are_passed_by_reference` 固定下来）。`agent_handoffs` / `sources` 是「累积型」字段，节点负责把旧值拼上新值再返回。
- 辅助函数：`new_graph_state(task, runtime, max_attempts=3, **extra)` 生成初始状态；`make_todo(...)` 校验 status（`pending` / `in_progress` / `completed` / `blocked`）；`todo_progress(todos)` 输出 `"2/4 todos completed (1 blocked)"`。

```python
from langgraph.graph import END, START, StateGraph
from wqagent.core.state import RuntimeState
from wqagent.graph import WQGraphState, new_graph_state

state = new_graph_state("给 utils.py 加 slugify 并测试", RuntimeState(workspace="./demo"))

graph = StateGraph(WQGraphState)
graph.add_node("plan", plan_node)      # -> {"plan_summary": ..., "todos": [...], "messages": [...]}
graph.add_node("act", act_node)        # -> {"messages": [...]} 追加，不覆盖
graph.add_edge(START, "plan")
graph.add_edge("plan", "act")
graph.add_edge("act", END)
result = graph.compile().invoke(state)
```

## 图节点（plan → delegate → verify → final）

[`graph/nodes.py`](src/wqagent/graph/nodes.py) 实现三个节点、两个交接工具和路由函数。模型默认懒加载 `create_model()`（用 `set_model(...)` 固定一个，或给单个节点传 `model=`）：

| 节点 | 做什么 | 返回的 state 更新 |
| --- | --- | --- |
| `planner_node` | **主管**：不自己动文件、不自己上网，而是绑定三个工具（`todo_write` / `call_search_agent` / `call_code_agent`）跑 ReAct，按任务需要决定先研究还是直接实现；失败重规划时把 `last_error` 写进 prompt，只交接缺的那一步 | `plan_summary` / `todos` / `acceptance_criteria` / `verification_commands`（来自 `todo_write`）、`research_notes` / `sources`（来自 searchAgent）、`code_agent_summary` / `todos` / `messages`（来自 codeAgent）、`agent_handoffs`（累积）、`last_actor_summary`（收尾总结） |
| `verifier_node` | 先用 `BashTool.execute` 真跑 `verification_commands` 收集 exit_code/stdout/stderr；再用**只读工具**（`build_read_only_tools` → `file_read` / `grep`）巡检；最后强制 `verification_report` 拿 JSON 判决 | `passed`、`attempts`(+1)、`verification_results`、`verification_checks`、`last_error`（失败时）、`todos`（通过则全部置 completed） |
| `final_node` | 先用 `format_final_answer(state)` 生成**确定性报告**（状态里的一切事实：结论、计划、验收项、命令结果、实现专家汇报、来源、交接记录、失败原因、剩余工作），再用 `FINAL_PROMPT` 让模型把它改写成给用户看的回答；模型不可用时保留报告 | `final_answer` |
| `context_monitor_node` | **上下文监测**：把 `messages` 与三层 Memory 负载（`state["memory_snapshot"]`，没有就现组装）一起交给模型的 `get_num_tokens_from_messages()` 估算 token；模型不会数（DeepSeek 就抛 `NotImplementedError`）或计数异常时退回 `len(text) // 4`。`should_compress = token_count > context_token_limit`（默认 400000） | `context_token_count` / `context_should_compress` / `context_next_node`（回显上游设的值，默认 `"verifier"`） |
| `verifier_route` | `passed` → `"final"`；`attempts >= max_attempts` → `"final"`；否则 → `"planner"` | — |
| `context_monitor_route` | `passed` → `"final"`（通过优先）；`context_should_compress` → `"context_compressor"`；否则 → `context_next_node`（默认 `"verifier"`） | — |
| `context_compressor_node` | **上下文压缩**：把「转录 + 三层 Memory 快照」交给模型（`CONTEXT_COMPRESSION_PROMPT`，stage 4）换回九键 JSON（`summary` / `active_goal` / `completed_work` / `open_todos` / `important_files` / `tool_findings` / `sources` / `next_steps` / `risks`）；模型失败或答不出 JSON 时退化为**确定性摘要**（任务 + 计划 + todos + 验收标准 + 验收命令 + 实现汇报 + 失败原因）。然后用 `RemoveMessage(REMOVE_ALL_MESSAGES)` + 一条 `AIMessage(summary)` 重建上下文，把摘要写进 `HISTORY_SUMMARY.md` | `messages`（清空 + 摘要）、`context_summary` / `history_summary`、`context_token_count`（压缩后重算）、`context_should_compress=False`、截断后的 `research_notes` / `agent_handoffs` / `sources` / `code_agent_summary`、追加的 `compression_events`、清空的 `memory_snapshot` |

两个交接工具（[`CallSearchAgentTool` / `CallCodeAgentTool`](src/wqagent/graph/nodes.py)）把专家包成 LLM 可调用的工具：调用时先发一条 `handoff` 事件，再同步跑 `run_search_agent` / `run_code_agent`，最后把专家的**压缩汇报**（`ok` / `summary` / 来源 URL 或 todo 状态）作为工具结果回给主管模型；副作用（笔记、来源、todo、专家对话、交接日志）由节点收集进 state 更新。

三条硬规则：

- **只读验收**：verifier 的工具集里没有 `file_write` / `file_edit` / `bash`——它只能看，不能改；跑验收命令由节点自己完成，验收结果不会被「顺手改一下」污染。
- **命令是硬证据**：模型给出 `passed=true` 但任一验收命令退出码非 0 时，判决会被强制改成失败，并在 `last_error` 里标注 `overruled`；`final_node` 也永远不会因为模型失败而写不出 `final_answer`。
- **结构化输出全容错**：verifier 拿到的 JSON 从不被信任（planner 的计划来自工具调用参数，同样走 `_coerce_todos` / `_text_list` 归一化）。` ```json ` 围栏、前后废话、数组包裹、坏对象后面跟着好对象都能解析；截断、尾逗号、单引号、类型错乱（`"todos": "x"`、`"checks": 5`、`"acceptance_criteria": 7`、`"plan_summary": 42`）、单个对象代替数组、未知 `status`、非对象条目都只降级不抛异常：计划退回上一次（或空计划），verifier 判失败并写明原因。判决走**严格布尔解析**，所以 `"passed": "false"` 会被当成失败而不是真值字符串。

> **`VERIFIER_PROMPT` 的能力声明 vs 当前工具集**：prompt 里写了「read files, grep, run safe shell checks, search the web」和 `NotepadReadTool`，但 verifier 节点目前绑定的是 `build_read_only_tools`（只有 `file_read` / `grep`），`NotepadReadTool` 也还没实现。模型真去调用时只会收到 `unknown tool ...` 的 ToolMessage 并自行绕过（有回归测试固定这个行为），不会崩；要让它名副其实，需要给 verifier 补上 `web_search` / `NotepadReadTool` / 受控的 shell 检查工具。

```python
from wqagent.core.state import RuntimeState
from wqagent.graph import build_workflow, new_graph_state

app = build_workflow()                     # planner → verifier ⇄ planner → final
state = new_graph_state("写一个 hello.py 并跑通", RuntimeState(workspace="./demo"), max_attempts=3)

result = app.invoke(state)
print(result["passed"], result["attempts"])
print(result["final_answer"])
print(result["agent_handoffs"])             # 这次跑都交接给了谁、给了什么指令

# 实时跟 planner / 专家 / verifier 的每一步（节点内部转发到 custom stream）
for mode, chunk in app.stream(state, stream_mode=["custom", "updates"]):
    if mode == "custom":
        print(chunk["type"], chunk.get("node", ""), chunk.get("name", ""))
```

`build_workflow(checkpointer=...)` 可以传入 checkpointer 以启用持久化 / 断点续跑；`build_workflow(model=..., planner_loops=..., verifier_loops=..., agent_model=...)` 可以固定模型与轮数预算（专家默认复用 planner 的模型）。等价的显式接线：
```python
graph = StateGraph(WQGraphState)
graph.add_node("planner", planner_node)     # 内部通过工具调用 searchAgent / codeAgent
graph.add_node("verifier", verifier_node)
graph.add_node("final", final_node)
graph.add_edge(START, "planner")
graph.add_edge("planner", "verifier")
graph.add_conditional_edges("verifier", verifier_route, {"final": "final", "planner": "planner"})
graph.add_edge("final", END)
```

> `actor_node` 与 `ACTOR_PROMPT` 仍然保留（stage 1 / stage 2 的产物），但**不在图里**——它的职责已经由 codeAgent 交接承担；需要旧行为时可以手动接线。

### 带上下文管理的完整图（`build_complex_workflow()`）

```python
from wqagent.graph import build_complex_workflow, new_graph_state

app = build_complex_workflow()          # 参数与 build_workflow 相同：model / planner_loops / verifier_loops / agent_model / checkpointer
result = app.invoke(new_graph_state("长任务…", runtime))
result["compression_events"]            # 每次压缩一条记录
```

```
planner ──▶ context_monitor ──┬─(passed)──────────────▶ final ──▶ END
  ▲                           ├─(预算用尽 attempts≥max)─▶ final
  │                           ├─(超预算)──▶ context_compressor
  │                           │                    │ context_compressor_route
  │                           ▼                    ▼（按 context_next_node）
  └──────────────────────── verifier ◀──────────────┘
                                │
                                └──▶ context_monitor   （每次验证后都再过一次 monitor）
```

- 五个节点：`COMPLEX_NODE_NAMES = (planner, context_monitor, context_compressor, verifier, final)`；`planner → verifier`、`verifier → final` 两条捷径**被移除**，一切经过 monitor。
- **退出条件三道**（按优先级）：`passed` → final；`attempts >= max_attempts` → final（与 `verifier_route` 同一条「不无限重试」硬规则）；否则压缩或继续。少了第二道，失败任务会 `planner → monitor → verifier → monitor → …` 一直转到 LangGraph 的 recursion limit。
- **压缩只做有意义的那次**：`_compression_would_be_pointless()` 记住上次压缩后的 token 数，若它本来就没压进预算（例如 limit 被设成 1），monitor 不再要求压缩——否则 `monitor → compressor` 会空转到 recursion limit。压缩后的尺寸由「摘要上限 + memory 负载」决定，是个确定值，重复压缩不会更小。
- 实测（真实 `deepseek-flash`）：一次 hello.py 任务里 `monitor 1945 → compressor 1945→970 → verifier → monitor 1215 → final`；把 limit 故意设成 60 也只压缩一次就正常收尾，不再空转。
- 目前 **CLI / `stream_agent_events()` 默认就走这张完整图**（`build_complex_workflow`）；`build_workflow()`（短流程）仍在，`stream_agent_events(graph=build_workflow(model=...))` 可以显式用回它。
- **预算可以用环境变量压小**：`WQ_CONTEXT_TOKEN_LIMIT=2000 wq "..."`（`new_graph_state()` 读取，非法值静默回落默认；`wq --show-config` 与启动横幅都会显示当前预算）。实测 `WQ_CONTEXT_TOKEN_LIMIT=2000` 跑一个稍大的任务：`context_monitor 12098 tokens (over budget)` → `🗜 Compressor` → `compressed 75 msgs/13321 tokens -> 1 msg/1908 tokens [HISTORY_SUMMARY.md, NOTEPAD.md]` → 继续跑到 `verifier` → `passed` ✓。

## 三层 Memory（`graph/memory.py`）

记忆由**运行时**组装，不暴露给 agent 任何「写记忆」的工具；需要时把整块快照塞进提示词即可：

```python
from wqagent.graph.memory import build_layered_memory, format_layered_memory_for_prompt

memory = build_layered_memory(state, node="planner")   # state 里必须有 runtime
prompt_block = format_layered_memory_for_prompt(memory)  # 格式化 JSON
```

| 层 | 键 | 内容 | 写入方 |
| --- | --- | --- | --- |
| 1. Rules Layer | `rules` | `RULES_LAYER`：`scope=workspace` / `storage=internal` + 5 条固定规则（只在工作区内、路径不要带 `workspace/` 前缀、持久上下文不要塞进 messages、`TODO.md`=计划态 / `NOTEPAD.md`=持久笔记 / `HISTORY_SUMMARY.md`=压缩历史、agent 不持有记忆写工具） | 代码，快照返回副本 |
| 2. Working Memory | `working_memory` | 当前任务态：`task` / `node` / `session_id` / `session_turn` / `plan_summary` / `todos` / `acceptance_criteria` / `verification_commands` / `research_notes` / `sources`（只留 `title`+`url`）/ `agent_handoffs`（最近 6 条）/ `code_agent_summary` / `verifier_summary` / `last_error` / `attempts` / `max_attempts` | 运行时，从 `WQGraphState` 读取 |
| 3. History Summary Store | `history_summary_store` | `history_path` / `history_exists` / `history_summary` / `notepad_path` / `notepad_exists` / `notepad` / `context_summary` / `compression_events`（最近 3 条） | 运行时读 `NOTEPAD.md` / `HISTORY_SUMMARY.md`；**写只发生在上下文压缩时**（见下） |

- `read_notepad(runtime)` / `read_history_summary(runtime)`：经 `RuntimeState.resolve()` 读工作区里的固定文件名，返回 `{path, exists, content, error}`——**文件不存在不是错误**（`exists=False`），目录同名、二进制内容、超长文件（读上限 20000 字符）都只降级不抛异常。
- 每个自由文本字段都有字符预算，`_short_text()` 截断并在末尾加 `...`：`research_notes` 1600、`code_agent_summary`/`verifier_summary` 1000、`last_error` 1400、`context_summary` 1600、`notepad` 1800、`history_summary` 2200；`_trim_handoffs()` 取最近 6 条**并把每条 `instruction`/`result` 各截到 400 字符**（否则 6 条长指令就能把快照撑到几万字符）。
- 快照返回的都是副本（`todos` / 来源 / 交接记录），改快照不会影响 state，也不会影响 `RULES_LAYER`。
- `session_id` 取 `state["session_id"]` → `runtime.session_id` → 进程级默认值（一次 CLI 运行一个进程）；`session_turn` 取 `state["session_turn"]`，否则是 `attempts + 1`（当前轮次）。`verifier_summary` 目前没有 state 字段写入，所以通常为空。
- 对应的 state 字段是 `memory_snapshot`（最近一次快照；planner / verifier 会把本次注入的快照写回，未组装时为 `{}`）、`history_summary`（状态内的压缩历史，为空时回退读文件）和 `context_*` / `compression_events`（由 `context_monitor_node` 测量与路由，默认预算 `DEFAULT_CONTEXT_TOKEN_LIMIT = 400000`）。

### 注入：每个 Agent 都看得到

三个节点在构造 HumanMessage 前先组装快照、发一条 `memory` 事件，并把 `format_layered_memory_for_prompt(memory)` 追加到消息里：

| 节点 | 组装 | 拼接进 HumanMessage 的内容 |
| --- | --- | --- |
| `planner_node` | `build_layered_memory(state, node="planner")` | `_planner_input()` = 规划简报（任务 / 已有计划 / 验收标准 / 验收命令 / 失败原因）+ 分层 Memory JSON；返回值里带 `memory_snapshot` |
| `verifier_node` | `node="verifier"` | `_verifier_input()` = 证据（命令 exit code/stdout/stderr、验收项、实现汇报）+ 分层 Memory JSON；同样写回 `memory_snapshot` |
| `codeAgent`（`run_code_agent`） | `node="codeAgent"` | `_code_agent_input()` = 任务 + planner 指令 + agent 本地 session/notes + 分层 Memory JSON |

```python
memory = build_layered_memory(state, node="planner")
writer(memory_event(memory, node="planner"))        # {"type": "memory", "node", "digest", "chars", "memory"}
messages = [SystemMessage(PLANNER_PROMPT), HumanMessage(_planner_input(state, memory))]
```

- `memory_event()` 把整块快照、字符数和一行 `memory_digest()`（`"2 todos, 1 sources, 0 handoffs, notepad=yes, history=no"`）打包成事件；CLI 只打印摘要行，程序化消费者可以拿到完整快照。
- **注意一处已收口**：codeAgent 的 HumanMessage 现在只有**一个**记忆块（分层 Memory）——早期那个「agent 本地 session/notes」（`build_memory_snapshot`）已经不再自动拼接，因为它带的信息（计划进度、实现汇报、上次失败、研究笔记）全在分层 Memory 的 `working_memory` 里。`build_memory_snapshot` / `MemorySnapshot` 仍作为公开 API 保留，且**显式**传 `memory=`/`session=` 时依然会渲染（可选的接缝）。
- 实测一次真实运行的事件顺序：`memory[planner] (1314 chars) → handoff → memory[codeAgent] (1966) → memory[verifier] (4274)`；快照里能看到 `notepad=yes`、todos 数与交接记录随流程增长。

### `TODO.md`：运行时真的在用的工作状态

`RULES_LAYER` 说 `TODO.md` 是 working plan state，运行时现在把它当成**真实文件**来维护和读取（[memory.py](src/wqagent/graph/memory.py) 的 `render_todo_plan` / `write_todos` / `read_todos`）：

```
# Working plan

**Task:** create hello.py and run it

write then verify

- [x] t1: write hello.py  <!-- 22 bytes -->
- [~] t2: run it

## Acceptance criteria

- prints Hello

## Verification commands

- `python hello.py`
```

- **写**：planner 发布/修订计划后写一次；codeAgent 每次 `todo_update` 通过 `persist=` 钩子即时写（中途被杀也留下准确的进度）；verifier 通过后把所有 todo 标成 completed 再写一次。写失败只返回 `{ok: false, error}`，不影响运行。
- **读**：新的一轮运行时，如果 state 里**没有** todos、且 `TODO.md` 记录的是**同一个任务**、且计划**尚未完成**，planner 会直接采用文件里的计划（todos / 验收标准 / 验收命令全部恢复），提示词里也会说明「A previous run left an unfinished plan for this exact task in TODO.md」。任务不同、或所有 todo 都已完成，则忽略并照常重新规划（不会把旧计划当成新任务的起点）。
- 标记与提示词共用一套：`[ ]` pending / `[~]` in_progress / `[x]` completed / `[!]` blocked（来自 `graph.state.TODO_MARKS`，`memory.py` 用惰性导入避免循环依赖）。
- **token 怎么数**：`context_monitor_node` 先试模型的 `get_num_tokens_from_messages()`；本仓库配置的 `deepseek-flash` 会抛 `NotImplementedError: ... not presently implemented for model deepseek-flash`，所以实际走的是 `len(text) // 4` 兜底。这个估算是按英文调的，**中文任务会低估**（1 个汉字通常 ≈ 1 token），预算 400000 下影响不大，但如果要更准，把 `_count_tokens` 的兜底换成 CJK 感知的估算即可。

### 上下文压缩（`context_compressor_node`）与完整图的接线

```
planner ─→ context_monitor ─┬─ passed ─────────────→ final
                            ├─ should_compress ────→ context_compressor ─→ context_monitor（重新测量后继续）
                            └─ 否则 ───────────────→ context_next_node（verifier / planner）
```

（上图的接线**还没落地**：`context_compressor` 已经存在，但 `build_workflow()` 仍走 `planner → verifier ⇄ planner → final`，等你说要接再改。）

- 送给压缩模型的是 `SystemMessage(CONTEXT_COMPRESSION_PROMPT) + HumanMessage(任务 + 转录 + 三层 Memory 快照)`；转录超过 `COMPRESS_TRANSCRIPT_CHARS`（120000 字符）时按 **35% 头部 + 65% 尾部**保留、中间丢弃并标注 `[N characters of transcript dropped]`——否则「上下文超预算」本身就装不进一次请求。
- 压缩后的 token 数用与监测节点**完全相同**的口径重算（新消息 + Memory 负载），所以 `context_monitor_node` 再测一次会低于预算，不会来回抖。
- `HISTORY_SUMMARY.md` 是 Markdown：摘要 + 八个分节（Active goal / Completed work / Open todos / Important files / Tool findings / Sources / Next steps / Risks）+ 一行 `<!-- compressed by context_compressor_node: tokens X -> Y (limit Z) -->`；写盘失败只记在事件里（`persisted.ok=false`），不影响压缩结果。
- **`NOTEPAD.md`（durable notes）也由压缩触发时写入**：九键 JSON 里的 `summary` 归 `HISTORY_SUMMARY.md`（压缩历史），其余七个键（`active_goal` + `completed_work` / `important_files` / `tool_findings` / `sources` / `next_steps` / `risks`）构成一个 `## Compression #N - <active goal>` 小节**追加**到 notepad（`<!-- attempt A, tokens X -> Y -->` 记录来源）。文件按 `NOTEPAD_FILE_CHARS`（12000 字符）保留**最新**的若干小节，标题 `# Notepad` 不会重复；压缩结果里没有可持久字段时（例如走了确定性兜底摘要）不写文件，事件里 `notepad.ok=false` + `error: no durable fields in the compression`。
- **谁写哪个文件**：`TODO.md` 由 planner / verifier / codeAgent 在计划变化时写；`HISTORY_SUMMARY.md` 与 `NOTEPAD.md` 只在压缩时由 `context_compressor_node` 写；三者都由运行时读（TODO.md 还能触发计划恢复）。
- 缩减的是**状态体积**：`research_notes`（1600）/ `code_agent_summary`（1000）截断，`agent_handoffs` 只留最近 6 条且每条指令/结果截 400 字符，`sources` 去重后最多 20 条（模型提取到的 URL 会以 `{title: "", url: ...}` 补进来）；`memory_snapshot` 清空，等下一次读取时按压缩后的状态重建。

## 搜索专家（searchAgent）

[`agents/search_agent.py`](src/wqagent/agents/search_agent.py) 是一个只带一个能力的 ReAct 小 Agent：联网查资料。planner 需要外部事实（版本号、API 用法、报错含义）时通过 `call_search_agent` 把它叫起来：

```python
from wqagent.agents import run_search_agent

result = run_search_agent(state, "FastAPI 现在该锁哪个版本？", writer=emit)
# {"ok": True, "error": "", "summary": "...", "queries": [...], "answers": [...],
#  "sources": [{"title", "url", "content", "score"}], "messages": [...], "tool_events": [...]}
if result["ok"]:
    state["research_notes"] = (state.get("research_notes", "") + "\n" + result["summary"]).strip()
```

- 消息是 `SystemMessage(SEARCH_AGENT_PROMPT) + HumanMessage(任务 + 指令 + 已有 research_notes)`，最多 `max_loops`（默认 4）轮；每一轮通过 `writer` 发出 `tool_call` 与 `search_results` 两类事件（`writer` 省略时自动写进 LangGraph 的 custom stream）。
- `sources` 按 URL 去重；`answers` 收集 Tavily 生成的短答案；模型没写出总结（例如撞到 `max_loops`）时用这些材料生成兜底摘要。
- 失败即数据：全部搜索都失败（没配 key、限流、网络）→ `ok=False` + `error`，但已经拿到的 `summary` / `sources` 照常返回；部分成功仍是 `ok=True`。工具结果解析不了也只会变成一条 `ok=False` 的 `search_results` 事件。
- **在图里**：planner 通过 `call_search_agent` 工具调用它，返回的 `summary` 会追加到 `research_notes`、命中来源合并进 `sources`，并留下一条 `agent_handoffs` 记录。
- [`WebSearchTool`](src/wqagent/tools/web_search_tool.py) 的名字是 `web_search`（不在 planner 的工具集里，只有搜索专家绑定它）：

| 参数 | 说明 |
| --- | --- |
| `query` | 自然语言查询 |
| `max_results` | 默认 5，上限 20（越界自动夹取） |
| `search_depth` | `basic`（默认，快）或 `advanced`（更深） |

返回 `{ok, query, answer, results: [{title, url, content, score}], error}`；没有 `TAVILY_API_KEY` 时返回 `{"ok": false, "error": "missing TAVILY_API_KEY"}`，没装 `tavily-python` 时返回对应的安装提示，都不会抛异常。

## 代码专家（codeAgent）

[`agents/code_agent.py`](src/wqagent/agents/code_agent.py) 带**完整工作区工具集 + `todo_update`**，负责把 planner 的一条指令真正落地，并在干活过程中同步 todo 状态：

```python
from wqagent.agents import run_code_agent

result = run_code_agent(state, "t2: 补上 slugify 并跑通 pytest", writer=emit, persist=save_todos)
# {"ok": True, "error": "", "summary": "...", "todos": [...], "messages": [...], "tool_events": [...]}
```

- 工具集：`build_tools(runtime) + [todo_update]` —— 即 `file_read/file_write/file_edit/grep/bash` 加待办更新（提示词要求"必须显式更新 todo 进度"）。
- **todo 边做边更新**：每次 `todo_update` 调用都会立刻应用到 `todos`，写出 `todo_update` 事件（带 `updates` / 结果 `todos` / `notes`），并调用 `persist=` 回调（默认无副作用）——所以中途失败也能留下准确的计划；返回值里的 `todos` 由调用方写回 state。
- **在图里**：planner 通过 `call_code_agent` 工具调用它，返回值里的 `todos` 写回 `todos`、`summary` 写进 `code_agent_summary`、完整对话追加进 `messages`、指令与结果记进 `agent_handoffs`。
- 事件：每次工具调用发 `tool_call`，结果发 `tool_result`（`{ok, result|error}`），待办变更额外发 `todo_update`；`tool_events` 就是 writer 收到的全部事件。
- **失败即数据**：模型/网络异常被捕获成 `ok=false` + `error`，但已经写进工作区的文件和已收集的事件全部保留（半成品比丢掉强）；state 里没有 `runtime` 时直接返回 `ok=false`。
- **layered memory 接口**（后续阶段填充）：[`MemorySnapshot`](src/wqagent/agents/code_agent.py) 有两层 —— `session`（本轮：计划进度 + 上一轮汇报 + 上次验收失败）与 `notes`（持久层：`research_notes`）。`build_memory_snapshot(state, memory=None)` 现在从 state 里现成字段拼装，也接受显式注入；将来换成真正的分层记忆时签名不变。消息结构是 `SystemMessage(CODE_AGENT_PROMPT) + HumanMessage(任务 + 指令 + session + memory)`。
- 已知缺口：`CODE_AGENT_PROMPT` 里提到的 `NotepadAppendTool` / `NotepadReadTool` 还没实现（属于 layered memory 那一阶段），当前模型调用它们会收到 `unknown tool` 反馈后自行绕过。

## 工具一览

| 工具 | 参数 | 说明 |
| --- | --- | --- |
| `file_read` | `file_path`, `offset=1`, `limit=2000` | 读取 UTF-8 文本，返回带行号的内容；二进制文件报错；超长文件提示用 `offset` 续读 |
| `file_write` | `file_path`, `content` | 创建或整文件覆盖，自动创建父目录 |
| `file_edit` | `file_path`, `old_text`, `new_text` | 替换**唯一**匹配片段；0 个匹配或多个匹配都返回错误（含重复行号）；自动兼容 CRLF |
| `grep` | `pattern`, `path="."`, `glob`, `head_limit=100`, `ignore_case` | 正则搜索，输出 `路径:行号: 内容`；跳过二进制/隐藏目录/`node_modules` 等；`WQ_DISABLE_RIPGREP=1` 可强制使用内置扫描器 |
| `bash` | `command`, `timeout_seconds` | 在 workspace 根目录执行命令，返回退出码、stdout、stderr；超时杀掉整个进程树；工具描述里会写明本机 shell 名称（Windows 上是 Windows PowerShell / pwsh），提示模型使用对应语法；**风险命令（装包 / 下载 / 起服务）要过审批**，见下节 |
| `todo_write` | `plan_summary`, `todos`, `acceptance_criteria`, `verification_commands` | planner 用它发布/修订计划（参数经 `_coerce_todos` 归一化，坏参数只降级不抛异常） |
| `todo_update` | `updates` | codeAgent 边做边更新待办状态（`id` + `status` 等） |
| `verification_report` | `passed`, `reason`, `checks`, `recommended_next_instruction` | verifier 的结构化判决（判决走严格布尔解析） |
| `call_search_agent` | `instruction` | **planner 专用**：把研究任务交接给 searchAgent，事件流里发一条 `handoff` |
| `call_code_agent` | `instruction` | **planner 专用**：把实现任务交接给 codeAgent，事件流里发一条 `handoff` |
| `web_search` | `query`, `max_results`, `search_depth` | **searchAgent 专用**：Tavily 联网搜索 |

工具失败不会抛异常打断循环，而是返回以 `Error:` 开头的字符串（事件流里转成 `ok: false`），让模型自行修正参数后重试。

## 执行追踪（`core/trace.py`）

每次运行都会留下一条时间线，用于事后复盘「它到底做了什么、花了多久、哪一步失败」。

| 模式 | 写什么 |
| --- | --- |
| `summary`（默认） | `trace.json`（计数 + 时间线首尾）、`timeline.md`（人读版） |
| `full` | 上述全部，**外加** `events.jsonl`（每个事件一行 JSON，带 `seq`） |
| `off` | 什么都不写，`end()` 返回 `None` |

```
<workspace>/.wqagent/traces/<trace_id>/
    trace.json     统计概览 + timeline_head/tail
    events.jsonl   full：原始事件日志
    timeline.md    人类可读时间线
```

```python
from wqagent.core.trace import TraceRecorder

recorder = TraceRecorder(runtime, "write hello.py")   # trace_id 默认取 runtime.trace_id
recorder.start(inputs, resumed=False)                 # run_start
recorder.record_custom_event(event)                   # 统计：tool_call / failed tool_result / approval / handoff / checkpoint
recorder.record_graph_update(node_end_event)          # node_visits[node] += 1
trace = recorder.end(status="passed", latest_node="final", final_state=state)
# trace["node_visits"] / tool_calls / failed_tool_calls / approval_count /
# checkpoint_count / handoff_count / timeline_head(20) / timeline_tail(80) / timeline_omitted
```

`trace.json` 字段：`trace_id`、`task`、`status`、`latest_node`、`started_at`、`ended_at`、`duration_ms`、`node_visits`、`tool_calls`、`failed_tool_calls`、`approval_count`、`checkpoint_count`、`handoff_count`、`timeline_total/head/tail/omitted`，另附 `passed`、`attempts`、`final_answer`、`last_error`。

实测（离线跑一次完整工作流）：
```
trace_saved: {'trace_id': 'wq-20261003-100814-0ec4', 'mode': 'full', 'status': 'passed',
              'tool_calls': 3, 'failed_tool_calls': 0, 'checkpoint_count': 6,
              'handoff_count': 1, 'duration_ms': 2902}
files: ['events.jsonl', 'timeline.md', 'trace.json']
node_visits: {'planner': 1, 'context_monitor': 2, 'verifier': 1, 'final': 1} | timeline: 30 omitted: 0

# Trace wq-20261003-100814-0ec4
**Status:** passed (latest node: final) | **Duration:** 2.90s
**Nodes:** planner x1, context_monitor x2, verifier x1, final x1
**Counters:** 3 tool calls (0 failed), 0 approval(s), 6 checkpoint(s), 1 handoff(s)
| 4 | 10:08:14 | tool_call | todo_write(plan_summary=do the thing, ...) |
| 8 | 10:08:14 | handoff | planner -> codeAgent: write hello.py |
```

- **统计口径**：`tool_call` → `tool_calls++`；`tool_result` 且 `ok=False` → `failed_tool_calls++`；命令审批过的结果（结构化 `requires_approval` 或渲染文本里的 `requires_approval: True`）→ `approval_count++`；`handoff` / `checkpoint_saved` → 各自计数；`node_end` → `node_visits[node]++`。
- **时间线截断**：`trace.json` 只留**前 20 + 后 80** 条（`timeline_omitted` 记录省略数），这样几百步的运行也不会把概览撑爆；`timeline.md` 同样展示首尾并写明省略条数。
- **`summary` 与 `full` 的区别只有 `events.jsonl`**：统计与时间线两种模式都有。
- 追踪写盘失败（目录不可写）只退化成「内存里继续统计」，绝不影响运行；`off` 模式下 `start` 返回 `None`、记账全部跳过、`end()` 返回 `None`。
- CLI：横幅显示 `trace : summary (WQ_TRACE_MODE), id wq-…`，`-v` 下每次运行结束打印一行统计与 `trace.json` 路径。

## 断点保存与恢复（`core/checkpoint.py`）

运行被打断（Ctrl-C、断网、模型报错、进程被杀）后，可以原地接着跑。三种模式：

| 模式 | 写什么 | 何时写 |
| --- | --- | --- |
| `light`（默认） | `checkpoint.json`（元数据 + 状态摘要）、`RECOVERY.md`、git 快照 | 每次节点切换 + 结束（`passed`/`failed`） |
| `strict` | 上述全部，**外加** `state.json`（含完整对话）与 `events.jsonl`（每个事件一行） | 同上；事件随到随写 |
| `off` | 什么都不写（`--checkpoint-mode off` / `WQ_CHECKPOINT_MODE=off`） | — |

```
<workspace>/.wqagent/checkpoints/
    checkpoint.json   元数据 + 状态摘要（resume 读它）
    RECOVERY.md       人类可读的恢复指南（任务/状态/文件清单/git commit/恢复命令）
    state.json        strict：完整 state
    events.jsonl      strict：每行一个事件
    git/              内部快照仓库（永远不碰工作区自己的 .git）
```

```python
from wqagent.core.checkpoint import CheckpointManager

manager = CheckpointManager(runtime, "write hello.py")     # 也可 CheckpointManager(runtime)
event = manager.save(state, status="running", latest_node="verifier")
# {"type": "checkpoint_saved", "mode", "status", "latest_node", "files",
#  "git_commit", "git_ok", "strict", "resume_command", ...}

inputs, resume_event = CheckpointManager.load_resume_inputs(runtime, max_attempts=3)
# inputs 可直接当 stream_agent_events(inputs=...) 的输入；没有检查点时是 (None, event)
```

- **git 快照是内部的**：用 `git --git-dir=<root>/git --work-tree=<workspace>`，所以工作区**自己的仓库不受影响**（测试断言工作区没有多出 `.git`）；`info/exclude` 排除 `.wqagent/` 与 `.git/`，提交时带 `user.name/email`、`commit.gpgsign=false`、`core.autocrlf=false`、`core.hooksPath=<空目录>`、`gc.auto=0`，并清掉继承来的 `GIT_DIR` 等变量。git 缺失或失败只降级成「有检查点、无快照」，不影响运行。
- **摘要与元数据分开**：`checkpoint.json` 是 `{元数据…, workspace_manifest, git, state: {状态摘要}}`，所以 `load_resume_inputs` 重建出来的 inputs 里**只有真正的 state 键**。（这里踩过一个真 bug：元数据曾与状态摘要平铺在一起，恢复后的运行把旧的 `status/latest_node` 带进新检查点，导致每次保存都写成同一个值；已有回归测试。）
- **恢复语义**：先把 git 快照里的文件 `checkout` 回工作区（快照之后新增的文件保留；被覆盖的文件数写进 `resume` 事件的 warning），然后重建 inputs —— `light` 只有摘要+计划+计数器（无对话，事件里提示 "transcript was not checkpointed"），`strict` 连对话一起恢复（`AIMessage` 的 `tool_calls`、`ToolMessage` 的 `tool_call_id` 原样重建）。
- 命令行：`wq --resume <workspace>`（`resume_command()` 生成的就是这条），恢复时打印 resume 面板（任务/状态/恢复了几个文件/消息数/警告）；运行失败或被 Ctrl-C 时提示 `resume with: wq --resume <workspace>`。

## 命令审批（`core/approval.py`）

工作区围栏管的是**路径**，审批管的是**有副作用的命令**。两道闸门顺序是：先过破坏性黑名单（`rm -rf /` 这类直接拒绝），再过风险分类。

```python
from wqagent.core.approval import classify_command_risk

classify_command_risk("pip install requests")      # "Python package installation"
classify_command_risk("pytest -q && npm install")  # "Node package installation"
classify_command_risk("echo 'pip install x'")      # None —— 只是提到，没有执行
```

`RISK_PATTERNS` 覆盖：`pip install`（含 `python -m pip`）、`uv add` / `uv sync` / `uv pip install`、`npm` / `pnpm` / `yarn install|add`、`curl` / `wget`、`uvicorn`、`python -m http.server`。匹配要求出现在命令开头或 `&&` / `||` / `;` 之后，并且**大小写不敏感**（Windows shell 不区分大小写）。

| 模式 | 行为 |
| --- | --- |
| `inline`（默认） | 构造 `ApprovalRequest(id="approval-xxxxxxxx", command, risk_reason, tool_name)` 交给运行时的 `approval_handler` 等人类决策；**没有 handler 时拒绝**（无人值守的运行不该自己装包） |
| `auto` | 直接执行，结果里带 `requires_approval: True (auto-approved: …)` 标记 |
| `deny` | 不执行，`ShellOutcome.ok=False`，`error` 说明风险原因 |

审批策略来自 `RuntimeState.approval_mode`（`WQ_APPROVAL_MODE` 设默认，非法值回落 `inline`），CLI 用 `--approval-mode` 覆盖，`inline` 模式下的提问长这样：

```
┌──────────────── approval required ─────────────────┐
│ pip install requests                               │
│                                                    │
│ risk     : Python package installation             │
│ tool     : BashTool                                │
│ approval : approval-1a2b3c4d                       │
└────────────────────────────────────────────────────┘
Run this command? [y/N]:
```

```python
state = RuntimeState(workspace="./demo", approval_mode="auto")          # 全放行（带标记）
state = RuntimeState(workspace="./demo", approval_handler=ask_human)    # inline：问 ask_human
BashTool(state).run("pip install requests")
# "$ pip install requests" / "exit_code: 0" /
# "requires_approval: True (auto-approved: Python package installation)"
```

- **闸门放在 `BashTool.execute()`**（真正启动进程的唯一入口），所以 `run_bash()` / `run()`（模型调用的那个）和 verifier 节点跑的 `verification_commands` **都绕不过去**；`core/approval.py` 只管分类与数据结构，模式分支在 `BashTool._decide_approval()`。
- handler 抛异常、stdin 关闭（管道/CI）都按**拒绝**处理，不会把风险命令放过去，也不会崩掉运行。
- 被拒绝的结果是 `Error: command not run - approval required (<原因>); <为什么>…`，模型会看到并自行改走别的路（例如请求用户手动执行）。

### 为什么一遇到 `pip install` 就"卡住"

默认 `--approval-mode inline`：装包/下载/起服务这类命令会**先停下来问你**。这正是设计意图，但以前有两个坑，现在都修好了：

1. **问题只写在 stdout 上** ✗ → 如果你把 stdout 重定向或捕获（`wq "..." | tee run.log`、IDE 的运行面板等），就只看到 stderr 的面板、看不到那句问话，而进程正阻塞在 `input()` 上——看起来就是"卡住"。现在**面板和问话都写 stderr**：

   ```
   ┌──────────────── approval required ────────────────┐
   │ pip install fastapi uvicorn                       │
   │ risk     : Python package installation            │
   │ approval : approval-1a2b3c4d                      │
   └───────────────────────────────────────────────────┘
   the agent is paused - run this command? [y/N] (no answer within 300s counts as no)
   ```

2. **没有超时、stdin 关闭还会 `Abort` 整个运行** ✗ → 现在等 `WQ_APPROVAL_TIMEOUT`（默认 300 秒，`0` 表示一直等）就自动按"否"处理；stdin 关闭（管道/`< /dev/null`）也只是拒绝，不会把整轮运行打断。超时/拒绝的原因会写进工具结果，模型看到后会改走别的路（不再原地打转）。

想让它自己装：`wq "<任务>" --approval-mode auto`（放行并给结果打 `requires_approval: True` 标记），或 `WQ_APPROVAL_MODE=auto`。想一直等：`WQ_APPROVAL_TIMEOUT=0`。

另外 **`pip install` 真的可能超过 120 秒**（下载 wheel、编译依赖），此时 bash 工具会杀掉整棵进程树并在结果里说明：

```
[timed out after 120s - the process tree was terminated; if this command legitimately
 needs longer, call it again with a larger `timeout_seconds` (up to 3600) or split it up.
 If it may be waiting for input it cannot show, pass the tool's non-interactive flag
 instead (pip: `--no-input`, git: `GIT_TERMINAL_PROMPT=0`)]
```

给它更多时间用 `--timeout 600`（上限 3600 秒）。

### 为什么 pip 会在工作区里留下 `pip-*` 空目录

pip 的临时目录（`pip-install-*`、`pip-unpack-*`、`pip-req-build-*`、`pip-ephem-wheel-cache-*`）建在 **`TEMP`/`TMP` 指向的位置**。有些终端（以及包装脚本）把 `TEMP` 设成**相对路径**（例如 `.`），于是这些目录就落在**子进程的工作目录**——也就是你的工作区；命令被超时杀掉时它们不会被清理，于是每失败一次就多几个空目录。实测对照（同一条 pip 命令、5 秒超时）：

```
TEMP 绝对路径 : workspace after: []
TEMP = "."    : workspace after: ['pip-build-tracker-…', 'pip-ephem-wheel-cache-…',
                                 'pip-install-…', 'pip-unpack-…']
```

现在 `BashTool._child_env()` 会把**相对的** `TEMP`/`TMP`/`TMPDIR` 换成工作区内绝对路径 `<workspace>/.wqagent/tmp`（绝对路径留给用户自己的设置），所以这类目录不再污染工作区——它们集中在 `.wqagent/tmp/` 里，且该目录不参与文件清单与 git 快照。已经产生的旧目录可以直接删掉（它们只是空的临时目录）。

顺带说明：关于"卡住"这件事，心跳行现在会**点名正在运行的命令，并附上它最近的输出**，而不是让你猜：

```
... bash has been running for 95s (timeout 120s, 4096 chars of output): python -m pip install -r requirements.txt
    | Collecting fastapi
    | Building wheel for pydantic-core (py3-none-any)
    Ctrl+C stops the run; if it needs longer, raise --timeout
```

（有这行就能分辨：在下载、在编译、还是根本没输出 = 在等输入。）

### 命令结束时不会再多等一秒

以前 bash 工具用 `communicate()` 等**管道 EOF**。若命令自己退出了、但它启动的某个子进程还继承着那对管道（pip 的构建/解包子进程很常见），EOF 就永远不来 —— 于是"依赖其实装好了，agent 却一直没收到通知"，直到超时。现在：

- 等待的是**进程本身**（`Popen.wait`），命令一退出就返回；管道由两个读者线程持续读取，命令结束后只给它们 `DRAIN_GRACE_SECONDS = 2` 秒收尾。
- 如果子进程仍占着管道，结果里会明说，而不是继续等：
  `[the command exited, but a child process still holds its output open - whatever it printed afterwards is not in this result]`
- 子进程的 **stdin 不再继承**（`DEVNULL`）：pip 在私有源上要密码、git 要凭据这类"隐藏提问"会**立刻失败并给出可见错误**，而不是无声地卡住整轮运行（`--no-input`、`GIT_TERMINAL_PROMPT=0` 仍然是好习惯）。
- 输出仍按流分别保存（stdout/stderr 各带标签），超长截断；截断前的完整内容保留在内存里供状态行显示。
- 子进程环境还补了三条：`PYTHONUNBUFFERED=1`（缓冲的子进程看起来就像卡死）、`PIP_NO_INPUT=1`、`PIP_DISABLE_PIP_VERSION_CHECK=1`（后者每次启动都要多一次网络往返；用户自己设过的值不会被覆盖）。

### 装包一直超时？先量一下你的索引延迟

`pip install` 会为**每个包**发多次 HTTPS 请求（simple 索引页、元数据、wheel 下载）。如果索引每次请求要 10 秒，装 FastAPI 这一套（7 个包）就是 3～5 分钟起步，而且中途任何一次连接抖动都会让 pip 重试——屏幕上却可能是 `0 chars of output`。这台机器上的实测：

| 索引 | 两次请求耗时 |
| --- | --- |
| `pypi.org`（默认） | `11.1s (200)` → 紧接着 `SSL: UNEXPECTED_EOF_WHILE_READING` 失败 |
| 清华 TUNA | `4.8s`、`4.7s` |
| 阿里云 | `4.9s`、`4.7s` |
| **中科大 USTC** | **`0.2s`、`0.1s`** |

（TCP 连通性都是好的：`pypi.org:443` 连上只要 1.2 秒——慢的是 TLS/首个字节。）

所以遇到"装包超时、还没有任何输出"，先换索引：

```powershell
# 只对这一次生效
python -m pip install -i https://mirrors.ustc.edu.cn/pypi/simple "fastapi>=0.110" "uvicorn[standard]>=0.27"

# 或永久写进 pip 配置（对所有命令生效）
python -m pip config set global.index-url https://mirrors.ustc.edu.cn/pypi/simple
```

也可以用 `python -m pip download --no-input --timeout 5 --retries 0 -d . six` 单独验证索引是否可用；另外把 `--timeout` 提到 1800（上限 3600）给慢索引留出余量。

现在这类情况在运行中就能看出来：命令跑了 30 秒还一个字都没输出时，心跳会直接说明

```
    | (no output at all yet - usually a stalled network request, e.g. a slow package
      index, or a tool waiting on input it cannot show)
```

超时结果的提示里也会点出这一点（"printed nothing at all → 多半卡在网络/索引"）。

### 模型卡住 / 超时时会发生什么

模型请求由 `create_model()` 配置为**每次 120 秒超时、失败重试 2 次**，所以 provider 卡住时，一轮运行可能**静默 6 分钟以上**：没有事件、没有输出。以前这期间 `trace.json` 也不会落盘（追踪只在运行结束时写），看起来就像"agent 没反应 + trace 是空的"。现在：

- **追踪按节点增量落盘**：每过一个节点（以及每次里程碑检查点）都会重写 `trace.json` + `timeline.md`，字段里带 `complete: true/false`。即使进程被杀，`timeline.md` 也能看到它卡在哪一步。
- **沉默会被解释**：超过 30 秒没有任何事件，stderr 会打印一行
  `... still in codeAgent: no events for 45s (waiting for the model? Ctrl+C stops the run)`。
- **超时可调**：`--request-timeout 45`（或 `$WQ_REQUEST_TIMEOUT`）决定"等一次模型响应"的上限；`--show-config` 显示 `model/request: 45s per attempt`，启动横幅也写 `model : deepseek-flash (request timeout 45s)`。
- **失败时给结论**：provider 超时会打印
  `[the model did not answer within 120s per attempt (it is retried twice). Check the endpoint/network, or lower it with --request-timeout / $WQ_REQUEST_TIMEOUT to fail faster. The run is resumable: the checkpoint and trace up to the last node are on disk.]`
  并把检查点与 trace 收尾为 `failed`（`status: failed`、`complete: true`），再给出 `resume with: wq --resume <workspace>`。

排查顺序：先看 `timeline.md` 最后一行（卡在哪个节点）、再看 `checkpoint.json` 的 `status/latest_node`（确认是不是在等模型），然后按需调 `--request-timeout` 或加固网络。

## 工作区安全模型

1. **单一收口**：所有路径都经过 `RuntimeState.resolve()` → `resolve_in_workspace()`，先 `resolve()` 展开符号链接，再判断是否位于 workspace 内。
2. **拒绝的输入**：`../` 逃逸、指向外部的绝对路径、`C:foo` 这类驱动器相对路径、含 NUL 的路径、指向外部目录的符号链接。
3. **`grep` 不跟随出界的符号链接**；`file_read` / `file_edit` 同样无法读取工作区外的文件。
4. **`bash` 说明**：工作目录固定为 workspace，但它是真实 shell，拥有当前用户的全部权限——**它不是系统级沙箱**。内置只拦截少量“远超工作区范围”的破坏性命令（`rm -rf /`、`format C:`、`diskpart`、fork bomb 等，见 `DEFAULT_BLOCKED_COMMAND_PATTERNS`），可用 `RuntimeState(blocked_command_patterns=...)` 覆盖。需要真正隔离时，请在容器/虚拟机里运行。

## 扩展：新增一个工具

```python
# src/wqagent/tools/my_tool.py
from pydantic import BaseModel, Field
from wqagent.core.state import RuntimeState

class MyInput(BaseModel):
    target: str = Field(description="要处理的目标")

class MyTool:
    name = "my_tool"
    description = "一句话说明这个工具做什么、什么时候用。"
    args_schema = MyInput

    def __init__(self, state: RuntimeState) -> None:
        self.state = state

    def run(self, target: str) -> str:
        path = self.state.resolve(target)      # 自动做工作区检查
        return f"handled {path}"

    def as_tool(self):                         # 交给 bind_tools()
        from langchain_core.tools import StructuredTool
        return StructuredTool.from_function(
            func=self.run, name=self.name,
            description=self.description, args_schema=self.args_schema,
        )
```

把 `MyTool` 加进 `tools/registry.py` 的 `TOOL_CLASSES` 即可，CLI、系统提示词和 `--list-tools` 会自动带上它。

## 测试

```powershell
.\.venv\Scripts\python.exe -m pytest      # 全部用例，无需 API Key
```

覆盖内容：路径围栏（含符号链接逃逸）、三个文件工具、grep（走内置扫描器）、bash（退出码/stderr/超时/黑名单/输出截断/shell 提示/`execute()` 结构化结果）、工具注册表与 `StructuredTool` 参数表、agent 入口的事件流（整图事件顺序与 `node` 标签、`node_end` 携带各节点更新、`handoff` 交接、失败裁决、重规划路径、`on_event`、`workspace=`/`state=`/`inputs=`/`graph=`/`max_loops`、`final_answer` 恰好一次、**新增入口参数：`create_runtime` 的模式/`None` 走环境变量/`resume_from` 变成 workspace、`trace_mode="on"` = summary、开局 `started` 检查点排在第一个 `node_end` 之前、`handoff` 之后紧跟检查点、事件仍是扁平协议、`state=` 优先于模式参数、崩溃写 `failed`、`Ctrl-C` 写 `interrupted` 且异常继续抛出、`resume_workspace=` 恢复并继续（无检查点抛 `LookupError`）、`trace.json` 的 `resumed` 标记**）、ReAct 引擎（事件顺序、JSON `ToolMessage`、工具失败/未知工具/抛异常、`history`）、LangGraph 状态（27 个 channel 与 `add_messages` reducer、真实 `StateGraph` 验证消息累加、部分状态启动、无 reducer 列表的引用语义）、**三层 Memory（`RULES_LAYER` 内容与副本语义、`_short_text` 边界、`_trim_handoffs` 只留最近 6 条且截断每条指令、缺文件/同名目录/二进制/超长文件的读取降级、工作区来源、`working_memory` 各字段与预算、来源只留 title+url、state 值被写坏时的容错、`session_id`/`session_turn` 取值链、`history_summary` 状态优先于文件、压缩事件只留最近 3 条、JSON 格式化与不可序列化兜底、`memory_event`/`memory_digest` 的形状与兜底、**`TODO.md` 渲染/写/读往返、缺文件与垃圾内容、写失败降级**、**`NOTEPAD.md` 小节渲染与追加（标题不重复、旧小节保留、超限只留最新、无 durable 字段时不建文件）、字符串化列表的归一化**）**、分层 Memory 注入（planner/verifier/codeAgent 的 HumanMessage 都带 `"rules"`/`"working_memory"` JSON、`memory` 事件只发一次且带 `node`、快照写回 `memory_snapshot`、NOTEPAD.md/HISTORY_SUMMARY.md 内容随注入进入提示词、注入后原有简报与证据仍在、codeAgent 只剩一个记忆块而显式 `memory=` 仍可渲染）**、`TODO.md` 与运行时联动（planner 写计划、verifier 写完成态、codeAgent 中途 `persist` 写、同任务未完成计划被恢复、异任务/已完成计划被忽略、state 里的计划优先）、shell 提示（planner 简报里出现本机 shell 名与具体禁忌）、**命令审批（`RISK_PATTERNS` 十类风险与理由、链式命令 `&&`/`||`/`;` 与大小写、安全命令不误报、`approval-xxxxxxxx` id 唯一且不可变、模式归一化与 `WQ_APPROVAL_MODE`、三种模式的执行/标记/拒绝、handler 缺失或抛异常时按拒绝处理、黑名单优先于审批、verifier 的验收命令同样受管、**审批提问（面板与问话都在 stderr、stdin 关闭时降级为拒绝而不 `Abort`、显式 n 拒绝、超时按否且不挂死、`WQ_APPROVAL_TIMEOUT` 解析（0=一直等、非法值不会变成 0））**、**断点保存/恢复（三种模式各写什么、`off` 什么都不写、`light` 的元数据/摘要/清单/恢复指南、`strict` 的 state.json 与 events.jsonl、清单排除 `.wqagent/`、超限截断、git 快照不改动工作区 `.git`、空工作区也建首个提交、无 git 时降级、`message_to_dict`/`from_dict` 往返（含 tool_calls）、摘要与元数据不混、恢复工作区文件、strict 恢复对话、任务不一致的警告、损坏 checkpoint 的安全返回、**恢复后的新检查点不继承旧状态**的回归）**、CLI 的 `--checkpoint-mode`/`--resume` 端到端、**CLI 参数声明（`Literal` 可选值、大小写不敏感、非法值（含来自环境变量的）被 click 拒绝且退出码 2、`--show-config` 的 `(cli|env|default)` 来源标注）**、**Ctrl+C 中断（`_InterruptGuard` 首次按键杀命令并提示、无命令时不显示计数、第二次按键 `os._exit(130)`、killer 抛异常不外泄、有/无 watcher 两条路径、`set_wakeup_fd` 收到字节即反应、退出时恢复原 handler 与 wakeup fd、`kill_running_commands()` 杀掉真实长命令且注册表清空、等待被 KeyboardInterrupt 打断时必定杀掉子进程）****、**执行追踪（三种模式、`trace_id` 生成与复用、run_start 记录、五类计数（含从渲染文本识别审批标记）、`node_visits`、`trace.json` 字段齐全、时间线首尾 20/80 与 `timeline_omitted`、`timeline.md` 可读性、`summary` 不写 `events.jsonl`、`full` 每条事件一行且以 `run_end` 收尾、未 start 也能 end、写盘失败只降级、`off` 完全不记录、以及一次真实工作流跑完后的统计与 CLI 端到端）**、图节点（**主管 planner：三个工具绑定、计划/交接/收尾总结、坏参数降级、交接日志累积、searchAgent 与 codeAgent 交接的 state 副作用**、verifier 判定与命令覆盖、**上下文监测（模型自带 token 计数器、异常/缺失/返回值不可用时退回 `len // 4`、超预算判定与边界值、复用已存的 memory 快照、缺 runtime、路由的通过优先与默认去向、planner/verifier 对 `context_next_node` 的设置）**、**上下文压缩（九键 JSON 解析与围栏/废话容错、`RemoveMessage(REMOVE_ALL_MESSAGES)` 在真实 `add_messages` 上的替换语义、模型异常/无 summary/`null` 时的确定性兜底、字段截断与来源合并、压缩事件追加且不就地改 state、`HISTORY_SUMMARY.md` 内容与写盘失败降级、超长转录的头尾保留与丢弃标记、压缩后 token 数确实下降）**、**完整图（`build_complex_workflow()` 的五节点两条条件边、`COMPLEX_NODE_NAMES`/两个 route-targets 常量、节点与 `model`/`loops` 绑定、无压缩的一次通过并核对节点顺序、`limit=1` 时压缩恰好一次并写入 `HISTORY_SUMMARY.md`、`attempts>=max_attempts` 时不再重规划、失败后重规划的九步节点序列、`stream_agent_events(graph=...)` 走完整图）**、prompt 点名但节点没有的工具（`NotepadReadTool` / `web_search`）只降级成 `unknown tool` 不崩、路由、整图 plan→delegate→verify→retry 跑通并核对模型调用次数）、**坏 JSON 容错（围栏/废话/数组包裹、截断/尾逗号/单引号、类型错乱、单对象代替数组、未知 status、非对象条目、严格布尔判决、`tool_choice` 不被支持时的回退链）**、搜索专家（`WebSearchTool` 的成功/缺 key/缺包/空查询/限流/异常响应/参数夹取、`run_search_agent` 的消息构造、查询与来源去重、事件流、`max_loops` 兜底摘要、全部失败 vs 部分成功、无法解析的工具输出）、代码专家（消息四段式与 memory 分层、工具集绑定、todo 边做边更新与 `persist` 回调、坏 todo 更新只记 notes、缺 runtime、模型中断保留半成品、`max_loops` 摘要兜底、工具覆盖）、工作流（`build_workflow()` 的节点与边——**已无 actor 节点**、`model`/`loops`/`agent_model` 绑定、四个 prompt 可从 `stage2` 导入、`final_node` 的成功/失败/模型失败/空回答）、CLI 端到端（`--version` / `--list-tools` / `--show-config` / 默认 workspace 创建 / 节点与专家分段渲染 / `--max-attempts` 与 `--max-steps` / 通过与被预算截断两种退出码 / 真实 HTTP 全图往返）。

`tests/helpers.py` 提供共用的 `ScriptedModel` 与脚本构造器（`plan_call` / `code_call` / `report_call` / `passing_script` 等）；一次完整通过的运行需要 8 次模型调用（planner 3 + codeAgent 2 + verifier 2 + final 1）。

`tests/test_openai_integration.py` 还会在本地启一个**假的 OpenAI 兼容服务**，用真实 HTTP 跑完整链路：验证每次请求发出的工具 schema（planner 的 3 个、codeAgent 的 6 个、verifier 的 2 个只读工具）、工具结果回传给模型、以及 `wq` 命令端到端写出文件。因此不联网也能验证协议层。

`tests/test_acceptance_context.py` 是上下文管理的**验收测试**，一节对应一条验收标准：`WQ_CONTEXT_TOKEN_LIMIT=2000` 触发压缩且任务继续（含非法值回落）、`HISTORY_SUMMARY.md` 落盘且内容可读（跟进节点的快照能看到 `history_exists=true`）、五个节点都注入分层 Memory（含"每个模型调用的人话消息里都有 `"rules"`/`"working_memory"`"）、压缩事件与 CLI 都给出「消息数 + token 数」的前后对比。

## 常见问题

- **`error: no API key found`**：没有 `.env`，或 `API_KEY` 为空。执行 `wq --show-config` 可以看到实际解析到的 `env file` / `model` / `base url`。
- **模型不调用工具**：换一个支持 function calling 的模型；部分小模型需要 `--temperature 0`（已是默认）。
- **Windows 上命令报语法错误**：本机 shell 是 Windows PowerShell 5.1，不支持 `&&`、`||`、`test`、`cat -A`、进程替换 `<(...)`。`bash` 工具的描述里写了本机 shell，**但 planner 只写验收命令、不执行命令**，看不到那句提示——所以 `_planner_input()` 现在会明确写出「Verification commands run in <shell>」以及该 shell 的具体禁忌（`shell_guidance()`，例如「不要用 `python3`，用 `python`」）。加上这句之前，实测两次运行都因为 planner 写了 `python3 hello.py` / `test -f … && echo OK` 而全命令退出码非 0、被 verifier 判失败；加上之后连续三次运行都自动写成 `Test-Path hello.py` / `python hello.py` / `if ($?) { … }` 并通过验收。
- **`wq: command not found`**：没有用 `pip install -e .` 安装，或虚拟环境未激活；也可以直接用 `python -m wqagent`。
- **Windows 中文乱码**：CLI 启动时会把 stdout/stderr 切到 UTF-8，`bash` 工具也强制 PowerShell 使用 UTF-8 输出，Python 子进程会带上 `PYTHONUTF8=1`。

## License

MIT
