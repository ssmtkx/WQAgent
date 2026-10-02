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
- **硬性路径围栏**：任何工具路径都先经过 `core/paths.py` 的 `resolve_in_workspace()`，`../`、绝对路径越界、符号链接逃逸都会被拒绝。
- **OpenAI 兼容**：`providers/openai_provider.py` 用 `.env` 里的 `API_KEY` / `MODEL` / `BASE_URL` 构造 `ChatOpenAI`，因此 OpenAI、Ollama、vLLM、DeepSeek 等都能用。
- **输出预算与超时**：工具输出按字符数截断；`bash` 有超时控制，超时会杀掉整个进程树。
- **无网络可测**：`tests/` 用脚本化的假模型跑完整 Agent 循环，不需要 API Key。

## 环境要求

Python 3.10+（依赖 `langchain-core`、`langchain-openai`、`langgraph`、`typer`、`python-dotenv`、`rich`）。

已通过的验证环境（`pytest`：**223 passed, 1 skipped**，跳过项是 Windows 下需要管理员权限的符号链接用例）：

| 组件 | 版本 |
| --- | --- |
| Python | 3.12.14 (Windows) |
| langchain-core | 1.6.6 |
| langchain-openai | 1.6.7 |
| langgraph | 1.2.12 |
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

复制 `.env.example` 为 `.env` 并填写：

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
| `TASK` | 任务描述（自然语言） | 必填 |
| `-w, --workspace` | 工作区目录，不存在会自动创建 | `./workspace`（`$WQ_WORKSPACE`） |
| `-m, --model` | 模型名，覆盖 `$MODEL` | `.env` 的 `MODEL` |
| `--base-url` | OpenAI 兼容端点，覆盖 `$BASE_URL` | `.env` 的 `BASE_URL` |
| `--api-key` | API Key，覆盖 `$API_KEY` | `.env` 的 `API_KEY` |
| `--env-file` | 指定 `.env` 文件 | 自动查找 |
| `--max-attempts` | plan/act/verify 轮数上限 | `3` |
| `--max-steps` | **每次 attempt 内** actor 的 ReAct 轮数 | `10` |
| `-t, --temperature` | 采样温度 | `0.0` |
| `--timeout` | 单条 shell 命令的默认超时（秒） | `120` |
| `-v, --verbose` | 打印每一步、工具参数与完整输出 | 关 |
| `-q, --quiet` | 只输出最终答案 | 关 |
| `--list-tools` | 列出已注册工具后退出 | — |
| `--show-config` | 打印解析后的配置后退出 | — |
| `-V, --version` | 版本号 | — |

退出码：`0` 成功、`1` 运行失败（含触发步数上限）、`2` 用法错误、`130` 用户中断。

## 项目结构

```
src/wqagent/
├── __init__.py            # 版本号与公开接口
├── __main__.py            # python -m wqagent
├── core/
│   ├── __init__.py
│   ├── state.py           # RuntimeState 数据类（workspace、超时、输出预算、命令黑名单）
│   ├── paths.py           # 工作区路径工具（resolve_in_workspace 等）
│   └── agent.py           # 入口：驱动 LangGraph 工作流并翻译事件；react_loop/Agent 供 actor 节点使用
├── prompts/
│   ├── __init__.py
│   ├── stage1.py          # ACTOR_PROMPT（actor 节点的系统提示词）
│   └── stage2.py          # PLANNER_PROMPT / VERIFIER_PROMPT / FINAL_PROMPT（并转出 ACTOR_PROMPT）
├── graph/
│   ├── __init__.py
│   ├── state.py           # LangGraph 共享状态：WQGraphState / TodoItem / VerificationResult
│   ├── nodes.py           # planner_node / actor_node / verifier_node / final_node / verifier_route
│   └── workflow.py        # build_workflow()：把节点接线并 compile()
├── tools/
│   ├── __init__.py
│   ├── registry.py        # build_tools(state) / build_read_only_tools(state)
│   ├── file_tools.py      # FileReadTool / FileWriteTool / FileEditTool
│   ├── grep_tool.py       # GrepTool（优先 ripgrep，缺失时内置扫描器）
│   ├── bash_tool.py       # BashTool（超时 + 进程树清理 + 危险命令过滤 + 结构化 ShellOutcome）
│   └── graph_tools.py     # todo_write / todo_update / verification_report（结构化输出工具）
├── providers/
│   ├── __init__.py
│   └── openai_provider.py # create_model() / load_env() / describe_config()
└── cli/
    ├── __init__.py
    └── app.py             # typer 入口，wq 命令（消费事件流并实时渲染）

tests/                     # pytest：路径围栏、5 个工具、ReAct 事件流、图状态、CLI 端到端（含真实 HTTP）
```

## 事件流与 CLI 输出

[`stream_agent_events()`](src/wqagent/core/agent.py) 是 agent 的入口：它**不再自己跑裸的 ReAct 循环**，而是用 `build_workflow().stream(inputs, stream_mode=["updates", "custom"])` 驱动整张图，把两种 chunk 翻译成一条扁平事件流：

- `custom` chunk = actor 每一步的 ReAct 事件（带 `node` 标签）→ 原样转发；
- `updates` chunk = 每个节点的状态更新 → 转成 `node_end` 事件；
- 结束时汇总出一条 `final_answer`（取自 `final` 节点的文本 + verifier 的裁决）。

```python
from wqagent.core.agent import stream_agent_events

for event in stream_agent_events("给 utils.py 加一个 slugify 函数并测试", workspace="./demo"):
    print(event["type"], event.get("node", ""), event)
```

| 事件 | 字段 | 含义 | CLI 表现 |
| --- | --- | --- | --- |
| `ai_message` | `node`, `content`, `tool_calls`, `loop` | actor 的一轮输出 | 默认不打印；`-v` 时显示推理文本与 `calling: ...` |
| `tool_call` | `node`, `name`, `args`, `loop` | 即将执行的工具调用 | 青色 `-> 工具名 参数摘要` |
| `tool_result` | `node`, `name`, `result`, `loop` | 工具结果 `{ok, tool, result/error}` | 结果预览；失败显示红色 `x 工具名 failed: ...` |
| `node_end` | `node`, `update` | 某个节点（planner/actor/verifier/final）产出 | 📋 Planner / 🔧 Actor / ✅❌ Verifier / 📝 Final 分段展示 |
| `final_answer` | `content`, `passed`, `attempts`, `stopped_early` | 最终回答（**最后一个事件**） | 绿色（通过）/ 红色（未通过）面板 |

`wq` 的实际输出长这样：

```
📋 Planner
write hello.py, then prove it runs
   [ ] t1: write hello.py
   $ exit 1
🔧 Actor
-> file_write hello.py
Created hello.py (30 bytes, 1 lines).
Wrote hello.py and marked t1 done.
❌ Verifier
   ✗ exit 1 -> exit code 1
Recommended next instruction: run python hello.py instead
📋 Planner                    ← 失败后回到 planner 修订计划
🔧 Actor
Re-ran the program.
✅ Verifier
   ✓ python hello.py -> exit code 0
📝 Final
┌────────── final answer ──────────┐
...
verdict  passed | attempts 2 | tool calls 2 | nodes planner -> actor -> verifier -> planner -> actor -> verifier -> final
```

参数：`max_attempts`（CLI `--max-attempts`，默认 3）控制 plan/act/verify 轮数；`max_loops`（CLI `--max-steps`，默认 10）控制**每次 attempt 内** actor 的 ReAct 轮数。还可以注入 `model=` / `state=` / `inputs=`（现成的 `WQGraphState`）/ `graph=`（预先编译好的图）。

要点：

- 工具结果统一序列化成 `{"ok": true, "tool": "bash", "result": "..."}`；失败是 `{"ok": false, "tool": ..., "error": "..."}`（工具返回的 `Error:` 文本、未知工具、工具抛异常都归到这一类），模型据此自我纠正。
- actor 每轮自己的 `final_answer` 事件会被丢掉，只有 `final` 节点关闭整轮运行——所以事件流里 `final_answer` 恰好出现一次。
- [`react_loop`](src/wqagent/core/agent.py)（以及对象封装 `Agent` / `AgentResult`）是**图内部的 ReAct 引擎**，由 `actor_node` 使用，也可以单独调用；图相关的导入在 `stream_agent_events` 内部延迟进行，避免 `core.agent` 与 `graph.*` 循环导入。


## LangGraph 共享状态

[`graph/state.py`](src/wqagent/graph/state.py) 定义整张图共用的状态 dict，所有节点读写同一个 `WQGraphState`：

| 字段 | 类型 | 主要写入方 | 说明 |
| --- | --- | --- | --- |
| `task` | `str` | 入口 | 用户原始请求 |
| `runtime` | `RuntimeState` | 入口 | 工作区 / 超时 / 输出预算，所有工具共用 |
| `messages` | `Annotated[list[BaseMessage], add_messages]` | 各节点 | **自动合并**：节点返回 `{"messages": [msg]}` 是追加而不是覆盖 |
| `plan_summary` | `str` | plan | 计划摘要 |
| `todos` | `list[TodoItem]` | plan / act | 待办项 `id` / `content` / `status` / `note` |
| `acceptance_criteria` | `list[str]` | plan | 「做完了」的判定标准 |
| `verification_commands` | `list[str]` | plan | 验收要执行的命令 |
| `verification_results` | `list[VerificationResult]` | verify | `command` / `ok` / `exit_code` / `stdout` / `stderr` |
| `verification_checks` | `list[VerificationCheck]` | verify | 逐条判定：`name` / `passed` / `detail` |
| `passed` | `bool` | verify | 验收是否通过 |
| `attempts` / `max_attempts` | `int` | verify / decide | 已重试轮数与预算（默认 3） |
| `last_actor_summary` | `str` | actor | actor 本轮结束时给自己的总结 |
| `last_error` | `str` | verify | 上次验收失败的原因与建议（通过时清空） |
| `final_answer` | `str` | 出口 | 返回给用户的答案 |

要点：

- `total=False`：15 个键**全部可选**，节点只返回自己改动的键，没返回的键保持原值；图也能从部分状态启动。
- **只有 `messages` 带 reducer**。其余键是「后写覆盖」，而且 LangGraph 把**同一个对象**交给下一个节点——所以 `todos` / `verification_results` 这类列表应当构造新列表再返回，不要就地修改（该行为由 `tests/test_graph_state.py::test_no_reducer_channels_are_passed_by_reference` 固定下来）。
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

## 图节点（plan → act → verify → final）

[`graph/nodes.py`](src/wqagent/graph/nodes.py) 实现四个节点和路由函数。模型默认懒加载 `create_model()`（用 `set_model(...)` 固定一个，或给单个节点传 `model=`）：

| 节点 | 做什么 | 返回的 state 更新 |
| --- | --- | --- |
| `planner_node` | 没有 todos → 生成计划；有 todos 且有 `last_error` → 按失败原因修订；已有计划且无失败 → 原样沿用（**不调用模型**）。用 `bind_tools([todo_write], tool_choice="todo_write")` 强制结构化输出，模型不支持强制工具调用时自动退回解析 JSON 内容 | `plan_summary` / `todos` / `acceptance_criteria` / `verification_commands` |
| `actor_node` | 用 `build_tools(runtime) + [todo_update]` 跑 ReAct（默认 `max_loops=10`），prompt 是 `ACTOR_PROMPT` + 历史 + 计划 + 验收标准 + 待跑命令；每一步的事件转发给 `on_event` 与 LangGraph custom stream | `messages`（**只回本轮增量**，由 `add_messages` 合并）、`last_actor_summary`、`todos`（按 `todo_update` 调用更新） |
| `verifier_node` | 先用 `BashTool.execute` 真跑 `verification_commands` 收集 exit_code/stdout/stderr；再用**只读工具**（`build_read_only_tools` → `file_read` / `grep`）巡检；最后强制 `verification_report` 拿 JSON 判决 | `passed`、`attempts`(+1)、`verification_results`、`verification_checks`、`last_error`（失败时）、`todos`（通过则全部置 completed） |
| `final_node` | 先用 `format_final_answer(state)` 生成**确定性报告**（状态里的一切事实：结论、计划、验收项、命令结果、actor 汇报、失败原因、剩余工作），再用 `FINAL_PROMPT` 让模型把它改写成给用户看的回答；模型不可用时保留报告 | `final_answer` |
| `verifier_route` | `passed` → `"final"`；`attempts >= max_attempts` → `"final"`；否则 → `"planner"` | — |

两条硬规则：

- **只读验收**：verifier 的工具集里没有 `file_write` / `file_edit` / `bash`——它只能看，不能改；跑验收命令由节点自己完成，验收结果不会被「顺手改一下」污染。
- **命令是硬证据**：模型给出 `passed=true` 但任一验收命令退出码非 0 时，判决会被强制改成失败，并在 `last_error` 里标注 `overruled`；`final_node` 也永远不会因为模型失败而写不出 `final_answer`。
- **结构化输出全容错**：planner/verifier 拿到的 JSON 从不被信任。` ```json ` 围栏、前后废话、数组包裹、坏对象后面跟着好对象都能解析；截断、尾逗号、单引号、类型错乱（`"todos": "x"`、`"checks": 5`、`"acceptance_criteria": 7`、`"plan_summary": 42`）、单个对象代替数组、未知 `status`、非对象条目都只降级不抛异常：planner 退回上一次计划（或空计划），verifier 判失败并写明原因。判决走**严格布尔解析**，所以 `"passed": "false"` 会被当成失败而不是真值字符串。

```python
from wqagent.core.state import RuntimeState
from wqagent.graph import build_workflow, new_graph_state

app = build_workflow()                     # planner → actor → verifier ⇄ planner → final
state = new_graph_state("写一个 hello.py 并跑通", RuntimeState(workspace="./demo"), max_attempts=3)

result = app.invoke(state)
print(result["passed"], result["attempts"])
print(result["final_answer"])

# 实时跟 actor 的每一步（节点内部转发到 custom stream）
for mode, chunk in app.stream(state, stream_mode=["custom", "updates"]):
    if mode == "custom":
        print(chunk["type"], chunk.get("name", ""))
```

`build_workflow(checkpointer=...)` 可以传入 checkpointer 以启用持久化 / 断点续跑。等价的显式接线：

```python
graph = StateGraph(WQGraphState)
graph.add_node("planner", planner_node)
graph.add_node("actor", actor_node)
graph.add_node("verifier", verifier_node)
graph.add_node("final", final_node)
graph.add_edge(START, "planner")
graph.add_edge("planner", "actor")
graph.add_edge("actor", "verifier")
graph.add_conditional_edges("verifier", verifier_route, {"final": "final", "planner": "planner"})
graph.add_edge("final", END)
```

## 工具一览

| 工具 | 参数 | 说明 |
| --- | --- | --- |
| `file_read` | `file_path`, `offset=1`, `limit=2000` | 读取 UTF-8 文本，返回带行号的内容；二进制文件报错；超长文件提示用 `offset` 续读 |
| `file_write` | `file_path`, `content` | 创建或整文件覆盖，自动创建父目录 |
| `file_edit` | `file_path`, `old_text`, `new_text` | 替换**唯一**匹配片段；0 个匹配或多个匹配都返回错误（含重复行号）；自动兼容 CRLF |
| `grep` | `pattern`, `path="."`, `glob`, `head_limit=100`, `ignore_case` | 正则搜索，输出 `路径:行号: 内容`；跳过二进制/隐藏目录/`node_modules` 等；`WQ_DISABLE_RIPGREP=1` 可强制使用内置扫描器 |
| `bash` | `command`, `timeout_seconds` | 在 workspace 根目录执行命令，返回退出码、stdout、stderr；超时杀掉整个进程树；工具描述里会写明本机 shell 名称（Windows 上是 Windows PowerShell / pwsh），提示模型使用对应语法 |

工具失败不会抛异常打断循环，而是返回以 `Error:` 开头的字符串（事件流里转成 `ok: false`），让模型自行修正参数后重试。

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

覆盖内容：路径围栏（含符号链接逃逸）、三个文件工具、grep（走内置扫描器）、bash（退出码/stderr/超时/黑名单/输出截断/shell 提示/`execute()` 结构化结果）、工具注册表与 `StructuredTool` 参数表、agent 入口的事件流（整图事件顺序与 `node` 标签、`node_end` 携带各节点更新、失败裁决、重规划路径、`on_event`、`workspace=`/`state=`/`inputs=`/`graph=`/`max_loops`、`final_answer` 恰好一次）、ReAct 引擎（事件顺序、JSON `ToolMessage`、工具失败/未知工具/抛异常、`history`）、LangGraph 状态（15 个 channel 与 `add_messages` reducer、真实 `StateGraph` 验证消息累加、部分状态启动、无 reducer 列表的引用语义）、图节点（planner 生成/修订/沿用/JSON 兜底、actor 工具执行与 `todo_update`、verifier 判定与命令覆盖、路由、整图 plan→act→verify→retry 跑通并核对模型调用次数）、**坏 JSON 容错（91 个用例：围栏/废话/数组包裹/坏对象后跟好对象、截断/尾逗号/单引号、类型错乱、单对象代替数组、未知 status、非对象条目、严格布尔判决、`tool_choice` 不被支持时的回退链）**、工作流（`build_workflow()` 的节点与边、`model`/`loops` 绑定、四个 prompt 可从 `stage2` 导入、`final_node` 的成功/失败/模型失败/空回答）、CLI 端到端（`--version` / `--list-tools` / `--show-config` / 默认 workspace 创建 / 节点分段渲染 / `--max-attempts` 与 `--max-steps` / 通过与被预算截断两种退出码 / 真实 HTTP 全图往返）。

`tests/test_openai_integration.py` 还会在本地启一个**假的 OpenAI 兼容服务**，用真实 HTTP 跑完整链路：验证 `bind_tools()` 发出的 5 个工具 schema、工具结果回传给模型、以及 `wq` 命令端到端写出文件。因此不联网也能验证协议层。

## 常见问题

- **`error: no API key found`**：没有 `.env`，或 `API_KEY` 为空。执行 `wq --show-config` 可以看到实际解析到的 `env file` / `model` / `base url`。
- **模型不调用工具**：换一个支持 function calling 的模型；部分小模型需要 `--temperature 0`（已是默认）。
- **Windows 上命令报语法错误**：本机 shell 是 Windows PowerShell 5.1，不支持 `&&`、`||`。`bash` 工具的描述里已经写明本机 shell 名称，若模型仍写 POSIX 语法，可在任务里补一句或在 `ACTOR_PROMPT` 中强调。
- **`wq: command not found`**：没有用 `pip install -e .` 安装，或虚拟环境未激活；也可以直接用 `python -m wqagent`。
- **Windows 中文乱码**：CLI 启动时会把 stdout/stderr 切到 UTF-8，`bash` 工具也强制 PowerShell 使用 UTF-8 输出，Python 子进程会带上 `PYTHONUTF8=1`。

## License

MIT
