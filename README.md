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

Python 3.10+（依赖 `langchain-core`、`langchain-openai`、`typer`、`python-dotenv`、`rich`）。

已通过的验证环境（`pytest`：**69 passed, 1 skipped**，跳过项是 Windows 下需要管理员权限的符号链接用例）：

| 组件 | 版本 |
| --- | --- |
| Python | 3.12.14 (Windows) |
| langchain-core | 1.6.6 |
| langchain-openai | 1.6.7 |
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
| `--max-steps` | 模型/工具往返上限 | `30` |
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
│   └── agent.py           # ReAct 循环：stream_agent_events() + Agent/AgentResult
├── prompts/
│   ├── __init__.py
│   └── stage1.py          # ACTOR_PROMPT（ReAct actor 节点的系统提示词）
├── tools/
│   ├── __init__.py
│   ├── registry.py        # build_tools(state) -> list[StructuredTool]
│   ├── file_tools.py      # FileReadTool / FileWriteTool / FileEditTool
│   ├── grep_tool.py       # GrepTool（优先 ripgrep，缺失时内置扫描器）
│   └── bash_tool.py       # BashTool（超时 + 进程树清理 + 危险命令过滤）
├── providers/
│   ├── __init__.py
│   └── openai_provider.py # create_model() / load_env() / describe_config()
└── cli/
    ├── __init__.py
    └── app.py             # typer 入口，wq 命令（消费事件流并实时渲染）

tests/                     # pytest：路径围栏、5 个工具、ReAct 事件流、CLI 端到端（含真实 HTTP）
```

## ReAct 事件流

核心是 [`stream_agent_events()`](src/wqagent/core/agent.py)：一个生成器，每完成一步就 `yield` 一个事件字典，CLI 边收边用 rich 打印。

```python
from wqagent.core.agent import stream_agent_events

for event in stream_agent_events("给 utils.py 加一个 slugify 函数并测试", workspace="./demo"):
    print(event["type"], event)
```

| 事件 | 字段 | 含义 | CLI 表现 |
| --- | --- | --- | --- |
| `ai_message` | `content`, `tool_calls`, `loop` | 模型的一轮输出 | 默认不打印；`-v` 时显示推理文本与 `calling: ...` |
| `tool_call` | `name`, `args`, `loop` | 即将执行的工具调用 | 青色 `-> 工具名 参数摘要` |
| `tool_result` | `name`, `result`, `loop` | 工具结果 `{ok, tool, result/error}` | 结果预览；失败显示红色 `x 工具名 failed: ...` |
| `final_answer` | `content`, `loop`, `stopped_early` | 最终回答（**最后一个事件**） | 绿色面板；触发上限时为黄色 `stopped early` |

流程与 `max_loops`（CLI 里是 `--max-steps`）：`SystemMessage(ACTOR_PROMPT)` + `HumanMessage(task)` → `model.bind_tools(build_tools(state))` → 循环「模型输出 → 执行工具 → 把结果的 JSON 作为 `ToolMessage` 回灌」，直到模型不再调用工具或达到 `max_loops`。函数默认 `max_loops=10`；CLI 的 `--max-steps` 默认 `30`，会作为 `max_loops` 传进去。

要点：

- 工具结果统一序列化成 `{"ok": true, "tool": "bash", "result": "..."}`；失败是 `{"ok": false, "tool": ..., "error": "..."}`（工具返回的 `Error:` 文本、未知工具、工具抛异常都归到这一类），模型据此自我纠正。
- `stream_agent_events(task, workspace=...)` 会在需要时自己 `create_model()`、自己创建 workspace；也可以注入 `model=` / `tools=` / `state=` / `system_prompt=`，测试用假模型即可跑通。
- `Agent` 类是同一条循环的对象封装（返回 `AgentResult`，带 `steps` / `tool_calls` / `messages`），给脚本和测试用；CLI 走的是事件流。

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

覆盖内容：路径围栏（含符号链接逃逸）、三个文件工具、grep（走内置扫描器）、bash（退出码/stderr/超时/黑名单/输出截断/shell 提示）、工具注册表与 `StructuredTool` 参数表、ReAct 事件流（事件顺序、`ACTOR_PROMPT`、JSON `ToolMessage`、工具失败、未知工具、抛异常、`max_loops` 上限、`history`/`on_event`、模型注入）、`Agent` 循环、CLI 端到端（`--version` / `--list-tools` / `--show-config` / 默认 workspace 创建 / 脚本化模型跑完整任务 / 上限退出码）。

`tests/test_openai_integration.py` 还会在本地启一个**假的 OpenAI 兼容服务**，用真实 HTTP 跑完整链路：验证 `bind_tools()` 发出的 5 个工具 schema、工具结果回传给模型、以及 `wq` 命令端到端写出文件。因此不联网也能验证协议层。

## 常见问题

- **`error: no API key found`**：没有 `.env`，或 `API_KEY` 为空。执行 `wq --show-config` 可以看到实际解析到的 `env file` / `model` / `base url`。
- **模型不调用工具**：换一个支持 function calling 的模型；部分小模型需要 `--temperature 0`（已是默认）。
- **Windows 上命令报语法错误**：本机 shell 是 Windows PowerShell 5.1，不支持 `&&`、`||`。`bash` 工具的描述里已经写明本机 shell 名称，若模型仍写 POSIX 语法，可在任务里补一句或在 `ACTOR_PROMPT` 中强调。
- **`wq: command not found`**：没有用 `pip install -e .` 安装，或虚拟环境未激活；也可以直接用 `python -m wqagent`。
- **Windows 中文乱码**：CLI 启动时会把 stdout/stderr 切到 UTF-8，`bash` 工具也强制 PowerShell 使用 UTF-8 输出，Python 子进程会带上 `PYTHONUTF8=1`。

## License

MIT
