import os
import sys
import json
import requests
import io
import time
import re
from datetime import datetime
from typing import Dict, Any, List, Union, TextIO

today_str = datetime.now().strftime("%Y年%m月%d日")
now = datetime.now()
weekday_str = ["星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日"][now.weekday()]
# Windows 控制台 UTF-8 支持（可选）
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# 从环境变量获取 API Key
API_KEY = os.environ.get("SILICONFLOW_SK")
if not API_KEY:
    print("Error: 请先设置环境变量 SILICONFLOW_SK=你的 Secret Key")
    sys.exit(1)

# 定义工具函数

def add_numbers(a: float, b: float) -> str:
    """对两个数字求和，返回字符串结果"""
    return str(a + b)

import sys
import io
import chess
import chess.engine
import chess.pgn
import io
# === Stockfish 函数工具定义 ===
STOCKFISH_PATH = r"C:\Users\Administrator\Downloads\Compressed\stockfish-windows-x86-64-avx2_2\stockfish\stockfish-windows-x86-64-avx2.exe"

def evaluate_position(fen: str, engine_path: str = STOCKFISH_PATH, time_limit: float = 1.0) -> Dict[str, Any]:
    board = chess.Board(fen)
    with chess.engine.SimpleEngine.popen_uci(engine_path) as engine:
        result = engine.analyse(board, chess.engine.Limit(time=time_limit))
        score = result['score'].white()
    return {"score": str(score)}


def get_best_move(fen: str, engine_path: str = STOCKFISH_PATH, time_limit: float = 1.0) -> Dict[str, Any]:
    board = chess.Board(fen)
    with chess.engine.SimpleEngine.popen_uci(engine_path) as engine:
        result = engine.play(board, chess.engine.Limit(time=time_limit))
    return {"best_move": result.move.uci()}


def evaluate_move_list(
    moves: List[str],
    engine_path: str = STOCKFISH_PATH,
    time_limit: float = 1.0,
    start_fen: str = chess.Board().fen()
) -> Dict[str, Any]:
    board = chess.Board(start_fen)
    evaluations: List[str] = []
    with chess.engine.SimpleEngine.popen_uci(engine_path) as engine:
        for move_str in moves:
            try:
                move = board.parse_san(move_str)
            except ValueError:
                move = chess.Move.from_uci(move_str)
            board.push(move)
            result = engine.analyse(board, chess.engine.Limit(time=time_limit))
            evaluations.append(str(result['score'].white()))
    return {"evaluations": evaluations}


def evaluate_board(fen: str, engine_path: str = STOCKFISH_PATH, time_limit: float = 1.0) -> Dict[str, Any]:
    board = chess.Board(fen)
    with chess.engine.SimpleEngine.popen_uci(engine_path) as engine:
        result = engine.analyse(board, chess.engine.Limit(time=time_limit))
        score = result['score'].white()
    return {"score": str(score)}

def pgn_to_fen(pgn: Union[str, TextIO], game_index: int = 0) -> str:
    """
    从初始局面出发，根据 PGN 字符串或文件读取第 game_index 个对局，返回最终局面的 FEN。

    :param pgn: PGN 格式内容，可以是完整 PGN 文本（str）或文本流（TextIO）。
    :param game_index: 在多局 PGN 时，选择第几局（0 基础索引）。默认第 0 局。
    :return: 对应对局走完后的 FEN 字符串。
    """
    # 将输入统一为 TextIO
    if isinstance(pgn, str):
        pgn_io = io.StringIO(pgn)
    else:
        pgn_io = pgn

    # 跳转到指定局
    game = None
    for idx in range(game_index + 1):
        game = chess.pgn.read_game(pgn_io)
        if game is None:
            raise ValueError(f"无法在 PGN 中找到第 {game_index} 局（索引从 0 开始）")

    # 拿到对局终局的棋盘
    board = game.end().board()
    return board.fen()
import subprocess
import tempfile
import os

def run_code(code: str) -> str:
    try:
        # 将用户代码写入临时 Python 文件
        with tempfile.NamedTemporaryFile(delete=False, suffix=".py", mode="w", encoding="utf-8") as f:
            f.write(code)
            temp_path = f.name

        # 使用 subprocess 执行该临时文件
        result = subprocess.run(
            ["python", temp_path],
            capture_output=True,
            text=True
        )

        # 删除临时文件
        os.unlink(temp_path)

        # 处理输出
        if result.returncode != 0:
            return f"[run_code] ❌ 错误:\n{result.stderr}"
        elif result.stdout.strip():
            return result.stdout.strip()
        else:
            return "[run_code] ✔ 执行成功，无输出"
    except Exception as e:
        return f"[run_code] ❌ 执行异常：{e}"

def run_terminal_command(command: str) -> str:
    """
    执行终端命令并返回结果
    
    :param command: 要执行的命令字符串
    :return: 命令执行的输出结果
    """
    try:
        # 使用subprocess执行命令
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True
        )
        
        # 处理输出
        if result.returncode != 0:
            return f"[run_terminal] ❌ 错误 (返回码: {result.returncode}):\n{result.stderr}"
        elif result.stdout.strip():
            return result.stdout.strip()
        else:
            return "[run_terminal] ✔ 执行成功，无输出"
    except Exception as e:
        return f"[run_terminal] ❌ 执行异常：{e}"
    
# 其他工具示例，可按需扩展

def semantic_rerank(query: str, documents: list, top_n: int = 5) -> str:
    api_key = os.environ.get("LANGSEARCH_API_KEY")
    if not api_key:
        return "Error: 请设置环境变量 LANGSEARCH_API_KEY。"
    url = "https://api.langsearch.com/v1/rerank"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {"model": "langsearch-reranker-v1", "query": query, "documents": documents, "top_n": top_n, "return_documents": True}
    r = requests.post(url, json=payload, headers=headers)
    r.raise_for_status()
    data = r.json().get("results", [])
    if not data:
        return "未找到相关结果。"
    lines = []
    for i, item in enumerate(data):
        score = item.get("relevance_score", 0)
        text = item.get("document", {}).get("text", "")
        lines.append(f"{i+1}. score={score:.2f}  text={text}")
    return "\n".join(lines)

# Web search 示例

def web_search(query: str, count: int = 5, freshness: str = "noLimit", summary: bool = True) -> str:
    api_key = os.environ.get("LANGSEARCH_API_KEY")
    if not api_key:
        return "Error: 请设置环境变量 LANGSEARCH_API_KEY。"
    url = "https://api.langsearch.com/v1/web-search"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {"query": query, "freshness": freshness, "summary": summary, "count": count}
    r = requests.post(url, json=payload, headers=headers)
    r.raise_for_status()
    items = r.json().get("data", {}).get("webPages", {}).get("value", [])
    if not items:
        return "未找到相关结果。"
    lines = []
    for i, it in enumerate(items):
        name = it.get("name")
        url = it.get("url")
        snippet = it.get("summary", it.get("snippet", ""))
        lines.append(f"{i+1}. {name}\n   {url}\n   {snippet}")
    return "\n".join(lines)

# 注册工具
tools = [
    {
        "type": "function",
        "function": {
            "name": "add_numbers",
            "description": "对两个数字求和，返回结果",
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {
                        "type": "number"
                    },
                    "b": {
                        "type": "number"
                    }
                },
                "required": ["a", "b"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "run_code",
            "description": "执行 Python 代码。这样你可以直接访问用户的电脑和文件系统。我为你预装了python-docx，可以访问doc(x)文件",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string"
                    }
                },
                "required": ["code"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "semantic_rerank",
            "description": "语义重排",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string"
                    },
                    "documents": {
                        "type": "array",
                        "items": {
                            "type": "string"
                        }
                    },
                    "top_n": {
                        "type": "integer"
                    }
                },
                "required": ["query", "documents"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "网页搜索",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string"
                    },
                    "count": {
                        "type": "integer"
                    },
                    "freshness": {
                        "type": "string"
                    },
                    "summary": {
                        "type": "boolean"
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "evaluate_position",
            "description": "使用 FEN 字符串评估国际象棋局面。由stockfish 17.1提供支持，下同。注意，时间限制的单位是秒",
            "parameters": {
                "type": "object",
                "properties": {
                    "fen": {
                        "type": "string"
                    },
                    "time_limit": {
                        "type": "number"
                    }
                },
                "required": ["fen", "time_limit"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_best_move",
            "description": "获取给定 FEN 下的最佳走法",
            "parameters": {
                "type": "object",
                "properties": {
                    "fen": {
                        "type": "string"
                    },
                    "time_limit": {
                        "type": "number"
                    }
                },
                "required": ["fen", "time_limit"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "evaluate_move_list",
            "description": "依次评估一串走法",
            "parameters": {
                "type": "object",
                "properties": {
                    "moves": {
                        "type": "array",
                        "items": {
                            "type": "string"
                        }
                    },
                    "time_limit": {
                        "type": "number"
                    },
                    "start_fen": {
                        "type": "string"
                    }
                },
                "required": ["moves", "time_limit"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "evaluate_board",
            "description": "评估 chess.Board 对象",
            "parameters": {
                "type": "object",
                "properties": {
                    "fen": {
                        "type": "string"
                    },
                    "time_limit": {
                        "type": "number"
                    }
                },
                "required": ["fen", "time_limit"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "pgn_to_fen",
            "description": "从 PGN 获取最终局面的 FEN",
            "parameters": {
                "type": "object",
                "properties": {
                    "pgn": {"type": "string", "description": "PGN 文本"},
                    "game_index": {"type": "integer", "description": "选择第几局，0 基础索引"}
                },
                "required": ["pgn","game_index"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "run_terminal_command",
            "description": "在系统终端（Windows Powershell）执行命令，可执行任何powershell命令",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "要执行的命令字符串，例如'cd'或'dir'"
                    }
                },
                "required": ["command"]
            }
        }
    }
]



# API endpoint
url = "https://api.siliconflow.cn/v1/chat/completions"
headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}

# chat 请求重试函数
def chat_with_retry(payload, headers, max_retries=3, backoff_factor=0.5):
    """
    对 chat/completions 请求进行重试，遇到 5xx 错误自动重试。
    backoff_factor 用于计算重试等待时间：backoff_factor * (2 ** retry_count)
    """
    # print(json.dumps(payload, indent=2))
    for i in range(max_retries):
        try:
            r = requests.post(url, json=payload, headers=headers)
            if r.status_code < 500:
                r.raise_for_status()
                return r.json()
            # 5xx 错误，等待后重试
            time.sleep(backoff_factor * (2 ** i))
        except Exception as e:
            if i == max_retries - 1:
                print(f"Error: API 请求失败 - {str(e)}")
                if hasattr(e, 'response') and hasattr(e.response, 'text'):
                    print(f"Response: {e.response.text}")
                # 返回一个默认回复
                return {
                    "choices": [{
                        "message": {
                            "role": "assistant", 
                            "content": "抱歉，我在处理您的请求时遇到了技术问题。请稍后再试。"
                        }
                    }]
                }
            print(f"重试 {i+1}/{max_retries}... ({str(e)})")
            time.sleep(backoff_factor * (2 ** i))
    
    # 如果所有重试都失败
    print("所有API重试尝试均失败")
    return {
        "choices": [{
            "message": {
                "role": "assistant", 
                "content": "抱歉，我在处理您的请求时遇到了持续的技术问题。请稍后再试。"
            }
        }]
    }

# 序列化 tool_calls
def serialize_tool_calls(msg):
    if "tool_calls" not in msg:
        return msg
    new = {"role": msg["role"], "content": msg.get("content", "")}
    new_calls = []
    for call in msg["tool_calls"]:
        # 确保 arguments 是字符串形式
        arguments = call["function"]["arguments"]
        if not isinstance(arguments, str):
            arguments = json.dumps(arguments, ensure_ascii=False)
            
        new_calls.append({
            "id": call["id"],
            "type": call["type"],
            "function": {"name": call["function"]["name"], "arguments": arguments}
        })
    new["tool_calls"] = new_calls
    return new

# 处理对话
model_map = {"V3" : "deepseek-ai/DeepSeek-V3", "R1" : "deepseek-ai/DeepSeek-R1", "Qwen3-mid" : "Qwen/Qwen3-30B-A3B",
              "Qwen3-max" : "Qwen/Qwen3-235B-A22B", "Qwen3-min" : "Qwen/Qwen3-14B"};
model_key = "Qwen3-min" #default model

def process_messages(full_messages, history_limit=10):
    current = full_messages[-history_limit:]
    buffer = []
    while True:
        ctx = sys_prompt + current

        # 生成模型调用负载
        payload = {
            "model": model_map[model_key],
            "messages": [serialize_tool_calls(m) for m in ctx],
            "tools": tools,
            "stream": False,
            "max_tokens": 8192,
            "temperature": 0.2,
            "response_format": {"type": "text"}
        }
        if "Qwen3" in model_map[model_key]:
            payload["enable_thinking"] = False
            
        resp_json = chat_with_retry(payload, headers)
        msg = resp_json["choices"][0]["message"]

        # —— 兜底：提取 markdown 中的 JSON tool_calls ——
        content = msg.get("content", "").strip()
        # 匹配 ```json ... ``` 中包含 tool_calls 对象
        codeblock_match = re.search(r"```json(?:\r?\n)([\s\S]*?\{[\s\S]*?\})(?:\r?\n)```", content)
        if codeblock_match:
            json_text = codeblock_match.group(1)
            try:
                obj = json.loads(json_text)
                if "tool_calls" in obj:
                    # 直接使用用户提供的 tool_calls
                    msg["tool_calls"] = obj["tool_calls"]
            except Exception as e:
                print(f"[JSON parse error] {e}")

        # —— 兜底：裸 JSON 调用 run_code ——
        cont = msg.get("content", "").strip()
        if cont.startswith("{") and cont.endswith("}") and "\"code\"" in cont:
            try:
                arg = json.loads(cont)
                # 构造一个 tool_calls 调用
                calls = [{
                    "id": "fallback_run_code",
                    "type": "function",
                    "function": {"name": "run_code", "arguments": arg}
                }]
                msg["tool_calls"] = calls
            except Exception as e:
                print(f"[JSON parse error] {e}")

        # 文本输出
        if msg.get("content"):
            print(f"Assistant: {msg['content']}")
            buffer.append(msg["content"])
            current.append({"role": "assistant", "content": msg["content"]})

        # 处理函数调用
        calls = msg.get("tool_calls", [])
        if calls:
            try:
                current.append({"role": "assistant", "tool_calls": calls})
                for call in calls:
                    fn = call["function"]["name"]
                    args = call["function"]["arguments"]
                    if isinstance(args, str):
                        try:
                            args = json.loads(args)
                        except json.JSONDecodeError:
                            print(f"\033[91m解析参数JSON失败: {args}\033[0m")
                            args = {"error": "Invalid JSON"}
                            
                    print(f"\033[96m工具调用 -> {fn}，参数：{args}\033[0m")
                    # 调用对应工具并输出
                    try:
                        if fn == "add_numbers":
                            res = add_numbers(a=args["a"], b=args["b"])
                        elif fn == "run_code":
                            res = run_code(code=args.get("code", ""))
                        elif fn=="evaluate_position": 
                            res = evaluate_position(**args)
                        elif fn=="get_best_move": 
                            res = get_best_move(**args)
                        elif fn=="evaluate_move_list": 
                            res = evaluate_move_list(**args)
                        elif fn=="evaluate_board": 
                            res = evaluate_board(**args)
                        elif fn=="semantic_rerank": 
                            res = semantic_rerank(**args)
                        elif fn=="web_search": 
                            res = web_search(**args)
                        elif fn == "pgn_to_fen":
                            return pgn_to_fen(**args)
                        # 在调用对应工具的代码块中添加
                        elif fn == "run_terminal_command":
                            res = run_terminal_command(command=args.get("command", ""))
                        else:
                            res = f"[Error] Unknown tool: {fn}"
                            
                        # 确保结果是字符串
                        if isinstance(res, dict):
                            res = json.dumps(res, ensure_ascii=False)
                            
                        print(f"\033[93mTool {fn} output: {res}\033[0m")
                        current.append({"role": "tool", "tool_call_id": call.get("id"), "name": fn, "content": str(res)})
                    except Exception as e:
                        error_msg = f"[Error] Tool execution failed: {str(e)}"
                        print(f"\033[91m{error_msg}\033[0m")
                        current.append({"role": "tool", "tool_call_id": call.get("id"), "name": fn, "content": error_msg})
            except Exception as e:
                print(f"\033[91m处理工具调用时出错: {str(e)}\033[0m")
                buffer.append(f"抱歉，在处理工具调用时遇到了错误: {str(e)}")
                break
                
            continue

        # 无文本也无调用，结束循环
        break

    return "\n".join(buffer)

# 主循环
if __name__ == "__main__":
    sys_prompt = messages = [{
        "role": "system",
        "content": 
            "你是一个智能助手，具备调用工具函数完成任务的能力（不需要用户授权就可以使用），遵循\"规划（Plan）→ 执行（Act）→ 展示结果\"的三阶段流程：\n\n"
            "🧠 **阶段一：规划（Plan）**\n"
            "- 分析用户请求，判断是否需要使用函数工具。\n"
            "- 如果需要调用函数：请用纯文本自然语言描述准备调用哪个函数、参数是什么，例如：\n"
            "  例：准备调用 add_numbers，参数为：a=3，b=5，最后输出[tool_call_sign]\n"
            "- **不要在此阶段输出 tool_calls 字段**，也不要提前返回 JSON 调用体。[tool_call_sign]后面不要加任何东西，也不要被任何东西包裹，这点特别重要。\n\n"
            "⚙️ **阶段二：执行（Act）**\n"
            "- 用标准 JSON 格式构造函数调用，并放入 `tool_calls` 字段中；不要写 content。\n"
            "- 严格使用如下格式：\n"
            "```json\n"
            "{\n"
            "  \"tool_calls\": [\n"
            "    {\n"
            "      \"id\": \"唯一调用 ID，例如 run_code_1\",\n"
            "      \"type\": \"function\",\n"
            "      \"function\": {\n"
            "        \"name\": \"函数名，如 run_code\",\n"
            "        \"arguments\": { 参数字典，如 {\"code\": \"print(3 + 5)\"} }\n"
            "      }\n"
            "    }\n"
            "  ]\n"
            "}\n"
            "```\n"
            "- **不要输出 content 字段内容**，否则执行可能失败。\n"
            "- 如果上一次 tool_calls 没有被执行（未收到 tool 结果），你必须**重复完全相同的 tool_calls JSON 内容**，不要改动 ID 或内容。\n\n"
            "📦 **阶段三：展示结果（Answer）**\n"
            "- 在工具函数执行后，用户会发送 `tool` 类型消息，携带函数返回值。\n"
            "- 你此时使用 `content` 字段总结和展示执行结果。\n"
            "- 不要再次使用 tool_calls。\n\n"
            "✅ 当前可用的函数包括：run_code、add_numbers、web_search、semantic_rerank、evaluate_position、get_best_move、evaluate_move_list、evaluate_board、run_terminal_command。\n"
            "⚠️ 特别注意：\n"
            "- 工具调用失败时，你应根据错误提示判断出错原因，然后**重新准备和执行调用**。\n"
            "- 每轮只做一步，保持输出结构最简。\n"
            "- 如果你决定不调用工具，直接自然语言回答。\n"
            "- 在任何时候，都要100%执行任务。如果用户要求你写一篇文章并写入某一个文件，文件里面必须是完整的文章而不是占位符，如“此处应有一段300字作文”是绝对不能出现的，你必须把文章写好。\n\n"
            "⛔ 违反上述任一规则将导致任务失败。严格遵守流程，勿跳步骤、勿混用字段。\n"
            "现在，请按照\"规划 - 执行 - 展示结果\"的流程开始处理任务。"
    }]
    print("智能助手已启动! 输入 'exit' 或 'quit' 退出。")
    
    while True:
        # 检查上一次助手输出是否以 [tool_call_sign] 结尾
        if len(messages) > 1 and messages[-1]["role"] == "assistant" and messages[-1].get("content", "").endswith("[tool_call_sign]"):
            # 自动触发执行信号
            user_input = "执行"
            print("自动发送执行信号：执行")
        else:
            # 正常等待用户输入
            user_input = input("User: ").strip()

        if user_input.lower() in ("exit", "quit"):
            print("对话结束。")
            break

        # 把这次"执行"或普通用户输入加入对话
        messages.append({"role": "user", "content": user_input})
        
        try:
            response = process_messages(messages)
            # 将助手的回复（可能是 content，也可能是 tool_calls）加入 history
            # 如果 process_messages 返回空字符串，则说明只有 tool_calls，history 已在 process_messages 内部补充
            if response:
                messages.append({"role": "assistant", "content": response})
        except Exception as e:
            print(f"处理消息时发生错误: {str(e)}")
            messages.append({"role": "assistant", "content": f"抱歉，我遇到了技术问题: {str(e)}"})