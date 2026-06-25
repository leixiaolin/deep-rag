import json
from typing import List, Dict
from datetime import datetime
from backend.knowledge_base import knowledge_base


def _create_base_system_prompt(root_summary: str) -> str:
    """Shared base system prompt"""
    current_time = datetime.now().strftime("%A, %B %d, %Y, at %I:%M:%S %p")
    
    return f"""
- Answers must strictly come from the knowledge base
- To answer completely, you may call the `retrieve_files` tool multiple times
- If you're 100% certain of the answer, you may skip calling the `retrieve_files` tool
- If after diligent multi-round retrieval you still haven't found relevant knowledge, please answer "I don't know"
- Current time: {current_time}

## Knowledge Base Overview
The overview below shows top-level categories only. Use `retrieve_summary` to explore deeper (files, chunks) before retrieving full content.
```
{root_summary}
```

## retrieve_summary
- Browse the knowledge base summary hierarchy. Returns summaries of child entries one level deeper.
- Use this FIRST to identify which files are relevant before retrieving their full content.
- Input format: {{"path": "category-name", "depth": 1}}
- path: "." for root, a category name, or a file path to see chunk-level summaries.

### Examples
- Explore root: {{"path": "."}}
- Explore a category: {{"path": "Product-Line-A-Smartwatch-Series"}}
- See chunk-level summaries: {{"path": "Product-Line-A-Smartwatch-Series/SW-2100-Flagship.md"}}

## retrieve_files
- Retrieve full file contents after identifying relevant files via `retrieve_summary`.
- When Milvus hybrid retrieval is enabled, include a concise English `query` so the backend can run sparse BM25 + dense vector retrieval in one Milvus hybrid search.
- NEVER answer "I don't know" without calling tools.
- Input format: {{"file_paths": ["path1", "path2"], "query": "english retrieval query", "top_k": 8}}
- `file_paths` may be empty to search the whole knowledge base. Directory paths constrain retrieval to that directory.

### Examples
- Retrieve specific files: {{"file_paths": ["Product-Line-A-Smartwatch-Series/SW-1500-Sport.md"], "query": "SW-1500 sport specifications battery IP68 TFT price", "top_k": 5}}
- Retrieve multiple directories: {{"file_paths": ["2024-Market-Layout/", "2023-Market-Layout/"], "query": "annual revenue retail stores by region", "top_k": 8}}
- Retrieve all files: {{"file_paths": [], "query": "display types besides AMOLED and OLED", "top_k": 8}}

""".strip()


def create_system_prompt(file_summary: str) -> str:
    """System prompt for function calling mode"""
    return _create_base_system_prompt(file_summary)


def create_react_system_prompt(file_summary: str) -> str:
    """System prompt for ReAct mode with format instructions"""
    base_prompt = _create_base_system_prompt(file_summary)
    
    return f"""
{base_prompt}

## Direct Answer
- The overview has the answer

### Example
- Question: Besides AMOLED and OLED screens, what other display types do we have?
- Answer: LCD, TFT

## Tool Call
- The overview doesn't have enough details

### Pattern
- <|Thought|> Think about what information you need to answer the question
- <|Action|> Tool
- <|Action Input|> Input format
- <|Observation|> [The system will provide file contents here]
- ... (repeat Thought/Action/Observation as needed)
- <|Final Answer|> [Your final answer based on the retrieved information]

### Example
- Question: What are all the technical specifications of SW-2100?
- <|Thought|> I need to locate the SW-2100 file. Let me browse the smartwatch category first
- <|Action|> retrieve_summary
- <|Action Input|> {{"path": "Product-Line-A-Smartwatch-Series"}}
- <|Observation|> [System provides summaries of smartwatch files]
- <|Thought|> Now I'll retrieve the full SW-2100 file for complete specifications
- <|Action|> retrieve_files
- <|Action Input|> {{"file_paths": ["Product-Line-A-Smartwatch-Series/SW-2100-Flagship.md"], "query": "SW-2100 technical specifications display battery sensors price", "top_k": 5}}
- <|Observation|> [System provides file content]
- <|Final Answer|> [Complete specifications based on retrieved file]

""".strip()
    

def create_file_retrieval_tool() -> Dict:
    return {
        "type": "function",
        "function": {
            "name": "retrieve_files",
            "description": (
                "Retrieve knowledge base content. In Milvus hybrid mode, this runs sparse "
                "BM25 + dense vector retrieval using the query and optional path constraints."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "file_paths": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "Optional list of file paths or directory paths to retrieve or constrain. "
                            "Use [] or '/' to search all files."
                        )
                    },
                    "query": {
                        "type": "string",
                        "description": (
                            "Concise English retrieval query for hybrid sparse+dense search. "
                            "Preserve key numbers, model names, regions, years, and product terms."
                        )
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Number of chunks to retrieve. Default 8; backend clamps to a safe range.",
                        "minimum": 1,
                        "maximum": 30
                    }
                },
                "required": []
            }
        }
    }


def create_summary_browsing_tool() -> Dict:
    return {
        "type": "function",
        "function": {
            "name": "retrieve_summary",
            "description": (
                "Browse the knowledge base summary hierarchy. "
                "Returns summaries of child entries (directories, files, or chunks) "
                "one level deeper than the specified path. "
                "Use this to identify relevant files before retrieving their full content."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": (
                            "Path to explore. Use '.' for root level, "
                            "a category name like 'Product-Line-A-Smartwatch-Series', "
                            "or a file path to see chunk-level summaries."
                        ),
                        "default": "."
                    },
                    "depth": {
                        "type": "integer",
                        "description": "How many levels to expand (default: 1)",
                        "default": 1
                    }
                },
                "required": []
            }
        }
    }


async def process_tool_calls(tool_calls: List[Dict], user_query: str = None) -> List[Dict]:
    results = []

    for tool_call in tool_calls:
        func_name = tool_call.get("function", {}).get("name")
        try:
            args = json.loads(tool_call["function"]["arguments"])

            if func_name == "retrieve_files":
                content = await knowledge_base.retrieve_files(
                    args.get("file_paths", []),
                    query=args.get("query") or user_query,
                    top_k=args.get("top_k"),
                )
            elif func_name == "retrieve_summary":
                content = await knowledge_base.retrieve_summary(
                    args.get("path", "."), args.get("depth", 1)
                )
            else:
                content = f"Unknown tool: {func_name}"

            results.append({
                "role": "tool",
                "tool_call_id": tool_call.get("id"),
                "content": content
            })
        except Exception as e:
            results.append({
                "role": "tool",
                "tool_call_id": tool_call.get("id"),
                "content": f"Error executing tool {func_name}: {str(e)}"
            })

    return results


def parse_react_response(text: str) -> tuple:
    """Parse ReAct-style response to extract action and input"""
    import re

    # 查找 <|Action|> 和 <|Action Input|> (新格式)
    action_pattern = r'<\|Action\|>\s*(\w+)'
    action_input_pattern = r'<\|Action Input\|>\s*(\{[\s\S]*?\})(?=\s*(?:<\||$))'

    action_match = re.search(action_pattern, text)
    action_input_match = re.search(action_input_pattern, text)

    if action_match and action_input_match:
        action = action_match.group(1)
        try:
            action_input = json.loads(action_input_match.group(1))
            return action, action_input, True
        except:
            pass

    return None, None, False


async def process_react_response(text: str, user_query: str = None) -> tuple:
    """Process ReAct response and execute actions"""
    action, action_input, has_action = parse_react_response(text)

    if not has_action:
        return None, False

    if action == "retrieve_files":
        content = await knowledge_base.retrieve_files(
            action_input.get("file_paths", []),
            query=action_input.get("query") or user_query,
            top_k=action_input.get("top_k"),
        )
    elif action == "retrieve_summary":
        content = await knowledge_base.retrieve_summary(
            action_input.get("path", "."), action_input.get("depth", 1)
        )
    else:
        return None, False

    return {
        "action": action,
        "input": action_input,
        "observation": content
    }, True
