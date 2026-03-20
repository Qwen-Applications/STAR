import jsonlines
from tqdm import tqdm
import re
import json
import random
import ast
import sys
import os
parent_parent_dir =os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_parent_dir)
from simrl import (
    get_ground_truth_from_label,
    parse_generation,
    extract_tools_from_prompt,
    check_tool_calls_valid,
    compute_function_calling_reward,
    get_rouge_score,
    get_generation_from_query
)
from transformers import AutoTokenizer


def parse_tools_python(function_text: str):
    """
    Parse Python-style tool calls
    Supports parameter values containing special characters like parentheses, commas

    example:
    INPUT:
    [ls(a=False), sort(file_name="yzjcq.py"), echo(content="def __init__(self):")]

    OUTPUT:
    tool_calls = [
        {"name": "ls", "arguments": {"a": False}},
        {"name": "sort", "arguments": {"file_name": "yzjcq.py"}},
        {"name": "echo", "arguments": {"content": "def __init__(self):"}}
    ]
    """
    function_text = function_text.strip()
    if not (function_text.startswith("[") and function_text.endswith("]")):
        return []

    tool_calls = []
    function_text = function_text[1:-1].strip()

    i = 0
    while i < len(function_text):
        while i < len(function_text) and function_text[i] in ' ,\t\n':
            i += 1
        if i >= len(function_text):
            break

        func_name_start = i
        while i < len(function_text) and (function_text[i].isalnum() or function_text[i] in '_.'):
            i += 1
        func_name = function_text[func_name_start:i]

        while i < len(function_text) and function_text[i] in ' \t\n':
            i += 1

        if i >= len(function_text) or function_text[i] != '(':
            raise ValueError(f"Expected '(' after function name {func_name}")
        i += 1

        args_str = ''
        paren_depth = 1
        in_quote = None
        escape_next = False

        while i < len(function_text):
            char = function_text[i]

            if escape_next:
                args_str += char
                escape_next = False
            elif char == '\\':
                args_str += char
                escape_next = True
            elif in_quote:
                args_str += char
                if char == in_quote:
                    in_quote = None
            elif char in '"\'':
                in_quote = char
                args_str += char
            elif char == '(':
                paren_depth += 1
                args_str += char
            elif char == ')':
                paren_depth -= 1
                if paren_depth == 0:
                    break
                args_str += char
            else:
                args_str += char

            i += 1

        if i >= len(function_text):
            break

        i += 1

        arguments = _parse_arguments_dict(args_str.strip())

        tool_calls.append({
            "name": func_name,
            "arguments": arguments
        })

    return tool_calls


def _parse_arguments_dict(args_str: str):
    """
    Parse argument string into dictionary
    Supports keyword arguments: key=value
    """
    arguments = {}

    if not args_str.strip():
        return arguments

    args_list = []
    current_arg = ''
    paren_depth = 0
    in_quote = None
    escape_next = False

    for char in args_str:
        if escape_next:
            current_arg += char
            escape_next = False
        elif char == '\\':
            current_arg += char
            escape_next = True
        elif in_quote:
            current_arg += char
            if char == in_quote:
                in_quote = None
        elif char in '"\'':
            in_quote = char
            current_arg += char
        elif char == '(':
            paren_depth += 1
            current_arg += char
        elif char == ')':
            paren_depth -= 1
            current_arg += char
        elif char == ',' and paren_depth == 0 and not in_quote:
            args_list.append(current_arg.strip())
            current_arg = ''
        else:
            current_arg += char

    if current_arg.strip():
        args_list.append(current_arg.strip())

    for pair in args_list:
        if '=' in pair:
            key, value = pair.split('=', 1)
            key = key.strip()
            value = value.strip()

            try:
                parsed_value = ast.literal_eval(value)
                arguments[key] = parsed_value
            except (ValueError, SyntaxError):
                if (value.startswith('"') and value.endswith('"')) or \
                   (value.startswith("'") and value.endswith("'")):
                    arguments[key] = value[1:-1]
                else:
                    arguments[key] = value

    return arguments


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict = None,
    print_info: bool = True,
    **kwargs
) -> dict:
    """
    Function Call training reward function

    Args:
        data_source: Data source identifier (should be "function_call_training")
        solution_str: Model generated text
        extra_info: Extra information (must contain prompt_text field)
        print_info: Whether to print detailed information

    Returns:
        {
            "score": float,
            "format_score": float,
            "answer_score": float
        }
    """
    if extra_info is None:
        extra_info = {}

    prompt_text = extra_info.get('prompt_text', '')

    # Construct full query
    query = prompt_text + solution_str
    prompt = prompt_text
    label = ground_truth

    try:
        gt_think, gt_tool_call_objs, gt_reply = get_ground_truth_from_label(label)
    except Exception as e:
        if print_info:
            print(f"[ERROR] Invalid ground_truth format: {e}")
        return {
            "score": -1.0,
            "format_score": -1.0,
            "answer_score": 0.0
        }

    if print_info:
        print(f"\n{'='*80}")
        print(f"[Function Call Reward] data_source={data_source}")
        print(f"[Prompt]\n{prompt}")
        print(f"[Response]\n{solution_str}")
        print(f"[Ground Truth]\n{label}")

    try:
        tool_dict = extract_tools_from_prompt(prompt)
    except Exception as e:
        if print_info:
            import traceback
            traceback.print_exc()
            print(f"[ERROR] Invalid prompt format: {e}")
        return {
            "score": -1.0,
            "format_score": -1.0,
            "answer_score": 0.0
        }

    try:
        # Extract generation from query
        generation = get_generation_from_query(query)
        p_think, p_tool_call_objs, p_reply = parse_generation(generation)
    except Exception as e:
        if print_info:
            print(f"[ERROR] Answer format error: {e}")
            print(f"[Format Score] -1.0")
            print(f"[Answer Score] 0.0")
            print(f"[Final Score] -1.0")
            print(f"{'='*80}\n")
        return {
            "score": -1.0,
            "format_score": -1.0,
            "answer_score": 0.0
        }

    if print_info:
        print(f"[Model think] {p_think}")
        print(f"[Model tool_calls] {p_tool_call_objs}")
        print(f"[Model reply] {p_reply}")

    try:
        result = check_tool_calls_valid(p_tool_call_objs, tool_dict)
        if not result:
            if print_info:
                print(f"[ERROR] Tool calls are invalid")
                print(f"[Format Score] -1.0")
                print(f"[Answer Score] 0.0")
                print(f"[Final Score] -1.0")
                print(f"{'='*80}\n")
            return {
                "score": -1.0,
                "format_score": -1.0,
                "answer_score": 0.0
            }
    except Exception as e:
        if print_info:
            print(f"[ERROR] Tool calls validation failed: {e}")
            print(f"[Format Score] -1.0")
            print(f"[Answer Score] 0.0")
            print(f"[Final Score] -1.0")
            print(f"{'='*80}\n")
        return {
            "score": -1.0,
            "format_score": -1.0,
            "answer_score": 0.0
        }

    format_score = 1.0

    # Calculate answer score
    if len(gt_tool_call_objs) > 0:
        # Ground truth is function calling case
        if len(p_tool_call_objs) > 0:
            answer_score = compute_function_calling_reward(gt_tool_call_objs, p_tool_call_objs)
        else:
            answer_score = 0.0
    else:
        # Ground truth is text reply case
        if p_reply:
            answer_score = get_rouge_score(p_reply, gt_reply)
        else:
            answer_score = 0.0

    final_score = answer_score

    if print_info:
        print(f"[Format Score] {format_score}")
        print(f"[Answer Score] {answer_score}")
        print(f"[Final Score] {final_score}")
        print(f"{'='*80}\n")

    return {
        "score": final_score,
        "format_score": format_score,
        "answer_score": answer_score
    }


# ============== ToolACE Helper Functions ==============

def find_closest_tool_name(tool_name, tool_names):
    """Fuzzy matching for tool names"""
    for item in tool_names:
        if tool_name in item:
            return item
    raise ValueError(f"Tool {tool_name} not found in {tool_names}")


def format_expected_output(tool_calls, tools):
    """Tool name validation and correction"""
    tool_names = [tool['name'] for tool in tools]

    for tool_call in tool_calls:
        if tool_call['name'] not in tool_names:
            # Find the most similar tool_name and replace
            tool_name = find_closest_tool_name(tool_call['name'], tool_names)
            tool_call['name'] = tool_name
    return tool_calls


def make_sure_args_valid(tool_calls, tools):
    """Argument validation"""
    tools_dict = {tool['name']: tool for tool in tools}
    for tool_call in tool_calls:
        tool_name = tool_call['name']
        tool = tools_dict[tool_name]
        args = tool_call['arguments']
        if not isinstance(args, dict):
            args = json.loads(args)
            tool_call['arguments'] = args
        for arg in args:
            if arg not in tool['parameters']:
                raise ValueError(f"Argument {arg} not found in {tool_name}")

# ============== ToolACE Helper Functions End ==============

def openai_func_decode(tool):
    if "type" in tool and "function" in tool:
        return openai_func_decode(tool['function'])
    else:
        return tool

def format_tool(tool):

    tool_obj = openai_func_decode(tool)
    
    if "name" not in tool_obj:
        raise ValueError("Invalid tool: missing name field")
    if "parameters" not in tool_obj:
        if "arguments" in tool_obj:
            tool_obj['parameters'] = tool_obj.pop('arguments')
        else:
            raise ValueError("Invalid tool: missing parameters field")

    return tool_obj

def format_toolmind(data_list):

    def format_assistant_message(message):
        if "tool_calls" in message:
            new_tool_calls = []
            tool_calls = message['tool_calls']
            for tc in tool_calls:
                if "function" in tc:
                    tc = tc['function']
                new_tool_calls.append(tc)
        else:
            new_tool_calls = None
            
        if "content" in message:
            if "<think>" in message['content']:
                whole_content = message['content']
                if whole_content.count("<think>") != 1 and whole_content.count("</think>") != 1:
                    raise ValueError("Invalid reasoning content")
                try:
                    reasoning_content = re.search(r"<think>(.*?)</think>", whole_content, re.S).group(1)
                except:
                    raise ValueError("Invalid reasoning content")
                content = whole_content.split("</think>")[-1]
            else:
                reasoning_content = None
                content = message['content']
        else:
            reasoning_content = None
            content = None
        new_message = {
            "role": role,
            "reasoning_content": reasoning_content,
            "tool_calls": new_tool_calls,
            "content": content
        }

        return new_message

    all_data = []
    for i, data in enumerate(tqdm(data_list)):
        messages = data.pop("conversations")
        data["id"] = f"toolmind_{i}"

        new_messages = []
        try:
            for message in messages:
                role = message['role']
                if role == "assistant":
                    new_message = format_assistant_message(message)
                                    
                elif role in ["user", "tool"]:
                    new_message = message
                else:
                    raise NotImplementedError
                new_messages.append(new_message)
        except ValueError as e:
            print(f"{data['id']} error: {e}")
            continue
            
        
        data["messages"] = new_messages
        tools = data["tools"]
        new_tools = []
        for tool_obj in tools:
            tool_obj = format_tool(tool_obj)
            new_tools.append(tool_obj)

        data['tools'] = new_tools
        all_data.append(data)

    return all_data


def filter_on_simrl(data_list):
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-14B/")
    filterd_data_list = []
    for data in tqdm(data_list):
        messages = data['messages']
        tools = data['tools']
        prompt_text = tokenizer.apply_chat_template(
            messages[:-1],
            tokenize=False,
            add_generation_prompt=True,
            tools=tools
        )

        solution_str = tokenizer.apply_chat_template(
            messages[-1:],
            tokenize=False,
        ).replace("<|im_start|>assistant", '')

        if "</think>"  not in solution_str:
            solution_str = "<think></think>"  + solution_str
        results = compute_score(
            data_source="function_call_training",
            solution_str=solution_str,
            ground_truth=solution_str,
            extra_info={"prompt_text": prompt_text},
            tokenizer=tokenizer,
            print_info=False
        )

        
        score = results['score']

        if score > 0.99:
            filterd_data_list.append(data)

        else:
            data_id = data.get('id', data.get('trace_id', 'unknown'))
            print(f"{data_id} score: {score}")

    return filterd_data_list


def format_toolace(input_path=None, dataset_size=None):
    """
    Format ToolACE dataset
    Implements the full logic of prepare_toolace in toolace.py
    """
    from datasets import load_dataset
    # Please use HF_ENDPOINT=https://hf-mirror.com
    if input_path:
        with jsonlines.open(input_path) as reader:
            data_list = list(reader)
    else:
        dataset = load_dataset("tryumanshow/ToolACE-Qwen-cleaned")
        data_list = dataset['train']  # Get training set

    # Random sampling (before processing)
    if dataset_size is not None and dataset_size < len(data_list):
        print(f"Randomly sampling {dataset_size} data points (original: {len(data_list)})")
        data_list = random.sample(list(data_list), dataset_size)

    all_data = []
    for i, data in enumerate(tqdm(data_list)):
        try:
            # Parse tools (JSON string)
            tools = []
            if isinstance(data['tools'], str):
                tools = json.loads(data['tools'])
            else:
                tools = data['tools']

            # Parse conversations (JSON string)
            conversations = []
            if isinstance(data['conversations'], str):
                conversations = json.loads(data['conversations'])
            else:
                conversations = data['conversations']

            new_messages = []
            parallel_tool_calls = []

            for j, message in enumerate(conversations):
                if message['role'] == "user" and message.get('content'):
                    new_messages.append({
                        "role": "user",
                        "content": message['content']
                    })

                elif message['role'] == "tool":
                    content = message['content']
                    if not isinstance(content, str):
                        content = json.dumps(content, ensure_ascii=False)
                    
                    new_messages.append({
                        "role": "tool",
                        "content": content
                    })

                elif message['role'] == "assistant":
                    if message.get('content'):
                        # Normal text response
                        expected_output = message['content']

                        # Build complete messages including assistant message
                        messages_with_assistant = new_messages.copy()
                        messages_with_assistant.append({
                            "role": "assistant",
                            "content": expected_output
                        })

                        new_data = {
                            "id": f"toolace-{i}-{j}",
                            "messages": messages_with_assistant,
                            "tools": tools
                        }
                        all_data.append(new_data)

                        new_messages.append({
                            "role": "assistant",
                            "content": expected_output
                        })

                    elif message.get('tool_calls'):
                        """
                        Tool call processing
                        Format: [{'function': {'name': 'Market Trends API', 'arguments': '{"trend_type": "MARKET_INDEXES", "country": "us"}'}}]
                        """
                        tool_calls = message['tool_calls']
                        expected_output = [item['function'] for item in tool_calls]

                        # Ensure all tool_calls' tool_name exist in tools
                        try:
                            expected_output = format_expected_output(expected_output, tools)
                        except ValueError as e:
                            print(f"toolace-{i}-{j} Error: {e}")
                            continue

                        # Validate argument validity
                        try:
                            make_sure_args_valid(expected_output, tools)
                        except ValueError as e:
                            print(f"toolace-{i}-{j} Error: {e}")
                            continue

                        # Accumulate parallel tool calls
                        parallel_tool_calls.extend(expected_output)

                        # If it's the last message or the next message has no tool_calls, output the sample
                        if j == len(conversations) - 1 or not conversations[j+1].get("tool_calls"):
                            # Build complete messages including assistant message
                            messages_with_assistant = new_messages.copy()
                            messages_with_assistant.append({
                                "role": "assistant",
                                "tool_calls": parallel_tool_calls
                            })

                            new_data = {
                                "id": f"toolace-{i}-{j}",
                                "messages": messages_with_assistant,
                                "tools": tools
                            }
                            all_data.append(new_data)

                            new_messages.append({
                                "role": "assistant",
                                "tool_calls": parallel_tool_calls
                            })
                            parallel_tool_calls = []
                    else:
                        print(f"toolace-{i}-{j} Unexpected message: {message}")
                else:
                    print(f"toolace-{i}-{j} Unexpected role: {message['role']}")

        except Exception as e:
            print(f"toolace-{i} error: {e}")
            import traceback
            traceback.print_exc()
            continue

    return all_data



def format_xlam(input_path=None, dataset_size=None):
    from datasets import load_dataset
    # Please use HF_ENDPOINT=https://hf-mirror.com
    if input_path:
        with jsonlines.open(input_path) as reader:
            data_list = list(reader)
    else:
        dataset = load_dataset("Salesforce/xlam-function-calling-60k")
        data_list = dataset['train']  # Get training set

    # Random sampling (before processing)
    if dataset_size is not None and dataset_size < len(data_list):
        print(f"Randomly sampling {dataset_size} data points (original: {len(data_list)})")
        data_list = random.sample(list(data_list), dataset_size)

    all_data = []
    for data in tqdm(data_list):
        data_id = f"xlam_{data['id']}"

        try:
            # Parse tools (JSON string)
            tools = []
            if isinstance(data['tools'], str):
                tools = json.loads(data['tools'])
            else:
                tools = data['tools']

            # Format tools
            new_tools = []
            for tool_obj in tools:
                new_tool = {
                    'name': tool_obj['name'],
                    'parameters': tool_obj.get('parameters', {}),
                    'description': tool_obj.get('description', '')
                }
                new_tools.append(new_tool)

            # Parse answers (JSON string)
            answers = []
            if isinstance(data['answers'], str):
                answers = json.loads(data['answers'])
            else:
                answers = data['answers']

            # Build messages
            new_messages = [
                {
                    'role': 'user',
                    'content': data['query']
                }
            ]

            # Build assistant message
            assistant_message = {
                'role': 'assistant',
                'content': None,
                'tool_calls': answers if answers else None
            }
            new_messages.append(assistant_message)

            new_data = {
                'id': data_id,
                'messages': new_messages,
                'tools': new_tools
            }
            all_data.append(new_data)

        except Exception as e:
            print(f"{data_id} error: {e}")
            import traceback
            traceback.print_exc()
            continue

    return all_data


def format_hammer(input_path, dataset_size=None):
    """
    Format HAMMER dataset
    HAMMER is an "irrelevant query" dataset where user requests don't match the provided tools
    """
    if not input_path:
        raise ValueError("Please provide input path for hammer dataset")

    with jsonlines.open(input_path) as reader:
        data_list = list(reader)

    # Random sampling (before processing)
    if dataset_size is not None and dataset_size < len(data_list):
        print(f"Randomly sampling {dataset_size} data points (original: {len(data_list)})")
        data_list = random.sample(data_list, dataset_size)

    all_data = []
    for data in tqdm(data_list):
        data_id = data.get('trace_id', f"hammer_{len(all_data)}")

        try:
            # Parse functions (HAMMER format)
            # HAMMER's parameters format: {param_name: {description, type}}
            # Need to convert to target format: {name, parameters: {param_name: {description, type}}, description}
            new_tools = []
            for func_obj in data.get('functions', []):
                new_tool = {
                    'name': func_obj['name'],
                    'parameters': func_obj.get('parameters', {}),
                    'description': func_obj.get('description', '')
                }
                new_tools.append(new_tool)

            # Build messages
            new_messages = []

            # Add user message
            for msg in data.get('messages', []):
                new_messages.append({
                    'role': msg['role'],
                    'content': msg.get('content')
                })

            # Add assistant message (expected_output as content, no tool_calls)
            assistant_message = {
                'role': 'assistant',
                'content': data.get('expected_output', ''),
                'tool_calls': None
            }
            new_messages.append(assistant_message)

            new_data = {
                'id': data_id,
                'messages': new_messages,
                'tools': new_tools
            }
            all_data.append(new_data)

        except Exception as e:
            print(f"{data_id} error: {e}")
            import traceback
            traceback.print_exc()
            continue

    return all_data


def get_args():

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, default="")
    parser.add_argument("--output_path", type=str, default="")
    parser.add_argument("--dataset_name", type=str, default="toolmind", choices=["toolmind", "toolace", "xlam", "hammer"])
    parser.add_argument("--dataset_size", type=int, default=None, help="Number of data points to randomly sample")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = get_args()

    # Set random seed
    random.seed(args.seed)

    if args.dataset_name == "toolmind":
        if not args.input_path:
            raise ValueError("Please provide input path")
        with jsonlines.open(args.input_path) as reader:
            data_list = list(reader)

        # Random sampling (before processing)
        if args.dataset_size is not None and args.dataset_size < len(data_list):
            print(f"Randomly sampling {args.dataset_size} data points (original: {len(data_list)})")
            data_list = random.sample(data_list, args.dataset_size)

        data_list = format_toolmind(data_list)

    elif args.dataset_name == "toolace":
        # format_toolace samples immediately after loading
        data_list = format_toolace(args.input_path, args.dataset_size)

    elif args.dataset_name == "xlam":
        # format_xlam samples immediately after loading
        data_list = format_xlam(args.input_path, args.dataset_size)

    elif args.dataset_name == "hammer":
        if not args.input_path:
            raise ValueError("Please provide input path")
        data_list = format_hammer(args.input_path, args.dataset_size)

    data_list = filter_on_simrl(data_list)

    with jsonlines.open(args.output_path, mode="w") as writer:
        writer.write_all(data_list)