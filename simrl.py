import re
import json
import collections
from typing import List, Dict, Any

import torch
from rouge_score import rouge_scorer

rouge = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)


def get_ground_truth_from_label(label: str) -> tuple[str, list[dict], str]:

    think, answer = label.split("</think>")
    think = think.strip()
    answer = answer.strip()

    tool_call_objs = []
    reply = None
    if "<tool_call>" in answer:
        tool_calls = re.findall(r"<tool_call>(.*?)</tool_call>", answer, re.S)
        for tool_call in tool_calls:
            tool_call_obj = json.loads(tool_call)
            name = tool_call_obj["name"]
            args = tool_call_obj["arguments"]
            tool_call_objs.append({"name": name, "arguments": args})

    else:
        reply = answer

    return think, tool_call_objs, reply


def parse_generation(content: str) -> tuple[str, list[dict], str]:
    think = re.findall(r"<think>(.*?)</think>", content, re.S)
    if len(think) != 1:
        raise ValueError(f"number of <think></think> tags is not 1")
    think = think[0].strip()

    answer = content.split("</think>")[-1].strip()

    tool_call_objs = []
    reply = None
    if "<tool_call>" in answer:
        tool_calls = re.findall(r"<tool_call>(.*?)</tool_call>", answer, re.S)
        for tool_call in tool_calls:
            tool_call_obj = json.loads(tool_call)
            name = tool_call_obj["name"]
            args = tool_call_obj["arguments"]
            tool_call_objs.append({"name": name, "arguments": args})
    else:
        reply = answer

    return think, tool_call_objs, reply


def get_generation_from_query(query: str) -> str:
    content = query.split("<|im_start|>assistant\n")[-1]
    return content


def extract_tools_from_prompt(prompt: str) -> List[str]:
    tools = re.findall(r"<tools>(.*?)</tools>", prompt, re.S)
    if len(tools) == 0:
        return []
    else:
        tools = tools[-1]

    tool_dict = {}

    for tool in tools.split("\n"):
        tool = tool.strip()
        if tool:
            tool_obj = json.loads(tool)
            name = tool_obj["name"]
            parameters = tool_obj["parameters"]
            if (
                "properties" in parameters
                and isinstance(parameters["properties"], dict)
                and parameters.get("type") is not None
            ):
                parameters = parameters["properties"]
            tool_dict[name] = parameters

    return tool_dict


def check_tool_calls_valid(tool_calls: List[Dict[str, Any]], tool_dict: Dict[str, Any]) -> bool:
    for tool_call in tool_calls:
        name = tool_call["name"]
        if name not in tool_dict:
            return False

        args = tool_call["arguments"]
        parameters = tool_dict[name]

        for param in args:
            if param not in parameters:
                return False

    return True


def get_rouge_score(p_value: str, gt_value: str) -> float:

    if p_value == gt_value:
        return 1
    if gt_value.strip() == "" or p_value.strip() == "":
        return 0

    return rouge.score(gt_value, p_value)["rougeL"].fmeasure


def pop_similar_tool_call(bucket: List[Dict[str, Any]], p_call: Dict[str, Any]) -> bool:

    def get_scores(gt_call: Dict[str, Any], p_call: Dict[str, Any]) -> float:

        i_score = 0

        gt_call = gt_call.copy()

        for key in p_call:
            if key in gt_call:
                gt_value = gt_call.pop(key)
                p_value = p_call[key]
                if isinstance(p_value, str) and isinstance(gt_value, str):
                    i_score += get_rouge_score(p_value, gt_value)

                elif isinstance(p_value, (int, float, bool)) and isinstance(gt_value, (int, float, bool)):
                    i_score += 1 if p_value == gt_value else 0

                else:
                    i_score += 1 if str(p_value) == str(gt_value) else 0

        u_score = len(p_call) + len(gt_call)

        return i_score / u_score if u_score > 0 else 1

    max_similarity = -1
    max_index = 0

    for index, gt_call in enumerate(bucket):
        similarity = get_scores(gt_call["arguments"], p_call["arguments"])
        if similarity > max_similarity:
            max_similarity = similarity
            max_index = index

    bucket.pop(max_index)

    return max_similarity


def compute_function_calling_reward(
    gt_tool_call_objs: List[Dict[str, Any]], p_tool_call_objs: List[Dict[str, Any]]
) -> float:

    gt_buckets = collections.defaultdict(list)
    for gt_call in gt_tool_call_objs:

        if "arguments" not in gt_call:
            gt_call["arguments"] = {}
        gt_buckets[gt_call["name"]].append(gt_call)

    i_score = 0

    for p_call in p_tool_call_objs:
        name = p_call["name"]
        if name not in gt_buckets or len(gt_buckets[name]) == 0:
            continue

        max_similarity = pop_similar_tool_call(gt_buckets[name], p_call)

        i_score += max_similarity

    u_score = sum(len(bucket) for bucket in gt_buckets.values()) + len(p_tool_call_objs)

    return i_score / u_score if u_score > 0 else 1


def print_gt_info(gt_think: str, gt_tool_call_objs: List[Dict[str, Any]], gt_reply: str):
    print(
        "=" * 80,
        f"\n[GT_think]: {gt_think}",
        f"\n[GT_tool_call_objs]: {gt_tool_call_objs}",
        f"\n[GT_reply]: {gt_reply}",
    )


def print_generation_info(generation: str):
    print(f"\n\n[Model Generation]: {generation}")


def print_parsed_generation_info(p_think: str, p_tool_call_objs: List[Dict[str, Any]], p_reply: str):
    print(f"\n\n[Model think]: {p_think}", f"\n[Model tool_calls]: {p_tool_call_objs}", f"\n[Model reply]: {p_reply}")


def print_scores(score: float, format_score: float, answer_score: float):
    print(f"\n\n[Format score]: {format_score}", f"\n[Answer score]: {answer_score}", f"\n[Total score]: {score}")


def reward_func(queries, prompts, labels, print_info=True) -> tuple[torch.Tensor, dict]:

    scores = []
    format_scores = []
    answer_scores = []

    for query, prompt, answer in zip(queries, prompts, labels):

        try:
            gt_think, gt_tool_call_objs, gt_reply = get_ground_truth_from_label(answer)
        except Exception as e:
            raise ValueError(f"label: {answer} is invalid")

        if print_info:
            print_gt_info(gt_think, gt_tool_call_objs, gt_reply)

        try:
            tool_dict = extract_tools_from_prompt(prompt)
        except Exception as e:
            raise ValueError(f"prompt: {prompt} is invalid")

        generation = get_generation_from_query(query)
        if print_info:
            print_generation_info(generation)

        try:
            p_think, p_tool_call_objs, p_reply = parse_generation(generation)
        except Exception as e:

            print("[ERROR] answer format error: ")
            scores.append(-1)
            format_scores.append(-1)
            answer_scores.append(0)
            continue

        if print_info:
            print_parsed_generation_info(p_think, p_tool_call_objs, p_reply)

        try:
            if not check_tool_calls_valid(p_tool_call_objs, tool_dict):
                print("[ERROR] tool calls are invalid: ", generation)
                scores.append(-1)
                format_scores.append(-1)
                answer_scores.append(0)
                continue
        except Exception as e:
            print("[ERROR] tool calls are invalid: ", generation)
            scores.append(-1)
            format_scores.append(-1)
            answer_scores.append(0)
            continue

        format_scores.append(1)

        if len(gt_tool_call_objs) > 0:

            if len(p_tool_call_objs) > 0:
                reward = compute_function_calling_reward(gt_tool_call_objs, p_tool_call_objs)
            else:
                reward = 0

        else:
            if p_reply:
                reward = get_rouge_score(p_reply, gt_reply)
            else:
                reward = 0

        answer_scores.append(reward)
        scores.append(reward)
        if print_info:
            print_scores(scores[-1], format_scores[-1], answer_scores[-1])

    rewards = torch.tensor(scores, dtype=torch.float)
    format_rewards = torch.tensor(format_scores, dtype=torch.float)
    answer_rewards = torch.tensor(answer_scores, dtype=torch.float)

    extra_info = {}
    extra_info["format_rewards"] = format_rewards
    extra_info["answer_rewards"] = answer_rewards

    return {"rewards": rewards, "extra_logs": extra_info, "scores": answer_rewards}
