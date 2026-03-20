"""
Label each round of assistant responses in messages
1. If assistant's content is not empty, tag = [response]
2. If assistant's tool_calls contains multiple tool_calls, tag.append(parallel), otherwise tag.append(single)
3. If the previous message's role is tool, tag.append(multi-step)
4. turns = assistant's position // 2
5. tool_call_num = number of assistant's tool_calls
6. Statistics on tools quantity distribution
"""

import jsonlines
import argparse
from typing import Dict, List
from collections import Counter, defaultdict
from tqdm import tqdm


def get_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--stat-report-path", type=str)
    parser.add_argument("--only-stat-last-round", action="store_true", help="Only statistics for the last round")
    args = parser.parse_args()

    return args


def label_single_assistant_message(message: Dict, messages: List[Dict], index: int) -> Dict:
    """
    Label a single assistant message

    Args:
        message: assistant message
        messages: complete message list
        index: position of assistant message in the list

    Returns:
        labeled message
    """
    # Copy message to avoid modifying original data
    labeled_message = message.copy()

    # Initialize tags
    tags = []

    # 1. Check if content is not empty
    if message.get("content") is not None and message.get("content") != "":
        tags.append("response")

    # 2. Check tool_calls count
    tool_calls = message.get("tool_calls", [])
    tool_call_num = len(tool_calls) if tool_calls else 0

    if tool_call_num > 1:
        tags.append("parallel")
    elif tool_call_num == 1:
        tags.append("single")

    # 3. Check if previous round is tool
    if index > 0:
        prev_message = messages[index - 1]
        if prev_message.get("role") == "tool" and tool_call_num > 0:
            tags.append("multi-step")

    # 4. Calculate turns
    turns = (index+1) // 2

    
    labeled_message["tag"] = tags
    labeled_message["turns"] = turns
    labeled_message["tool_call_num"] = tool_call_num

    return labeled_message


def label_single_conversation(conversation: Dict, only_stat_last_round: bool = False) -> Dict:
    """
    Label all assistant messages in a single conversation

    Args:
        conversation: conversation data containing messages
        only_stat_last_round: whether to only count the last round

    Returns:
        labeled conversation data
    """
    messages = conversation.get("messages", [])
    labeled_messages = []
    stats = {
        "conversation_id": conversation.get("id", ""),
        "total_messages": len(messages),
        "assistant_messages": 0,
        "tags": [],
        "last_round_tags": None,
        "turns_list": [],
        "tool_call_nums": []
    }

    for index, message in enumerate(messages):
        if message.get("role") == "assistant":
            # Label assistant message
            labeled_message = label_single_assistant_message(message, messages, index)
            labeled_messages.append(labeled_message)
            stats["assistant_messages"] += 1

            # Collect tags for statistics
            if not only_stat_last_round or index == len(messages) - 1:
                # If not only counting last round, or if it's the last round
                if not only_stat_last_round:
                    stats["tags"].append(labeled_message["tag"])
                    stats["turns_list"].append(labeled_message["turns"])
                    stats["tool_call_nums"].append(labeled_message["tool_call_num"])
                else:
                    # Only count last round
                    if index == len(messages) - 1 or (index < len(messages) - 1 and messages[index + 1].get("role") != "assistant"):
                        # If it's the last assistant message
                        stats["last_round_tags"] = labeled_message["tag"]
                        stats["tags"].append(labeled_message["tag"])
                        stats["turns_list"].append(labeled_message["turns"])
                        stats["tool_call_nums"].append(labeled_message["tool_call_num"])
        else:
            labeled_messages.append(message)

    # Create labeled conversation
    labeled_conversation = conversation.copy()
    labeled_conversation["messages"] = labeled_messages

    return labeled_conversation, stats


def flatten_tags(tag_list: List[List[str]]) -> List[str]:
    """
    Flatten nested tag list
    """
    flattened = []
    for tags in tag_list:
        flattened.extend(tags)
    return flattened


def generate_statistics(all_stats: List[Dict], only_stat_last_round: bool) -> Dict:
    """
    Generate statistics report

    Args:
        all_stats: statistics for all conversations
        only_stat_last_round: whether to only count the last round

    Returns:
        statistics report
    """
    # Collect all tags
    all_tags = []
    for stat in all_stats:
        all_tags.extend(flatten_tags(stat["tags"]))

    # Count tag frequency
    tag_counter = Counter(all_tags)
    total_tag_count = len(all_tags)

    # Count tag combination frequency
    tag_combinations = []
    for stat in all_stats:
        for tags in stat["tags"]:
            # Convert tag list to hashable tuple
            tag_tuple = tuple(sorted(tags))
            tag_combinations.append(tag_tuple)

    combination_counter = Counter(tag_combinations)
    total_combinations = len(tag_combinations)

    # Count turns and tool_call_num distribution
    all_turns = []
    all_tool_call_nums = []
    for stat in all_stats:
        all_turns.extend(stat.get("turns_list", []))
        all_tool_call_nums.extend(stat.get("tool_call_nums", []))

    turns_counter = Counter(all_turns)
    tool_call_num_counter = Counter(all_tool_call_nums)

    # Count total assistant messages
    total_assistant_messages = sum(stat["assistant_messages"] for stat in all_stats)
    total_conversations = len(all_stats)

    # Generate report
    report = {
        "summary": {
            "total_conversations": total_conversations,
            "total_assistant_messages": total_assistant_messages,
            "only_stat_last_round": only_stat_last_round
        },
        "tag_distribution": {
            "total_tags": total_tag_count,
            "tag_counts": dict(tag_counter),
            "tag_percentages": {
                tag: (count / total_tag_count * 100) if total_tag_count > 0 else 0
                for tag, count in tag_counter.items()
            }
        },
        "tag_combination_distribution": {
            "total_combinations": total_combinations,
            "combination_counts": {
                "+".join(comb): count for comb, count in combination_counter.most_common()
            },
            "combination_percentages": {
                "+".join(comb): (count / total_combinations * 100) if total_combinations > 0 else 0
                for comb, count in combination_counter.items()
            }
        },
        "turns_distribution": {
            "turns_counts": dict(turns_counter),
            "turns_percentages": {
                turns: (count / len(all_turns) * 100) if len(all_turns) > 0 else 0
                for turns, count in turns_counter.items()
            }
        },
        "tool_call_num_distribution": {
            "tool_call_num_counts": dict(tool_call_num_counter),
            "tool_call_num_percentages": {
                num: (count / len(all_tool_call_nums) * 100) if len(all_tool_call_nums) > 0 else 0
                for num, count in tool_call_num_counter.items()
            }
        }
    }

    return report


def print_statistics_report(report: Dict):
    """
    Print statistics report
    """
    print("\n" + "=" * 80)
    print("Statistics Report")
    print("=" * 80)

    print("\n## Overview")
    print(f"Total conversations: {report['summary']['total_conversations']}")
    print(f"Total assistant messages: {report['summary']['total_assistant_messages']}")
    print(f"Only stat last round: {report['summary']['only_stat_last_round']}")

    print("\n## Tag Distribution")
    print(f"Total tags: {report['tag_distribution']['total_tags']}")
    print("\nTag frequency:")
    for tag, count in sorted(report['tag_distribution']['tag_counts'].items(), key=lambda x: -x[1]):
        percentage = report['tag_distribution']['tag_percentages'][tag]
        print(f"  {tag}: {count} ({percentage:.2f}%)")

    print("\n## Tag Combination Distribution")
    print(f"Total combinations: {report['tag_combination_distribution']['total_combinations']}")
    print("\nTag combination frequency (Top 20):")
    for combination, count in list(report['tag_combination_distribution']['combination_counts'].items())[:20]:
        percentage = report['tag_combination_distribution']['combination_percentages'].get(combination, 0)
        print(f"  {combination}: {count} ({percentage:.2f}%)")

    print("\n## Turns Distribution")
    print("\nTurns frequency:")
    for turns, count in sorted(report['turns_distribution']['turns_counts'].items(), key=lambda x: x[0]):
        percentage = report['turns_distribution']['turns_percentages'][turns]
        print(f"  Turn {turns}: {count} ({percentage:.2f}%)")

    print("\n## Tool Call Count Distribution")
    print("\nTool call count frequency:")
    for num, count in sorted(report['tool_call_num_distribution']['tool_call_num_counts'].items(), key=lambda x: x[0]):
        percentage = report['tool_call_num_distribution']['tool_call_num_percentages'][num]
        print(f"  Call {num} tools: {count} ({percentage:.2f}%)")

    print("\n" + "=" * 80 + "\n")


def main():
    args = get_args()

    print(f"Processing file: {args.input}")
    print(f"Output file: {args.output}")
    print(f"Only stat last round: {args.only_stat_last_round}")

    # Read input file
    all_labeled_conversations = []
    all_stats = []

    with jsonlines.open(args.input) as reader:
        for conversation in tqdm(reader):
            # Label each conversation
            labeled_conversation, stats = label_single_conversation(
                conversation,
                only_stat_last_round=args.only_stat_last_round
            )
            all_labeled_conversations.append(labeled_conversation)
            all_stats.append(stats)

    # Save labeled data
    with jsonlines.open(args.output, "w") as writer:
        for labeled_conversation in all_labeled_conversations:
            writer.write(labeled_conversation)

    print(f"Processed {len(all_labeled_conversations)} conversations")

    # Generate statistics report
    report = generate_statistics(all_stats, args.only_stat_last_round)

    # Print report
    print_statistics_report(report)

    # Save statistics report
    if args.stat_report_path:
        with open(args.stat_report_path, "w", encoding="utf-8") as f:
            import json
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"Statistics report saved to: {args.stat_report_path}")

    print("Processing completed!")


if __name__ == "__main__":
    main()
