import argparse
import json
import jsonlines
import random
from typing import Dict, List, Tuple
from collections import Counter
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Input file path(s), multiple files can be comma-separated")
    parser.add_argument("--dataset-size", type=int, required=True, help="Target dataset size")
    parser.add_argument("--output", type=str, required=True, help="Output file path")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--max-turns", type=int, default=10, help="Maximum number of turns")
    parser.add_argument("--response-rate", type=float, default=0.2, help="Proportion of response-tagged data")
    parser.add_argument("--multi-step-rate", type=float, default=0.3, help="Proportion of multi-step-tagged data")
    parser.add_argument("--single-turns-rate", type=float, default=0.3, help="Single-turn proportion")
    parser.add_argument("--max-prompt-tokens", type=int, default=8192, help="Maximum prompt token count")
    parser.add_argument("--parallel-rate", type=float, default=0.5, help="Parallel proportion")
    parser.add_argument("--tokenizer-path", type=str, default="Qwen/Qwen3-8B/", help="Tokenizer path (for accurate prompt token calculation)")

    args = parser.parse_args()

    return args


def read_conversations(file_paths: str) -> List[Dict]:
    """Read conversation data from files, supports multiple files separated by commas"""
    all_conversations = []
    file_path_list = file_paths.split(',')

    for file_path in file_path_list:
        file_path = file_path.strip()
        print(f"Reading file: {file_path}")
        with jsonlines.open(file_path) as reader:
            for item in reader:
                all_conversations.append(item)

    print(f"Total {len(all_conversations)} records read")
    return all_conversations


def get_last_assistant_message(conversation: Dict) -> Tuple[Dict, int]:
    """Get the last assistant message and its index from the conversation"""
    messages = conversation.get("messages", [])
    for i in range(len(messages) - 1, -1, -1):
        if messages[i].get("role") == "assistant":
            return messages[i], i
    return None, -1


def count_prompt_tokens(conversation: Dict, tokenizer=None) -> int:
    """
    Calculate the number of prompt tokens
    Use tokenizer for accurate calculation if available, otherwise estimate using character count
    """
    messages = conversation.get("messages", [])
    tools = conversation.get("tools", [])

    # Get all messages before the last assistant message as prompt
    last_assistant_msg, last_idx = get_last_assistant_message(conversation)

    if last_idx < 0:
        # No assistant message, use all messages
        prompt_messages = messages
    else:
        prompt_messages = messages[:last_idx]

    if tokenizer is not None:
        # Use tokenizer for accurate calculation
        prompt_text = tokenizer.apply_chat_template(
            prompt_messages,
            tokenize=False,
            add_generation_prompt=True,
            tools=tools
        )
        tokens = tokenizer.encode(prompt_text)
        return len(tokens)
    else:
        # Estimate using character count (rough estimate: 1 token ≈ 4 characters)
        total_chars = 0
        for msg in prompt_messages:
            # Estimate character count of message content
            for key, value in msg.items():
                if key == "role":
                    total_chars += len(str(value)) + 10  # Add some format characters
                elif key == "content" and value:
                    total_chars += len(str(value))
                elif key == "tool_calls" and value:
                    total_chars += len(json.dumps(value, ensure_ascii=False))
                elif key == "reasoning_content" and value:
                    total_chars += len(str(value))

        # Estimate character count of tools
        if tools:
            total_chars += len(json.dumps(tools, ensure_ascii=False))

        # Rough estimate: 1 token ≈ 4 characters (for English), Chinese may be less
        # Use divide by 3 as a conservative estimate
        return total_chars // 3


def get_conversation_categories(conversation: Dict) -> Dict[str, bool]:
    """
    Get conversation categories (based on the last assistant message's tags)

    Returns:
        {
            "single_turn": bool,  # turns == 1
            "multi_turn": bool,   # turns > 1
            "parallel": bool,     # tool_calls > 1
            "single": bool,       # tool_calls == 1
            "response": bool,     # "response" in tags (tool_calls == 0 with content)
            "multi_step": bool    # "multi-step" in tags
        }
    """
    last_assistant_msg, _ = get_last_assistant_message(conversation)

    if last_assistant_msg is None:
        # No assistant message, return all False
        return {
            "single_turn": False,
            "multi_turn": False,
            "parallel": False,
            "single": False,
            "response": False,
            "multi_step": False
        }

    turns = last_assistant_msg.get("turns", 0)
    tags = last_assistant_msg.get("tag", [])
    tool_calls = last_assistant_msg.get("tool_calls", [])
    tool_call_num = len(tool_calls) if tool_calls else 0

    return {
        "single_turn": turns == 1,
        "multi_turn": turns > 1,
        "parallel": tool_call_num > 1,
        "single": tool_call_num == 1,
        "response": "response" in tags,
        "multi_step": "multi-step" in tags
    }


def greedy_sample(
    conversations: List[Dict],
    dataset_size: int,
    min_counts: Dict[str, int],
    max_turns: int,
) -> List[Dict]:
    """
    Sample dataset using greedy algorithm (without considering prompt tokens)

    Args:
        conversations: Original conversation list
        dataset_size: Target dataset size
        min_counts: Minimum counts for each category
        max_turns: Maximum turns limit

    Returns:
        Sampled dataset
    """
    # Target sample count (1.1x to ensure enough data for subsequent filtering)
    target_size = int(dataset_size * 1.1)

    # Current counts for each category
    current_counts = {key: 0 for key in min_counts.keys()}

    # Sampled dataset
    sampled = []

    for conv in tqdm(conversations, desc="Sampling data"):
        if len(sampled) >= target_size:
            break

        # Get conversation categories
        categories = get_conversation_categories(conv)

        # Check if turns exceed limit
        last_assistant_msg, _ = get_last_assistant_message(conv)
        if last_assistant_msg and last_assistant_msg.get("turns", 0) > max_turns:
            continue

        # Special handling: skip if current conversation is response category and response quota is met
        if categories.get('response', False) and current_counts['response'] >= min_counts['response']:
            continue

        # Check if at least one category has not reached the minimum count
        should_add = False
        for category, is_match in categories.items():
            if is_match and current_counts[category] < min_counts[category]:
                should_add = True
                break

        if should_add:
            sampled.append(conv)
            # Update counts
            for category, is_match in categories.items():
                if is_match:
                    current_counts[category] += 1

    print(f"\nSampling completed (before prompt tokens filtering):")
    for category, count in current_counts.items():
        target = min_counts[category]
        status = "✓" if count >= target else "✗"
        print(f"  {status} {category}: {count}/{target} (target: {target})")

    return sampled


def filter_by_prompt_tokens(
    conversations: List[Dict],
    max_prompt_tokens: int,
    tokenizer=None
) -> List[Dict]:
    """
    Filter dataset based on prompt tokens

    Args:
        conversations: Conversation list
        max_prompt_tokens: Maximum prompt tokens limit
        tokenizer: Tokenizer (for accurate token calculation)

    Returns:
        Filtered dataset
    """
    filtered = []
    for conv in tqdm(conversations, desc="Filtering prompt tokens"):
        prompt_tokens = count_prompt_tokens(conv, tokenizer)
        if prompt_tokens <= max_prompt_tokens:
            filtered.append(conv)

    filtered_count = len(conversations) - len(filtered)
    print(f"\nFiltering completed: Removed {filtered_count} records exceeding max_prompt_tokens")
    print(f"Remaining data: {len(filtered)} records")

    return filtered


def calculate_statistics(sampled: List[Dict]) -> Dict:
    """
    Calculate the proportion of each category in the sampled dataset

    Returns:
        Statistics report
    """
    total = len(sampled)

    # Collect counts for each category
    category_counts = {
        "single_turn": 0,
        "multi_turn": 0,
        "parallel": 0,
        "single": 0,
        "response": 0,
        "multi_step": 0
    }

    for conv in sampled:
        categories = get_conversation_categories(conv)
        for category, is_match in categories.items():
            if is_match:
                category_counts[category] += 1

    # Calculate proportions
    category_proportions = {
        category: (count / total * 100) if total > 0 else 0
        for category, count in category_counts.items()
    }

    return {
        "total_count": total,
        "category_counts": category_counts,
        "category_proportions": category_proportions
    }


def print_statistics_report(stats: Dict):
    """Print statistics report"""
    print("\n" + "=" * 60)
    print("Dataset Statistics Report")
    print("=" * 60)
    print(f"Total records: {stats['total_count']}")
    print("\nCategory counts and proportions:")

    category_names = {
        "single_turn": "Single-turn (turns==1)",
        "multi_turn": "Multi-turn (turns>1)",
        "parallel": "Parallel calls (tool_calls>1)",
        "single": "Single call (tool_calls==1)",
        "response": "Response with content",
        "multi_step": "Multi-step reasoning"
    }

    for category, count in stats["category_counts"].items():
        proportion = stats["category_proportions"][category]
        name = category_names.get(category, category)
        print(f"  {name}: {count} ({proportion:.2f}%)")

    print("=" * 60 + "\n")


def main():
    args = parse_args()

    # Set random seed
    random.seed(args.seed)

    print("=" * 60)
    print("Data Proportion Sampling Tool")
    print("=" * 60)
    print(f"Input file: {args.input}")
    print(f"Output file: {args.output}")
    print(f"Target dataset size: {args.dataset_size}")
    print(f"Random seed: {args.seed}")
    print(f"Max turns: {args.max_turns}")
    print(f"Max prompt tokens: {args.max_prompt_tokens}")
    print("=" * 60)

    # Load tokenizer (if provided)
    tokenizer = None
    if args.tokenizer_path:
        print(f"\nLoading tokenizer: {args.tokenizer_path}")
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, trust_remote_code=True)
        print("Tokenizer loaded successfully")
    else:
        print("\nNo tokenizer path provided, will estimate prompt tokens using character count")

    # Calculate minimum counts for each category
    min_counts = {
        "single_turn": int(args.single_turns_rate * args.dataset_size),
        "multi_turn": int((1 - args.single_turns_rate) * args.dataset_size),
        "parallel": int(args.parallel_rate * args.dataset_size),
        "single": int((1 - args.parallel_rate) * args.dataset_size),
        "response": int(args.response_rate * args.dataset_size),
        "multi_step": int(args.multi_step_rate * args.dataset_size),
    }

    print("\nTarget counts for each category:")
    for category, count in min_counts.items():
        print(f"  {category}: {count}")

    # Read data
    print("\nStarting to read data...")
    conversations = read_conversations(args.input)

    # Shuffle data
    print("\nShuffling data...")
    random.shuffle(conversations)

    # Greedy sampling (without considering prompt tokens)
    print(f"\nStarting data sampling (target: {int(args.dataset_size * 1.1)} records)...")
    sampled = greedy_sample(
        conversations,
        args.dataset_size,
        min_counts,
        args.max_turns,
    )

    print(f"\nSampled records: {len(sampled)}")

    # Filter by prompt tokens
    print(f"\nStarting prompt tokens filtering (limit: {args.max_prompt_tokens})...")
    filtered = filter_by_prompt_tokens(
        sampled,
        args.max_prompt_tokens,
        tokenizer
    )

    # Sample dataset_size records from filtered data
    if len(filtered) >= args.dataset_size:
        print(f"\nRandomly sampling {args.dataset_size} records from filtered data...")
        final_sampled = random.sample(filtered, args.dataset_size)
    elif len(filtered) > 0:
        print(f"\nWarning: Only {len(filtered)} records remain after filtering, less than target {args.dataset_size}")
        print("Using all filtered data")
        final_sampled = filtered
    else:
        print("\nError: No data left after filtering! Please check parameter settings")
        return

    print(f"Final dataset size: {len(final_sampled)} records")

    # Calculate proportions
    stats = calculate_statistics(final_sampled)
    print_statistics_report(stats)

    # Save dataset
    print(f"Saving dataset to: {args.output}")
    with jsonlines.open(args.output, "w") as writer:
        writer.write_all(final_sampled)

    print("\nProcessing completed!")


if __name__ == "__main__":
    main()
