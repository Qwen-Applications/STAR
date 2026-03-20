import argparse
import jsonlines
from transformers import AutoTokenizer

def get_args():

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True, help="output jsonl")
    parser.add_argument("--tokenizer-path", type=str, default="Qwen/Qwen3-8B", help="tokenizer path")
    parser.add_argument("--add-reasoning-content", action="store_true", help="add reasoning_content")

    args = parser.parse_args()

    return args

def main():

    args = get_args()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    
    dataset = []
    with jsonlines.open(args.input) as f:
        for obj in f:
            messages = obj['messages']
            tools = obj['tools']
            prompt_text = tokenizer.apply_chat_template(messages[:-1], tools, tokenize=False, add_generation_prompt=True) + "<think>"
            ground_truth = tokenizer.apply_chat_template(messages[-1:], tokenize=False).replace("<|im_start|>assistant\n", '')

            if ground_truth.startswith("<think>"):
                ground_truth = ground_truth[7:]

            if "</think>" not in ground_truth:
                ground_truth = "</think>" + ground_truth

            reasoning_content = messages[-1].get('reasoning_content', None)
            if reasoning_content and args.add_reasoning_content:
                ground_truth = reasoning_content + ground_truth

            dataset.append({
                "id": obj['id'],
                "inputs": prompt_text,
                "targets": ground_truth,
            })

    with jsonlines.open(args.output, 'w') as f:
        f.write_all(dataset)

if __name__ == "__main__":
    main()