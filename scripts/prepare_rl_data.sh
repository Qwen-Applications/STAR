
# Prepare toolmind dataset

# Loading data directly via datasets.load_dataset() will cause errors, so manually download the required datasets

pip install huggingface_hub

mkdir -p ./data

hf download --repo-type dataset Nanbeige/ToolMind graph_syn_datasets/graphsyn.jsonl --local-dir ./data/

python ./data_process/format_data.py --input_path ./data/graphsyn.jsonl --output_path ./data/toolmind_messages.jsonl --dataset_name toolmind


# Data proportion & stratified sampling

python ./data_process/labeling_messages.py --input ./data/toolmind_messages.jsonl --output ./data/toolmind_labeled_messages.jsonl --only-stat-last-round

python ./data_process/data_proportion.py --input ./data/toolmind_labeled_messages.jsonl --output ./data/toolmind_labeled_messages_proportion.jsonl --dataset-size 50000

# Prepare xlam dataset
# Since xlam dataset is gated, you need to apply for access first, then configure HF token to download
python ./data_process/format_data.py --output_path ./data/xlam_messages.jsonl --dataset_name xlam

# Prepare toolace dataset
python ./data_process/format_data.py --output_path ./data/toolace_messages.jsonl --dataset_name toolace

# Since hammer dataset requires model rollout, we provide pre-processed data at ./data/hammer_messages.jsonl


# merge data

jq -c '.' ./data/toolmind_labeled_messages_proportion.jsonl ./data/xlam_messages.jsonl ./data/toolace_messages.jsonl ./data/hammer_messages.jsonl  > ./data/merged_messages.jsonl

# Convert to training set format
python ./messages_to_trainset.py --input ./data/merged_messages.jsonl --output ./data/trainset.jsonl





