MISTRAL_META="/data/cache/huggingface/hub/models--meta-math--MetaMath-Mistral-7B/snapshots/ae2db13ef5cfff0291560ad098bd8e8c615bf362"
TUNED_MODEL="/root/models/llama-tuned/llama-2-7b-alpaca-meta_math-10-27-21-24/checkpoint-800"
EVAL_MODEL=$MISTRAL_META

echo python eval_math.py --model $EVAL_MODEL --data_file ./data/test/MATH_test.jsonl --tensor_parallel_size 4