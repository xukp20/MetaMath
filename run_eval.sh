MISTRAL_META="/data/cache/huggingface/hub/models--meta-math--MetaMath-Mistral-7B/snapshots/ae2db13ef5cfff0291560ad098bd8e8c615bf362"
META_LLAMA_TUNED="/root/models/llama-tuned/llama-2-7b-alpaca-meta_math-10-28-10-20/checkpoint-6000"
MY_METAMATH="/root/models/llama-tuned/llama-2-7b-alpaca-meta_math-10-28-10-20"
TUNED_MODEL="/root/models/llama-tuned/llama-2-7b-alpaca-augmented_metalike-10-29-15-36/checkpoint-200"
LLAMA_MODEL="/data/cache/huggingface/hub/models--meta-llama--Llama-2-7b-hf/snapshots/6fdf2e60f86ff2481f2241aaee459f85b5b0bbb9"
AUGMENTED_METALIKE_MODEL="/root/models/llama-tuned/llama-2-7b-alpaca-augmented_metalike-10-29-09-17/checkpoint-400"
TUNED_METAMATH="/root/models/llama-tuned/llama-2-7b-alpaca-augmented_steps-10-30-10-36"
EVAL_MODEL=$MY_METAMATH

GPUS=8

echo python eval_math.py --model $EVAL_MODEL --data_file ./data/test/MATH_test.jsonl --tensor_parallel_size $GPUS