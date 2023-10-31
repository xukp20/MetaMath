MISTRAL_META="/data/cache/huggingface/hub/models--meta-math--MetaMath-Mistral-7B/snapshots/ae2db13ef5cfff0291560ad098bd8e8c615bf362"
META_LLAMA_TUNED="/root/models/llama-tuned/llama-2-7b-alpaca-meta_math-10-28-10-20/checkpoint-6000"
MY_METAMATH="/root/models/llama-tuned/llama-2-7b-alpaca-meta_math-10-28-10-20"
TUNED_MODEL="/root/models/llama-tuned/llama-2-7b-alpaca-augmented_metalike-10-29-15-36/checkpoint-200"
LLAMA_MODEL="/data/cache/huggingface/hub/models--meta-llama--Llama-2-7b-hf/snapshots/6fdf2e60f86ff2481f2241aaee459f85b5b0bbb9"
AUGMENTED_METALIKE_MODEL="/root/models/llama-tuned/llama-2-7b-alpaca-augmented_metalike-10-29-09-17/checkpoint-400"
TUNED_METAMATH="/root/models/llama-tuned/llama-2-7b-alpaca-augmented_steps-10-30-17-10"
BASE_MISTRAL="/data/cache/huggingface/hub/models--mistralai--Mistral-7B-v0.1/snapshots/5e9c98b96d071dce59368012254c55b0ec6f8658"
TUNED_MISTRAL="/root/models/llama-tuned/mistral-7b-alpaca-solving_nve_nin_nups-10-30-20-56"
EVAL_MODEL=$BASE_MISTRAL

GPUS=4
# END=5000
END=1000

python eval_math.py --model $EVAL_MODEL --data_file ./data/test/MATH_test.jsonl --tensor_parallel_size $GPUS --end $END