# MISTRAL_META="/data/cache/huggingface/hub/models--meta-math--MetaMath-Mistral-7B/snapshots/ae2db13ef5cfff0291560ad098bd8e8c615bf362"
# META_LLAMA_TUNED="/root/models/llama-tuned/llama-2-7b-alpaca-meta_math-10-28-10-20/checkpoint-6000"
# MY_METAMATH="/root/models/llama-tuned/llama-2-7b-alpaca-meta_math-10-28-10-20"
# TUNED_MODEL="/root/models/llama-tuned/llama-2-7b-alpaca-augmented_metalike-10-29-15-36/checkpoint-200"
# LLAMA_MODEL="/data/cache/huggingface/hub/models--meta-llama--Llama-2-7b-hf/snapshots/6fdf2e60f86ff2481f2241aaee459f85b5b0bbb9"
# AUGMENTED_METALIKE_MODEL="/root/models/llama-tuned/llama-2-7b-alpaca-augmented_metalike-10-29-09-17/checkpoint-400"
# TUNED_METAMATH="/root/models/llama-tuned/llama-2-7b-alpaca-augmented_steps-10-30-17-10"
# BASE_MISTRAL="/data/cache/huggingface/hub/models--mistralai--Mistral-7B-v0.1/snapshots/5e9c98b96d071dce59368012254c55b0ec6f8658"
# TUNED_MISTRAL="/root/models/llama-tuned/mistral-7b-alpaca-solving_nve_nin_nups-10-30-20-56"
# EVAL_MODEL=$BASE_MISTRAL

GPUS=8
# END=5000
# END=1000

# MODEL_PATH="/cephfs/xukangping/code/experiments/loop/modeling_coloop_phi.py"
# export MODELING_PATH=$MODEL_PATH
SAVING_DIR="./logs_base_48"
export SAVING_DIR=$SAVING_DIR

# base
# tuned on metamath 6k
# EVAL_MODEL1="/cephfs/xukangping/root/models/xukp20-metamathQA/alpaca-metamathQA-04-21-21-15/checkpoint-9000"
# EVAL_MODEL1="/cephfs/xukangping/root/models/xukp20-metamathQA/alpaca-metamathQA-04-30-14-44/checkpoint-9000"
# 6 epoch = 18k
# EVAL_MODEL1="/cephfs/xukangping/root/models/xukp20-metamathQA/alpaca-metamathQA-04-25-00-47/checkpoint-12000"
# 48 layers
# EVAL_MODEL1="/cephfs/xukangping/root/models/xukp20-metamathQA/alpaca-metamathQA-04-26-18-18/checkpoint-9000"
# 0430, 5 epoch, 1e-5
# EVAL_MODEL1=/cephfs/xukangping/root/models/xukp20-metamathQA/alpaca-metamathQA-04-30-16-41/checkpoint-15000
# 0501, 6 epoch, 1e-5
# EVAL_MODEL1=/cephfs/xukangping/root/models/xukp20-metamathQA/alpaca-metamathQA-05-01-00-26/checkpoint-18000
# 0501, 48
EVAL_MODEL1=/cephfs/xukangping/root/models/xukp20-metamathQA/alpaca-metamathQA-05-01-00-29/checkpoint-18000

# coloop
# tuned on metamath 6k
# EVAL_MODEL2="/cephfs/xukangping/root/models/xukp20-metamathQA/alpaca-metamathQA-04-21-14-49/checkpoint-6000"
# continued on metamath 1epoch,2e-6, only gate 
# EVAL_MODEL2="/cephfs/xukangping/root/models/xukp20-metamathQA/alpaca-metamathQA-04-23-10-29"
# only trained gate on base, 2e-5 3epoch
# 
# fix loop=1
# EVAL_MODEL2="/cephfs/xukangping/root/models/xukp20-metamathQA/alpaca-metamathQA-04-24-11-32-custom/checkpoint-9000"
# 0502 loop=1, 6 epoch
EVAL_MODEL2="/cephfs/xukangping/root/models/xukp20-metamathQA/alpaca-metamathQA-05-01-14-01-custom/checkpoint-3000"

# accelerate launch eval_math.py --model $EVAL_MODEL2 --data_file ./data/test/sample_MATH_test.jsonl --tensor_parallel_size $GPUS --batch_size=128
# accelerate launch eval_math.py --model $EVAL_MODEL1 --data_file ./data/test/sample_MATH_test.jsonl --tensor_parallel_size $GPUS --batch_size=128
# python eval_math.py --model $EVAL_MODEL1 --data_file ./data/test/sample_MATH_test.jsonl --tensor_parallel_size $GPUS --batch_size=400 --use_vllm

# all test
# accelerate launch eval_math.py --model $EVAL_MODEL2 --data_file ./data/test/GSM8K_test.jsonl --tensor_parallel_size $GPUS --batch_size=128 --gsm_8k
# accelerate launch eval_math.py --model $EVAL_MODEL2 --data_file ./data/test/MATH_test.jsonl --tensor_parallel_size $GPUS --batch_size=128
# accelerate launch eval_math.py --model $EVAL_MODEL1 --data_file ./data/test/GSM8K_test.jsonl --tensor_parallel_size $GPUS --batch_size=128 --gsm_8k


EVAL_MODEL1_PREFIX="/cephfs/xukangping/root/models/xukp20-metamathQA/alpaca-metamathQA-05-01-00-29/checkpoint-"
for checkpoint_step in 3000 6000 9000 12000 15000 18000
do
    EVAL_MODEL1=$EVAL_MODEL1_PREFIX$checkpoint_step
    accelerate launch eval_math.py --model $EVAL_MODEL1 --data_file ./data/test/MATH_test.jsonl --tensor_parallel_size $GPUS --batch_size=128
done