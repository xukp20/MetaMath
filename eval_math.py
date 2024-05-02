import argparse
import json
import pdb
import jsonlines

import util
from vllm import LLM, SamplingParams
import sys
MAX_INT = sys.maxsize
INVALID_ANS = "[invalid]"

invalid_outputs = []
def remove_boxed(s):
    left = "\\boxed{"
    try:
        assert s[:len(left)] == left
        assert s[-1] == "}"
        return s[len(left):-1]
    except:
        return None

def process_results(doc, completion, answer):
    split_ans = completion.split('The answer is: ')
    if len(split_ans) > 1:
        ans = split_ans[-1]
        extract_ans_temp = ans.split('.\n')[0]
        extract_ans_temp = extract_ans_temp.strip()
        if len(extract_ans_temp)>0 and extract_ans_temp[-1] == '.':
            extract_ans = extract_ans_temp[0:-1]
        else:
            extract_ans = extract_ans_temp
        extract_ans = extract_ans.strip()
        if util.is_equiv(extract_ans, answer):
            return True
        else:
            return False
    else:
        temp = {'question': doc, 'output': completion, 'answer': answer}
        invalid_outputs.append(temp)
        return False

from eval_gsm8k import extract_answer_number
        
def batch_data(data_list, batch_size=1):
    n = len(data_list) // batch_size
    batch_data = []
    for i in range(n-1):
        start = i * batch_size
        end = (i+1)*batch_size
        batch_data.append(data_list[start:end])

    last_start = (n-1) * batch_size
    last_end = MAX_INT
    batch_data.append(data_list[last_start:last_end])
    return batch_data

def test_hendrycks_math(model, data_path, start=0, end=MAX_INT, batch_size=1, tensor_parallel_size=1, use_vllm=True, gsm_8k=False):
    model_path = model
    hendrycks_math_ins = []
    hendrycks_math_answers = []
    problem_prompt = (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:\n"
        # "### Instruction:\n{instruction}\n\n### Response: Let's think step by step."
    )
    print('prompt =====', problem_prompt)
    with open(data_path, "r+", encoding="utf8") as f:
        for idx, item in enumerate(jsonlines.Reader(f)):
            if "instruction" in item:
                temp_instr = problem_prompt.format(instruction=item["instruction"])
            else:
                temp_instr = problem_prompt.format(instruction=item["query"])
            hendrycks_math_ins.append(temp_instr)
            if "output" in item:
                solution = item['output']
                temp_ans = remove_boxed(util.last_boxed_only_string(solution))
            else:
                temp_ans = item['response'].split('#### ')[1]
                temp_ans = int(temp_ans.replace(',', ''))

            hendrycks_math_answers.append(temp_ans)

    print('total length ===', len(hendrycks_math_ins))
    hendrycks_math_ins = hendrycks_math_ins[start:end]
    hendrycks_math_answers = hendrycks_math_answers[start:end]
    print('length ====', len(hendrycks_math_ins))
    batch_hendrycks_math_ins = batch_data(hendrycks_math_ins, batch_size=batch_size)
    if use_vllm:
        stop_tokens = ["Question:", "Question", "USER:", "USER", "ASSISTANT:", "ASSISTANT", "Instruction:", "Instruction", "Response:", "Response"]
        sampling_params = SamplingParams(temperature=0, top_p=1, max_tokens=2048, stop=stop_tokens)
        print('sampling =====', sampling_params)
        llm = LLM(model=model,tensor_parallel_size=tensor_parallel_size)
    else:
        # use accelerator
        from accelerate import Accelerator
        from accelerate.utils import gather_object
        accelerator = Accelerator()

        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True, padding_side='left')
        tokenizer.pad_token_id = tokenizer.eos_token_id

        import sys,os
        if os.environ.get("MODELING_PATH", None) is not None:
            modeling_path = os.environ.get("MODELING_PATH")
            sys.path.append(os.path.dirname(modeling_path))
            from modeling_coloop_phi import CoLoopPhiForCasualLM, CoLoopPhiConfig

            model = CoLoopPhiForCasualLM.from_pretrained(model, torch_dtype="auto", device_map={"": accelerator.process_index})

            print(model)
            print(model.dtype)

            # model.model.stop_i = 2
            # model.model.gate.stop_i = 2
            model.model.start_log()
            coloop_model = True
        else:
            from transformers import AutoModelForCausalLM, PhiForCausalLM
            model = AutoModelForCausalLM.from_pretrained(model, torch_dtype="auto", device_map={"": accelerator.process_index}, trust_remote_code=True)
            if accelerator.is_main_process:
                print(model)
                print(model.dtype)
            coloop_model = False

        accelerator.wait_for_everyone()

        sampling_params = {
            # "max_new_tokens": 512,
            "max_new_tokens": 1536, # for MATH
            "top_p": 1,
            "use_cache": True,
        }

    res_completions = []
    from tqdm import tqdm
    for idx, (prompt, prompt_answer) in tqdm(enumerate(zip(batch_hendrycks_math_ins, hendrycks_math_answers)), total=len(batch_hendrycks_math_ins)):
        if isinstance(prompt, list):
            pass
        else:
            prompt = [prompt]
        
        if use_vllm:
            completions = llm.generate(prompt, sampling_params)
            for output in completions:
                prompt_temp = output.prompt
                generated_text = output.outputs[0].text
                res_completions.append(generated_text)
        else:
            # generate small batch of tensor parallel
            completions = []

            with accelerator.split_between_processes(prompt) as prompt_batch:
                results = []
                batch_size = 16
                for i in tqdm(range(0, len(prompt_batch), batch_size)):
                    batch_prompt = prompt_batch[i:i+batch_size]
                    inputs = tokenizer(batch_prompt, return_tensors="pt", padding=True, truncation=True)
                    batch_completion = model.generate(
                        inputs['input_ids'].to(model.device),
                        attention_mask=inputs['attention_mask'].to(model.device),
                        **sampling_params,
                        pad_token_id=tokenizer.eos_token_id
                    )
                    completion = tokenizer.batch_decode(batch_completion, skip_special_tokens=True)
                    results.extend(completion)
                
                if coloop_model:
                    model.model.logging_loops()

            completions = gather_object(results)

            for i in range(len(prompt)):
                res_completions.append(completions[i].lstrip(prompt[i]))

        

    results = []
    correct_outputs = []
    wrong_outputs = []
    for idx, (prompt, completion, prompt_answer) in enumerate(zip(hendrycks_math_ins, res_completions, hendrycks_math_answers)):
        if not gsm_8k:
            res = process_results(prompt, completion, prompt_answer)
            results.append(res)
            if res:
                correct_outputs.append({
                    'idx': idx+1,
                    'prompt': prompt,
                    'completion': completion,
                    'answer': prompt_answer
                })
            else:
                wrong_outputs.append({
                    'idx': idx+1,
                    'prompt': prompt,
                    'completion': completion,
                    'answer': prompt_answer
                })
        else:
            y_pred = extract_answer_number(completion)
            if y_pred:
                results.append(float(y_pred) == float(prompt_answer))
                if y_pred == prompt_answer:
                    correct_outputs.append({
                        'idx': idx+1,
                        'prompt': prompt,
                        'completion': completion,
                        'answer': prompt_answer
                    })
                else:
                    wrong_outputs.append({
                        'idx': idx+1,
                        'prompt': prompt,
                        'completion': completion,
                        'answer': prompt_answer
                    })
            else:
                results.append(False)
                invalid_outputs.append({
                    'idx': idx+1,
                    'prompt': prompt,
                    'completion': completion,
                    'answer': prompt_answer
                })

    # only save in the first process
    if use_vllm or accelerator.is_main_process:
        acc = sum(results) / len(results)
        import os, time
        # mm-dd-hh-mm
        timestamp = time.strftime("%m-%d-%H-%M", time.localtime())
        default_dir = os.path.join(os.path.dirname(__file__), 'logs')
        # load saving dir from env
        saving_dir = os.environ.get("SAVING_DIR", default_dir)
        saving_dir = os.path.join(saving_dir, str(timestamp))
        os.makedirs(saving_dir, exist_ok=True)

        with open(os.path.join(saving_dir, 'invalid_outputs.json'), 'w') as f:
            json.dump(invalid_outputs, f, indent=4)
        with open(os.path.join(saving_dir, 'correct_outputs.json'), 'w') as f:
            json.dump(correct_outputs, f, indent=4)
        with open(os.path.join(saving_dir, 'wrong_outputs.json'), 'w') as f:
            json.dump(wrong_outputs, f, indent=4)
        # save info
        info = {
            "model_path": model_path,
            "data_path": data_path, 
            "acc": acc,
            "correct_outputs": len(correct_outputs),
            "wrong_outputs": len(wrong_outputs),
            "invalid_outputs": len(invalid_outputs),
        }
        with open(os.path.join(saving_dir, 'info.json'), 'w') as f:
            json.dump(info, f, indent=4)
            
        print('len correct outputs ====', len(correct_outputs))
        print('len invalid outputs ====', len(invalid_outputs))
        print('start===', start, ', end====',end)
        print('length====', len(results), ', acc====', acc)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default='')  # model path
    parser.add_argument("--data_file", type=str, default='')  # data path
    parser.add_argument("--start", type=int, default=0) #start index
    parser.add_argument("--end", type=int, default=MAX_INT)  # end index
    parser.add_argument("--batch_size", type=int, default=400)  # batch_size
    parser.add_argument("--tensor_parallel_size", type=int, default=8)  # tensor_parallel_size
    parser.add_argument("--use_vllm", action='store_true')  # use vllm or not
    parser.add_argument("--gsm_8k", action='store_true')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    test_hendrycks_math(model=args.model, data_path=args.data_file, start=args.start, end=args.end, batch_size=args.batch_size, tensor_parallel_size=args.tensor_parallel_size, use_vllm=args.use_vllm, gsm_8k=args.gsm_8k)
