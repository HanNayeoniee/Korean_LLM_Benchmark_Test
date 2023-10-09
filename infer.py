import os
import jsonlines
import torch
from transformers import pipeline, AutoModelForCausalLM
import pandas as pd
from tqdm import tqdm




MODEL_NAME = 'beomi/llama-2-ko-7b'

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
).to(device=f"cuda", non_blocking=True)
model.eval()

pipe = pipeline(
    'text-generation', 
    model=model,
    tokenizer=MODEL_NAME,
    device=0
)

def ask(x, is_input_full=False):
    ans = pipe(
        x,
        do_sample=True,
        max_new_tokens=512,
        temperature=0.7,
        top_p=0.9,
        return_full_text=False,
        eos_token_id=2,
    )
    return ans[0]['generated_text']


idx_list, q_list, prompt_list, pred_list, ans_list = [], [], [], [], []

cnt = 1
file_path = "./Belebele/kor_Hang.jsonl"


pbar = tqdm(total=900, unit='unit', unit_scale=True)
with jsonlines.open(file_path, "r") as f:
    for line in f.iter():
        prompt = f"Q: {line['question']}\n\nP: {line['flores_passage']}\n\nA: {line['mc_answer1']}\nB: {line['mc_answer2']}\nC: {line['mc_answer3']}\nD: {line['mc_answer4']}\n\n정답은"
        response = ask(prompt)
        idx_list.append(cnt)
        q_list.append(line['question'])
        prompt_list.append(prompt)
        pred_list.append(response)
        ans_list.append(line["correct_answer_num"])
        cnt += 1
        pbar.update(1)    
pbar.close()


res_df = pd.DataFrame({
    "Index": idx_list,
    "Question": q_list,
    "Prompt": prompt_list,
    "Pred_Answer": pred_list,
    "Real_Answer": ans_list
})

save_path = f"./output/{MODEL_NAME.replace('/', '_')}.xlsx"
res_df.to_excel(save_path, index=False)
print(f"Saved to ... {save_path}")