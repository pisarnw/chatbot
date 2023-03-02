
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
import numpy as np

model_weigth = "gpt_checkpoint_4500/"

tokenizer = GPT2Tokenizer.from_pretrained('kobkrit/openthaigpt-gpt2-instructgpt-poc-0.0.3', bos_token='<|startoftext|>',unk_token='<|unk|>', eos_token='<|endoftext|>', pad_token='<|pad|>')
# model_chat = GPT2LMHeadModel.from_pretrained(model_weigth).cuda()
torch.device('cpu')
model_chat = GPT2LMHeadModel.from_pretrained(model_weigth)
model_chat.resize_token_embeddings(len(tokenizer))

question = 'เย็นนี้กินอะไรดี'

def gpt_fuction(question) :
    pre_prompt = '<|startoftext|> ผมเป็นผู้เชี่ยวชาญด้านการเงิน มีคนถามผมว่า\nQ:'
    post_prompt = '\nแต่ผมไม่สามารถตอบได้ ผมจึงตอบอย่างสุภาพว่า\nA:'
    # generated = tokenizer(f"{pre_prompt} {question}\n {post_prompt}", return_tensors="pt").input_ids.cuda()
    generated = tokenizer(f"{pre_prompt} {question}\n {post_prompt}", return_tensors="pt").input_ids
    sample_outputs = model_chat.generate(generated, do_sample=True, top_k=50, max_length=300, top_p=0.95, temperature=0.5, num_return_sequences=5, pad_token_id=tokenizer.eos_token_id,)

    shortest_ans = ''
    ans_len = np.inf
    for i, sample_output in enumerate(sample_outputs):
        decoded = tokenizer.decode(sample_output, skip_special_tokens=True).replace('\n','')
        if len(decoded) < ans_len and len(decoded) > 150:
            ans_len = len(decoded)
            shortest_ans = decoded

    index = len(pre_prompt) + len(question) + len(post_prompt) - len('<|startoftext|> ')
    output = shortest_ans[index:]
    return output

print(gpt_fuction(question))