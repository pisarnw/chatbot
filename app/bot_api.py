# ตัวอย่างของ Chatbot API
# run with
#   uvicorn --host 0.0.0.0 --reload --port 3000 bot_api:app

from fastapi import FastAPI
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field
from model.ticm.function_ticm import context_mapping

# for QA model
from simpletransformers.question_answering import QuestionAnsweringModel
import torch
import numpy as np

# for GBT model
from transformers import GPT2LMHeadModel, GPT2Tokenizer
# import torch
# import numpy as np

# for classification
import pickle
from sentence_transformers import SentenceTransformer


# load model QA
cuda_available = torch.cuda.is_available()
model = QuestionAnsweringModel(
    "xlmroberta", "./app/model/qanlp/xlm-roberta-large-squad2-118/best_model",use_cuda=cuda_available
)


# conect huggingface
token = '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'
from huggingface_hub import HfApi, HfFolder
api=HfApi()
api.set_access_token(token)
folder = HfFolder()
folder.save_token(token)

# load model GPT
model_weigth = "./app/model/chat_gpt/gpt_checkpoint_4500"
tokenizer = GPT2Tokenizer.from_pretrained('kobkrit/openthaigpt-gpt2-instructgpt-poc-0.0.3', bos_token='<|startoftext|>',unk_token='<|unk|>', eos_token='<|endoftext|>', pad_token='<|pad|>')
# model_chat = GPT2LMHeadModel.from_pretrained(model_weigth).cuda()
model_chat = GPT2LMHeadModel.from_pretrained(model_weigth)
model_chat.resize_token_embeddings(len(tokenizer))

# load model clssifition question
model_NLU = SentenceTransformer('mrp/simcse-model-m-bert-thai-cased') # NLU
cls_model = pickle.load(open('./app/model/classification/model_final.pickle', 'rb'))

def question_answer_funtion(input):
    global model
    # predictions, raw_outputs = model.predict(input)
    # output=predictions[0]['answer'][np.argmin(raw_outputs[0]['probability'])]
    # if output == "empty" :
    #     output = ""
    ans_select_max_len =[]
    for i in input:
        ans1,prob1=model.predict([i])
        if len(ans1[0]['answer']) > 1:
            if ans1[0]['answer'][0] == '':
                x = ans1[0]['answer'][1]
                ans_select_max_len.append(x)
            else:
                x = ans1[0]['answer'][0]
                ans_select_max_len.append(x)
        else:
            x = ans1[0]['answer'][0].replace('empty','')
            ans_select_max_len.append(x)
    return ans_select_max_len[0]

def gpt_fuction(question) :
    global model_chat
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

    output = shortest_ans[shortest_ans.index('A: ')+3:]
    return output

def classified_question(question):
    global cls_model, model_NLU
    cls = int(cls_model.predict(model_NLU.encode([question]))[0])# 0 qa, 1 chitchat
    return cls

app = FastAPI()

@app.get("/chat")
async def context_mapping_function(line:str):
    context_to_predict = context_mapping(line, category = False)
    if context_to_predict[0]["context"] == "" :
        question = context_to_predict[0]["qas"][0]["question"]
        clss = classified_question(question)
        # 0 qa, 1 chitchat
        if clss == 1 : #chitchat
            output = gpt_fuction(context_to_predict)
        else : # blank QA
            output = ""
    else : # QA or blank QA
        output = question_answer_funtion(context_to_predict)

    return  PlainTextResponse(output) # None = QA with no answer

if __name__ == "__main__":
    print('xxx')