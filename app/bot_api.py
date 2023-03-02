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

# load model
cuda_available = torch.cuda.is_available()
model = QuestionAnsweringModel(
    "camembert", "./app/model/qanlp/outputs/best_model",use_cuda=cuda_available
)

def question_answer_funtion(input):
    predictions, raw_outputs = model.predict(input)
    output=predictions[0]['answer'][np.argmin(raw_outputs[0]['probability'])]
    if output == "empty" :
        output = ""
    

    return output

app = FastAPI()

# class input(BaseModel):
#     line:str = Field(example = "ความเสี่ยงต่อผลการดำเนินงานของกิจการ คืออะไร")

# @app.post("/context_mapping_function")
# async def context_mapping_function(line: input):
#     context_to_predict = context_mapping(line.line, category = False)
#     # if context_to_predict[0]["context"] == "" :
#         #classifition
#     output = question_answer_funtion(context_to_predict)
#     return output  # None = QA with no answer

@app.get("/chat/{line}/")
async def context_mapping_function(line:str):
    context_to_predict = context_mapping(line, category = False)
    # if context_to_predict[0]["context"] == "" :
        #classifition
    output = question_answer_funtion(context_to_predict)
    return output  # None = QA with no answer

# @app.get("/chat")
# async def echo(line: str):
#     return PlainTextResponse(None)      # None = QA with no answer