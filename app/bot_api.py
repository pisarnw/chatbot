# ตัวอย่างของ Chatbot API
# run with
#   uvicorn --host 0.0.0.0 --reload --port 3000 bot_api:app

from fastapi import FastAPI
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field
from model.ticm.function_ticm import context_mapping

app = FastAPI()

class input(BaseModel):
    question:str = Field(example = "ความเสี่ยงต่อผลการดำเนินงานของกิจการ คืออะไร")

@app.post("/context_mapping_function")
async def context_mapping_function(line: input):
    output = context_mapping(line.question, category = True)

    return output     # None = QA with no answer

# @app.get("/chat")
# async def echo(line: str):
#     return PlainTextResponse(None)      # None = QA with no answer