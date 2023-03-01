from simpletransformers.question_answering import QuestionAnsweringModel
import torch
import numpy as np
torch.multiprocessing.freeze_support()
import torch

cuda_available = torch.cuda.is_available()
model = QuestionAnsweringModel(
    "camembert", "./chatbot/app/model/qanlp/outputs/best_model",use_cuda=cuda_available
)


def run(input):
    predictions, raw_outputs = model.predict(
       input
    )
    # print(raw_outputs)
    print(ret)
    ret=predictions[0]['answer'][np.argmin(raw_outputs[0]['probability'])]
    return ret
   
   

