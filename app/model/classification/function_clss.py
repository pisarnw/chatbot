# import pickle
# from sentence_transformers import SentenceTransformer

# model_NLU = SentenceTransformer('mrp/simcse-model-m-bert-thai-cased') # NLU
# cls_model = pickle.load(open('model_final.pickle', 'rb'))

# def classified_question(question):
#     cls = int(cls_model.predict(model_NLU.encode([question]))[0])# 0 qa, 1 chitchat
#     return cls

# # 0 qa, 1 chitchat
# question = "อยากรวย ทำอย่างไรดี"
# print(classified_question(question))