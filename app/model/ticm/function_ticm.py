import pythainlp
import pandas as pd
from pythainlp import sent_tokenize, word_tokenize
import glob

def search_str(file_path, word):
    with open(file_path, 'r') as file:
        content = file.read()
        if word in content:
            return 1
        else:
            return 0

def search_similar_txt(text_file,text) :
    search_result = []
    words = word_tokenize(text, keep_whitespace=True)
    pair = [''.join([x,y]) for x,y in zip(words[:-1], words[1:])]
    tree = [''.join([x,y,z]) for x,y,z in zip(words[:-2], words[1:-1], words[2:])]
    words = words + pair + tree
    for word in words :
        search_result.append(search_str(f'{text_file}', word))
    confidence = sum(search_result)/len(words)
    # print(words)
    # print(search_result)
    # print(text_file,'confident :',confidence)
    return text_file, confidence

def context_mapping(question, category = False) :
    # threshold = 0.35, category = False
    # threshold = 0.40, category = True
    select_text_files = []
    cut_words= ['อย่างไร','ใคร','หมายถึงอะไร']
    question = question.replace('คืออะไร','คือ')
    for cut_word in cut_words :
        question = question.replace(cut_word,' ')

    if category == True :
        folder_path = 'model/data_context/context_cat/'
        threshold = 0.40
    else : 
        folder_path = 'model/data_context/context_nocat/'
        threshold = 0.35
        
    text_paths = glob.glob(folder_path+'*.txt')

    confidences = []
    for text_path in text_paths :
        # question = 'NP คืออะไร'
        text_file,confidence = search_similar_txt(text_path,question)
        confidences.append(confidence)

    max_confidence = max(confidences)
    for index,confidence in enumerate(confidences):
        if (confidence == max_confidence) and (confidence >= threshold):
            select_text_files.append(text_paths[index].split('/')[-1])

    cotext =''
    for context_file in select_text_files :
        cotext += open(folder_path + context_file, "r").read()
        cotext += '\n\n'

    to_predict = [{ "context": cotext,
                    "qas": [{ "question": question,
                            "id": "0",}] }]
    # print(to_predict)
    return to_predict


question = 'ความเสี่ยงต่อผลการดำเนินงานของกิจการ คืออะไร'
context_mapping(question, category = False)