import src.local_config as local_config
import os
import src.core_functions as core_functions
import json
def dataloader(dataset:str,):
    resp=[]

    if dataset=='novelQA':
        txtdir=local_config.novelQAtxtdir
        jsondir=local_config.novelQAjsondir
        ansdir=local_config.novelQAansdir

        files = sorted(os.listdir(txtdir)) 
        start_file = "A Game of Thrones.txt"
        start_index = files.index(start_file) if start_file in files else 0
        with open(ansdir, "r", encoding="utf-8") as f:
            correct_answer_dict= json.load(f)  # 解析 JSON 数据为 Python 字典或列表
        # Start processing from the specified file
        for i, file in enumerate(files[start_index:], start=start_index):
            file_name_without_extension = os.path.splitext(file)[0]
            with open(os.path.join(txtdir, file), "r") as f:
                text=f.read()
            jsonfile = file.replace(".txt", ".json")

            with open(os.path.join(jsondir, jsonfile), "r") as f_json:
                queries = json.load(f_json)

            queries_formatted=[]
            queries_only=[]
            
            for query,answer in zip(queries,correct_answer_dict):
                formatted_question,question_only = core_functions.format_question_and_options(query)
                # print(queries_only)
                queries_formatted.append(formatted_question)   
                queries_only.append(question_only)
                   
            # print(queries_only)
            # exit()
            data={
                'file_id':file,
                'context':text,
                'queries':queries_only,
                'queries_with_options':queries_formatted,
                'correct_answer_dict':correct_answer_dict[file_name_without_extension]
            }
            resp.append(data)
    elif dataset=='marathon':
        jsondir=local_config.marathonjsondir
        with open(jsondir, 'r', encoding='utf-8') as file:
            files = json.load(file)  # 解析 JSON 文件
            return files
        





