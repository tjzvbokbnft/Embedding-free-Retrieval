import src.local_config as local_config
import src.prompt as prompt
import src.core_functions as core_functions
import time
import logging
import ollama
from string_noise import noise
import numpy as np
from numpy.linalg import norm
def judge_options(query:str,queryWithOption:str):
    resp=core_functions.send(f"""
            Here is a query with options:{queryWithOption}
            PROMPT:{prompt.need_option}
            """)
    logging.debug(resp)
    logging.debug(resp[0])
    if resp[0]=='y':
        return query
    else:
        return queryWithOption

def construct_context(retrieval:str,queryWithOption:str):
    final_input=retrieval+"\n\n"+prompt.question_prompt_cot+"\n\n"+queryWithOption+"\n\n"+ prompt.filter_prompt
    return final_input

def mask_text(text, p, n):
    if p != 0:
        p = 1 - pow((1 - p), 1/n) #since this is a process of n randomizations
    #print(p)
    if(n >= 1):
        text = noise.ocr(text, probability=p)
    if(n >= 2):
        text = noise.moe(text, probability=p) 
    if(n >= 3):
        text = noise.homoglyph(text, probability=p)
    return text

def cos_similarity(vec1, vec2):
    return (np.dot(vec1, vec2) / (norm(vec1) * norm(vec2)))

def compute_imp(query, ori_memory):
    print("start computing")
    q_prompt = f"You will be given a question and some information that could be used to answer the question. Your task is to answer the question. Note that not all information are useful to your answer, they could be irrelevant. You SHOULD disregard all irrelevent sentences and NOT make ANY inference based on YOUR knowledge. You DO NOT need to describe the text. Here is the question: {query}"
    ori_text = core_functions.send(q_prompt + "\nGiven the information: " + ori_memory)
    baseline_text = mask_text(ori_memory, 0.1, 3)
    noisy_text = mask_text(ori_memory, 0.9, 3)
    baseline_text = core_functions.send(q_prompt + "\nGiven the information: " + baseline_text)
    noisy_text = core_functions.send(q_prompt + "\nGiven the information: " + noisy_text)
    ori_emb = ollama.embed(model="mxbai-embed-large", input=ori_text)["embeddings"][0]
    baseline_emb = ollama.embed(model="mxbai-embed-large", input=baseline_text)["embeddings"][0]
    noisy_emb = ollama.embed(model="mxbai-embed-large", input=noisy_text)["embeddings"][0]
    return (2 - cos_similarity(ori_emb, noisy_emb) - cos_similarity(baseline_emb, noisy_emb))/2

def augment_keywords(keywords, query, retrieve_data):
    prompt = f"""You are given a database that contains all information you need to solve a problem, but you have to find the keywords used to search for the correct information. 
    Now, it is known that you have used the following keywords {keywords} but couldn't obtain all nessesary information to answer the question {query}.
    Your task is to carefully analyse the information needed to answer the question based on question and the retrieved data, then make additions to the list of keywords to find all nessesary information to answer the question.
    Output the keywords **ONLY** in the following format: \n[\"keyword1\", \"keyword2\", \"keyword3\", ...] (A JSON array of strings). Do not include any extra text, explanation, or formatting. 
    Important note: realize that only a very vanilla(exact word) search is used to retrieve data from the database, so you MUST add all forms(past tense, present continuing tense, etc) of your keywords into the list. For example: you are given ["jumped"] as a keyword, then you should add ["jump", "jumping", "jumped", and all synonym of jump you find related] to the list.
    In addition, you should read through the retrieved content(with old keywords) carefully and add keywords logically. For example, if you want to find data happened in a week, and Monday data is already retrieved, then you should add Tuesday, Wedsday, Thursday, etc into your list instead of simply use keyword "week".
    You may also delete keywords that seems not contributing to meaningful retrived data. For example, if the query is How many times has the author mentioned "a phrase", then you should delete everything except the phrase.
    Here is the information retrieved using the old keywords: {retrieve_data}
    """
    """while(True):
        try:
            keywords = send(prompt)
            resp = ast.literal_eval(resp)
            if isinstance(resp[0], list):
                temp = []
                for ins in resp:
                    temp.extend(ins)
                resp = temp
            break
        except Exception as e:
            print(e)"""
    keywords = core_functions.send(prompt)
    #print("new keywords: ", keywords)
    return keywords

def depth_expand(retrieve_data):
    prompt = f"""Act as a linguistic analysis specialist. Extract keywords from the following sentences that will be used to search passages for contextual understanding. Follow these requirements:

    1. **Selection Criteria**:  
    - Choose specific technical terms/proper nouns first  
    - Include conceptual phrases (2-3 words max)  
    - Exclude common verbs/adjectives unless critical  

    2. **Format Rules**:  
    - Output EXACTLY as: ["keyword1", "keyword2", ...]  
    - Use double quotes around each keyword  
    - Maintain original capitalization  
    - No trailing commas  
    - DO NOT include SentenceID if it is in the text

    3. **Example**:  
    Input: "BERT revolutionized NLP through masked language modeling."  
    Output: ["BERT", "NLP", "masked language modeling"]  

    Analyze these sentences:  
    [SENTENCES]\n {retrieve_data}
    """
    return core_functions.send(prompt)

def Answer(final_input:str,  correctAns:str):
    print("start answering!")
    
    all_ans=[]
    all_cot=[]
    startGenerate=time.time()
    for i in range(local_config.voter_num):
        print("Answer Round:",i)
        ans_cot1=core_functions.send(final_input)
        ans=core_functions.send(ans_cot1+"\n\n"+prompt.question_prompt_cot_final)
        all_ans.append(ans[0])
        all_cot.append(ans_cot1)
    endGenerate=time.time()
    generateTime=endGenerate-startGenerate
    all_cot_str  = "\n".join(all_cot)  # 直接拼接字符串
    final_answer=core_functions.majority_voting(all_ans)
    if final_answer==correctAns:
        tfValue='T'
    else:
        tfValue='F'
    matching_count=0
    correct_count=0
    for temp in all_ans:
        # 判断是否与最终答案一致
        if temp == final_answer:
            matching_count += 1
        # 判断是否与正确答案一致
        if temp == correctAns:
            correct_count += 1   
    confidenceRate = matching_count / len(all_ans) if all_ans else 0
    correctnessRate = correct_count / len(all_ans) if all_ans else 0
    DTI = correctnessRate * confidenceRate * 100
    BTS = 2 * (correctnessRate * confidenceRate) / (correctnessRate + confidenceRate + 1e-7)
    resp = {
        "ans": final_answer,
        "tfValue": tfValue,
        "correctnessRate": correctnessRate,
        "confidenceRate": confidenceRate,
        "DTI": DTI,
        "BTS": BTS,
        "all_cot": all_cot,
        "generateTime": generateTime,
        "all_ans": all_ans
    }

    return resp
