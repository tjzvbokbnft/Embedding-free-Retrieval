import logging
from ollama import Client
import numpy as np
import requests
import re
import time
import numpy as np
from nltk.tokenize import word_tokenize
#from google import genai
import src.local_config as local_config
import src.prompt as prompt
import ollama
from string_noise import noise
import numpy as np
from numpy.linalg import norm
from src.prompt import format_prompt

import json

recall_index=local_config.recall_index
neighbor_num=local_config.neighbor_num
voter_num=local_config.voter_num
num_ctx=local_config.num_ctx
question_prompt_cot=prompt.question_prompt_cot
question_prompt_cot_final=prompt.question_prompt_cot_final
#config ollama api and model
url=local_config.url
common_model = local_config.common_model

def calculate_cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def get_embedding(text):
    model = embedder_config["model"]
    try:
        response = embedder.embeddings(model=model, prompt=text)
        return response["embedding"]
    except Exception as e:
        print(f"Error getting embedding: {e}")
        return None

# Configuration for the embedding model
embedder_config = {
    "model": "nomic-embed-text",
    "ollama_base_url": "http://localhost:11434"
}

embedder = Client(host=embedder_config["ollama_base_url"])

#utils
def calculate_cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def majority_voting(answers):
    """
    实现少数服从多数逻辑，支持 5 个答案。
    
    参数：
    - answer1, answer2, answer3, answer4, answer5: 5 次调用的结果
    
    返回：
    - 最终的结果：返回次数最多的答案；如果出现次数相同，返回第一个答案。
    """
    # 将所有答案存入一个列表
    
    
    # 使用字典统计每个答案的出现次数
    counts = {}
    for answer in answers:
        counts[answer] = counts.get(answer, 0) + 1
    
    # 找到出现次数最多的答案
    max_count = max(counts.values())
    for answer in answers:
        if counts[answer] == max_count:
            return answer  # 返回第一个满足条件的答案

def get_majority_answer(sentences_list,sorted_results, num_answers,question):
    """
    发送多次请求并通过投票机制选择多数答案。
    
    参数:
        combined_prompt (str): 输入的提示信息。
        num_answers (int): 需要获取的答案数量。

    返回:
        str: 最终的多数答案。
    """
    # 用于模拟 send 函数的结果 (可以替换为实际的 send 函数)
    

    top_k_sentences = sorted_results[:recall_index]
    
    
    all_sentences=extract_neighboring_sentences(sentences_list, top_k_sentences, k_neighbors=neighbor_num)
    
    all_sentences_sorted = sorted(all_sentences, key=lambda x: x['SID'])  # 按照 SID 排序

    retrieve_data = ""

    for entry in all_sentences_sorted:
        retrieve_data += "SentenceID" + str(entry['SID']) + ":" + entry['sentence'] + "\n\n"  # 提取 'sentence' 并加上换行符

    final_input=retrieve_data+"\n\n"+question_prompt_cot+"\n\n"+question+"\n\n"+" PROMPT: tell me do you think if the retrieved information is enough for answering this question"

    # 收集多个答案
    answers = []
    answers_cot=[]
    for _ in range(num_answers):
        ans_cot1=send(final_input)
        #ans_cot2=send("RETRIEVED DATA:"+retrieve_data+"\n\nQUESTION :"+question_formatted+"\n\nSOMEONE'S THINKING:"+ans_cot1+"\n\nPROMPT: Combine the retrieved data with the previous person's reasoning, and come to your own conclusion. Carefully review the previous reasoning and analyze it with the SentenceIDs to determine the answer.")
        ans=send(ans_cot1+"\n\n"+question_prompt_cot_final)
        
        answers.append(ans[0])
        answers_cot.append(ans_cot1)

    

    # 投票统计答案出现次数
    counts = {}
    for answer in answers:
        counts[answer] = counts.get(answer, 0) + 1

    # 找到出现次数最多的答案
    max_count = max(counts.values())
    for answer in answers:
        if counts[answer] == max_count:
            return answer,answers,answers_cot,final_input  # 返回第一个满足条件的答案    

def preprocess_text(sentence,stopwords):
    """
    预处理文本：移除停用词并返回关键词集合。
    """
    stop_words = stopwords
    words = word_tokenize(sentence.lower())
    keywords = set(word for word in words if word not in stop_words)

    return keywords

def find_semantic_common_words(set1, set2):
    """
    查找两个集合中的公共词汇，忽略大小写。
    """
    # 转换集合中的所有字符串为小写
    set1_lower = {word.lower() for word in set1}
    set2_lower = {word.lower() for word in set2}
    
    # 找到交集
    common_words = set1_lower & set2_lower
    return list(common_words)  # 返回列表格式

def count_common_keywords(sentence1, sentence2,stopwords):
    """
    统计两个句子的共同关键词数量。
    """
    keywords1 = preprocess_text(sentence1,stopwords)
    keywords2 = preprocess_text(sentence2,stopwords)
    #print(keywords1)
    #print(keywords2)
    common_keywords = find_semantic_common_words(keywords1,keywords2)
    #print(common_keywords)
    #exit()
    return len(common_keywords),common_keywords

def send(chat):
      # Prompt for summarization
    prompt =chat# Combine the prompt and the text
    # Parameters to pass to Ollama for generating a summary
    payload = {
        "model": common_model,  
        "prompt": prompt,
        "stream": False,
        "options":{
            "num_ctx":num_ctx,
           
        }
    }
    response = requests.post(url, json=payload)
    ret=response.json()["response"]
    return ret

def send_json(chat):
      # Prompt for summarization
    prompt =chat# Combine the prompt and the text
    # Parameters to pass to Ollama for generating a summary
    payload = {
        "model": common_model,  
        "prompt": prompt,
        "stream": False,
        "options":{
            "num_ctx":num_ctx,
        },
        "format":"json"
        

       
    }
    response = requests.post(url, json=payload)
    ret=response.json()["response"]
    return ret

def send_with_seed(chat,seed):#回答的时候用
      # Prompt for summarization
    prompt =chat# Combine the prompt and the text
    # Parameters to pass to Ollama for generating a summary
    payload = {
        "model": common_model,  
        "prompt": prompt,
        "stream": False,
        "options":{
            "num_ctx":num_ctx
        }
        
       
    }
    response = requests.post(url, json=payload)
    #print(response.json()["prompt_eval_count"])
    #print(response.json()["eval_count"])
    ret=response.json()["response"]

    return ret

def chat(chat):
      # Prompt for summarization
    url="http://localhost:11434/api/chat"
    prompt =chat# Combine the prompt and the text

    # Parameters to pass to Ollama for generating a summary
    payload = {
        "model": common_model, 
        "messages": [
                    {
                    "role": "user",
                    "content": prompt
                    }
                ],
        "stream": False,
        "options": {
                       
                        "seed": 42,
                        "num_predict": 1000,
                        "temperature": 0.1,
                        "num_ctx": 10000,
                        "stop": [],
                    }
        
    }
    response = requests.post(url, json=payload)
    print(response.json())
    #print(response.json()['message']['content'])
    exit()
    
    
    return ret

def send_by_specific_model(chat,model):
      # Prompt for summarization

    prompt =chat# Combine the prompt and the text

    # Parameters to pass to Ollama for generating a summary
    payload = {
        "model": model,  
        "prompt": prompt,
        "stream": False,
        "max_tokens": 110000,
        "num_ctx": 110000,
        "num_predict":1000
    }
    response = requests.post(url, json=payload)
   
    ret=response.json()["response"]

    return ret

def format_question_and_options(query):
    """
    格式化问题文本、选项，并包括QID。
    Args:
        query (dict): 包含 'QID', 'Question', 和 'Options' 的字典。
    Returns:
        str: 格式化后的字符串，包含 QID、问题和选项。
    """
    # 提取 QID 和问题文本
    qid = query.get('QID', 'Unknown QID')
    question_text = query.get('Question', 'No Question Provided')
    
    # 开始构建格式化字符串
    formatted_text = f'"QID": "{qid}",\n"Question": "{question_text}",\n"Options": {{\n'
    
    # 遍历选项
    options = query.get('Options', {})
    option_parts = []
    for key, value in options.items():
        option_parts.append(f'    "{key}": "{value}"')
    
    # 将选项合并成一个字符串，并加入到问题字符串中
    formatted_text += ",\n".join(option_parts)
    formatted_text += '\n}'
    
    return formatted_text,question_text

def split_into_sentences_and_count_tokens(text):
    """
    将输入文本分割成句子，并统计每个句子的单词数量。
    """
    # 分句，使用正则表达式匹配句子结束标点（如 . ! ?）并保留标点
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    
    # 统计每个句子的单词数量
    results = []
    for sentence in sentences:
        # 使用空格分词，过滤掉空字符
        tokens = [word for word in sentence.split() if word.strip()]
        token_count = len(tokens)
        results.append({"sentence": sentence, "token_count": token_count})
    
    return results

def extract_neighboring_sentences(sentence_list, top_k_sentences, k_neighbors=5):
    """
    从 sentence_list 中提取 top_k_sentences 的句子及其前后 k_neighbors 个句子。
    
    Args:
        sentence_list (list of dict): 原始句子列表，每个元素是一个字典，包含 SID 和 sentence。
        top_k_sentences (list of tuple): 已选中的句子列表，每个元素是 (sid, sentence, similarity) 的元组。
        k_neighbors (int): 需要提取的相邻句子个数（默认前后各 5 个）。
        
    Returns:
        List of dict: 包含提取结果的句子列表。
    """

    
    # 构建 sid 到句子索引的映射
    sid_to_index = {item['SID']: idx for idx, item in enumerate(sentence_list)}
    
    # 初始化结果列表
    extracted_sentences = []
    
    for sid, _ , _ , _ in top_k_sentences:
        # 获取当前句子的索引
        if sid in sid_to_index:
            current_index = sid_to_index[sid]
            
            # 确定前后句子的范围
            start_index = max(0, current_index - k_neighbors)
            end_index = min(len(sentence_list), current_index + k_neighbors + 1)
            
            # 提取范围内的句子
            neighboring_sentences = sentence_list[start_index:end_index]
            
            # 将句子加入结果列表
            extracted_sentences.extend(neighboring_sentences)
    
    return extracted_sentences

def split_into_sentences_and_count_tokens_with_sid(text):
    """
    将输入文本分割成句子，统计每个句子的单词数量，并为每个句子分配一个 SID。
    """
    # 分句，使用正则表达式匹配句子结束标点（如 . ! ?）并保留标点
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    
    # 统计每个句子的单词数量，并添加 SID
    results = []
    for sid, sentence in enumerate(sentences, start=1):  # SID 从 1 开始
        # 使用空格分词，过滤掉空字符
        tokens = [word for word in sentence.split() if word.strip()]
        token_count = len(tokens)
        results.append({
            "SID": sid,  # 添加句子标识
            "sentence": sentence,
            "token_count": token_count
        })
    return results

def split_into_sentences_and_count_tokens_with_sid_cos(text):
    """
    将输入文本分割成句子，统计每个句子的单词数量，并为每个句子分配一个 SID。
    """
    # 分句，使用正则表达式匹配句子结束标点（如 . ! ?）并保留标点
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    
    # 统计每个句子的单词数量，并添加 SID
    results = []
    for sid, sentence in enumerate(sentences, start=1):  # SID 从 1 开始
        # 使用空格分词，过滤掉空字符
        tokens = [word for word in sentence.split() if word.strip()]
        token_count = len(tokens)
        results.append({
            "SID": sid,  # 添加句子标识
            "sentence": sentence,
            "token_count": token_count,
            "cos":get_embedding(sentence)
        })
    return results

# Configure logging to write to the specified file
def setup_logger(log_file):
    """
    配置日志记录器，并清空日志文件内容。
    """
    # 清空日志文件内容
    with open(log_file, 'w') as f:
        f.truncate(0)  # 清空文件内容

    # 清除旧的处理器，避免重复日志
    logging.getLogger().handlers = []

    # 配置新的日志处理器
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logging.debug("检查日志是否仍然是 DEBUG 级别")
   # print("当前日志级别:", logging.getLogger().getEffectiveLevel())  # 输出当前日志级别

def get_top_k_sentences(sentences_list, question, literary_stopwords, recall_index):
    """
    Calculate similarity between a question and sentences, and return the top-k sentences.

    Args:
        sentences_list (list): List of dictionaries containing 'sentence' and 'SID'.
        question (str): The target question for similarity comparison.
        literary_stopwords (set): A set of stopwords to exclude from similarity calculation.
        recall_index (int): Number of top sentences to return.

    Returns:
        list: A list of tuples with (SID, sentence, similarity, common_words) for top-k sentences.
    """
    similarity_results = []

    for result in sentences_list:
        # Extract current sentence and SID
        sentence = result['sentence']
        sid = result['SID']

        # Calculate similarity with target text
        similarity, common_words = count_common_keywords(question, sentence, literary_stopwords)
        
        
        # bm25_similarity=bm25.bm25_score(query=preprocess_text(question,literary_stopwords), document=result, corpus=sentences_list)
        # exit()
        #print(similarity)
        #print(common_words)
        #exit()
        # Append results
        similarity_results.append((sid, sentence, similarity, common_words))

    # Sort results by similarity in descending order
    sorted_results = sorted(similarity_results, key=lambda x: x[2], reverse=True)

    # Return top-k sentences
    top_k_sentences = sorted_results[:recall_index]
    return top_k_sentences

def get_top_k_sentences_cos(sentences_list, question, recall_index):
    """
    Calculate similarity between a question and sentences, and return the top-k sentences.

    Args:
        sentences_list (list): List of dictionaries containing 'sentence' and 'SID'.
        question (str): The target question for similarity comparison.
        literary_stopwords (set): A set of stopwords to exclude from similarity calculation.
        recall_index (int): Number of top sentences to return.

    Returns:
        list: A list of tuples with (SID, sentence, similarity, common_words) for top-k sentences.
    """
    similarity_results = []
    target_embedding=get_embedding(question)
    for result in sentences_list:
        # Extract current sentence and SID
        sentence = result['sentence']
        sid = result['SID']
        cos_sim_sentence=result['cos']
        # Calculate similarity with target text
        similarity = calculate_cosine_similarity(cos_sim_sentence,target_embedding)
        #print(similarity)
        #print(common_words)
        #exit()
        # Append results
        similarity_results.append((sid, sentence, similarity))

    # Sort results by similarity in descending order
    sorted_results = sorted(similarity_results, key=lambda x: x[2], reverse=True)

    # Return top-k sentences
    top_k_sentences = sorted_results[:recall_index]
    return top_k_sentences



def extract_chapter_info(sentence):
    """
    提取句子中包含的 'chapter' 或其大小写变体，及后面的一个单词。

    Args:
        sentence (str): 输入的句子。

    Returns:
        str: 提取的 'chapter' 和后面的一个单词，若发现是'twenty'及以上，则返回 chapter 和后两个单词。
    """
    # 使用正则表达式匹配 'chapter'（大小写忽略）及其后面的单词
    match = re.search(r'\bchapter\b\s+(\w+(?:\s+\w+)?)', sentence, re.IGNORECASE)
    if match:
        next_word = match.group(1)
        # 如果 next_word 包含 'twenty' 或后续结构，则返回后两个单词
        words = next_word.split()
        if next_word.lower().startswith("twenty") or any(w.lower() in ["thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety", "hundred"] for w in words):
            return f"chapter {' '.join(words[:2])}" if len(words) > 1 else f"chapter {words[0]}"
        return f"chapter {next_word}"
    return "None"  # 若未找到匹配项，则返回 None

def clean_chapter_list(chapter_list):
    ret=[]
    for result in chapter_list:
        result = list(result)  # 将元组转换为列表
        
        temp=extract_chapter_info(result[1])
        result[1]=temp
        if(temp != "None"):
            ret.append(result)
        #print(result[0],":",result[1])
        #print("\n")
    return ret

def insert_chapter_information(chapter_list,all_sentences_sorted):
    chapter_pointer = 1
    sentence_pointer = 0

    # 遍历 chapter_list 和 all_sentences_sorted
    while chapter_pointer < len(chapter_list) and sentence_pointer < len(all_sentences_sorted):
        # 如果当前章节对应的 SID 小于等于当前句子的 SID，插入章节信息
        #print(chapter_list[chapter_pointer][0])
        #print(all_sentences_sorted[sentence_pointer]['SID'])
        if chapter_list[chapter_pointer][0] > all_sentences_sorted[sentence_pointer]['SID']:
            all_sentences_sorted[sentence_pointer]['sentence'] += f" .(FROM  {chapter_list[chapter_pointer-1][1]})."
            sentence_pointer += 1  # 移动到下一个句子
        else:
            chapter_pointer += 1  # 移动到下一个章节
    return all_sentences_sorted

def calculate_score(correct_answers, user_answers):
    # 计算正确率并生成TF列表
    tf_list = [True if correct == user else False for correct, user in zip(correct_answers, user_answers)]
    correct_count = sum(tf_list)  # 统计正确的数量
    total_count = len(correct_answers)  # 总题目数量
    accuracy = correct_count / total_count  # 计算正确率
    return accuracy

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
    q_prompt = f"You will be given a question and some information that could be used to answer the question. Your task is to answer the question. Note that not all information are useful to your answer, they could be irrelevant. You SHOULD disregard all irrelevent sentences and NOT make ANY inference based on YOUR knowledge. You DO NOT need to describe the text. Here is the question: {query}"
    ori_text = send(q_prompt + "\nGiven the information: " + ori_memory)
    baseline_text = mask_text(ori_memory, 0.1, 3)
    noisy_text = mask_text(ori_memory, 0.9, 3)
    baseline_text = send(q_prompt + "\nGiven the information: " + baseline_text)
    noisy_text = send(q_prompt + "\nGiven the information: " + noisy_text)
    ori_emb = ollama.embed(model="mxbai-embed-large", input=ori_text)["embeddings"][0]
    baseline_emb = ollama.embed(model="mxbai-embed-large", input=baseline_text)["embeddings"][0]
    noisy_emb = ollama.embed(model="mxbai-embed-large", input=noisy_text)["embeddings"][0]
    return (2 - cos_similarity(ori_emb, noisy_emb) - cos_similarity(baseline_emb, noisy_emb))/2

def augment_keywords( query, retrieve_data):
    prompt = f"""
    IMPORTANT RULE:{format_prompt}
    Retrieved information{retrieve_data}
    QUERY: {query}
    IMPORTANT RULE:{format_prompt}
    \nPROMPT: Extract proper keywords from retrieved data in order to search more relevant information
    IMPORTANT RULE:{format_prompt}
    """
    keywords = send(prompt)
    return keywords

def depth_expand(retrieve_data,query):
    prompt = f"""
    IMPORTANT RULE:{format_prompt}
    Retrieved information{retrieve_data}
    QUERY: {query}
    IMPORTANT RULE:{format_prompt}
    \nPROMPT: Extract proper keywords from retrieved data in order to search more relevant information
    IMPORTANT RULE:{format_prompt}
  
    """
    return send(prompt)

def compress(retrieve_data,query):
    prompt = f"""
    write more than 400 words!!
    filter important information from RETRIEVAL which can help to solve this query{query}, do not explain, only response with original sentences
    RETRIEVAL:{retrieve_data}
    filter important information from RETRIEVAL which can help to solve this query{query}, do not explain, only response with original sentences
    filter important information from RETRIEVAL which can help to solve this query{query}, do not explain, only response with original sentences
    write more
    write more than 400 words!!
    """
    return send(prompt)

def retrieve(text_input:str,query:str)->str:
    temp_literary_stopwords=local_config.literary_stopwords
    recall_index=local_config.recall_index
    neighbor_num=local_config.neighbor_num
    deep_search_num=local_config.deep_search_num
    deep_search_index=local_config.deep_search_index
    start_prepare_time=time.time()
    logging.debug("splitting...")
    sentences_list = split_into_sentences_and_count_tokens_with_sid(text_input)
    logging.debug("splitting done")
    end_prepare_time=time.time()
    preparetime=end_prepare_time-start_prepare_time
    start_retrieve_time = time.time()
    logging.debug("extracting...")

    keywords_extracted = send(
            f"QUERY: {query}" 
            "\nPROMPT: Extract proper keywords from both the query and its options. "
            "Output the keywords **ONLY** in the following format: "
            "\n[\"keyword1\", \"keyword2\", \"keyword3\", ...] (A JSON array of strings). "
            "DO NOT include any extra text, explanation, or formatting!!"
            "Note: Do not include 'chapter' as a keyword."
        )    
    print("here are keywords extracted from the query:")
    print(keywords_extracted)
    #得到topk个句子
    logging.debug(recall_index)
    top_k_sentences=get_top_k_sentences(sentences_list=sentences_list,question=keywords_extracted,recall_index=recall_index,literary_stopwords=temp_literary_stopwords)
    cnt=0
    for item in top_k_sentences:
        cnt+=1
        print(item)
        if(cnt>=5):
            break
    
    print("get top k sentence done!")
    # 遍历所有 topk sentences
    Max=top_k_sentences[0][2]
    first_class_index =1

    for sid, _, similarity, common_words in top_k_sentences:
        if similarity < Max :
            break
        elif first_class_index >= deep_search_index:
            break
        first_class_index += 1
    for i in range(first_class_index+first_class_index):
            # Convert the tuple to a list
            temp_index=top_k_sentences[i][0]
            # Modify the second element of the list
            temp = (
                "\n!!!PAY MORE ATTENTION TO THIS IMPORTANT SENTENCE!!!\n<START OF THIS SENTENCE>\n"
                + top_k_sentences[i][1]
                + "\n<END OF THIS SENTENCE>\n"
            )

            sentences_list[temp_index-1]['sentence']=temp
    deep_search_neighbour_num = (deep_search_num // first_class_index - 1) // 2
    logging.debug(f"first_class_index:{first_class_index}")
    logging.debug(f"deep neighbour num:{deep_search_neighbour_num}")
    chapter_list=get_top_k_sentences(sentences_list,"CHAPTER",temp_literary_stopwords,100)
    chapter_list=clean_chapter_list(chapter_list)
    #插入chapter信息到所有句子中
    all_sentence_1=extract_neighboring_sentences(sentences_list, top_k_sentences[:first_class_index], k_neighbors=deep_search_neighbour_num)#加深搜索
    all_sentence_2=extract_neighboring_sentences(sentences_list, top_k_sentences[first_class_index:], k_neighbors=neighbor_num)#正常搜索
    all_sentences=all_sentence_1+all_sentence_2
    all_sentences_sorted = sorted(all_sentences, key=lambda x: x['SID'])  # 按照 SID 排序
    all_sentences_sorted=insert_chapter_information(chapter_list=chapter_list,all_sentences_sorted=all_sentences_sorted)
    retrieve_data = ""
    for entry in all_sentences_sorted:
        if len(entry['sentence']) <=3:
            pass
        else:
            retrieve_data += "SentenceID" + str(entry['SID']) + ":" + entry['sentence'] + "\n\n"  # 提取 'sentence' 并加上换行符
    end_retrieve_time = time.time()
    retrieve_time=end_retrieve_time-start_retrieve_time
    print("---retrieve done!---")
    resp={
        'retrieve_data':retrieve_data,
        'prepare_time':preparetime,
        'retrieve_time':retrieve_time,
        'keywords_extracted':keywords_extracted,
    }
    return resp

def retrieve_useful(text_input:str,query:str,cached_keywords:str)->str:
    temp_literary_stopwords=local_config.literary_stopwords
    recall_index=local_config.recall_index
    neighbor_num=local_config.neighbor_num
    deep_search_num=local_config.deep_search_num
    deep_search_index=local_config.deep_search_index
    start_prepare_time=time.time()
    logging.debug("splitting...")
    sentences_list = split_into_sentences_and_count_tokens_with_sid(text_input)
    logging.debug("splitting done")
    end_prepare_time=time.time()
    preparetime=end_prepare_time-start_prepare_time
    start_retrieve_time = time.time()
    logging.debug("extracting...")
    router_index = send(f"""
                        You are a query classifier.

                        QUERY: {query}

                        INSTRUCTIONS:
                        Please judge the type of the question. Your task is to classify it into one of the following two categories:

                        1. If the question is asking for a high-level overview, summary of the whole document, general gist, or global understanding, respond with exactly:
                        global

                        2. If the question is asking for specific details, particular facts, numbers, entities, events, or local information from a specific part of the document, respond with exactly:
                        detailed

                        Do NOT respond with anything other than "global" or "detailed".
                        Do NOT explain your reasoning.
                        Do NOT include punctuation or extra words.
                        Just return: global OR detailed
                        """)
    
    print("\033[1;32m" + router_index + "\033[0m")
    if router_index[0]=="g":
        print("\033[33m" + "A global question detected!" + "\033[0m")
        keywords_extracted = send(
                f"QUERY: {query}" +"Here are some cached keywords"+cached_keywords+
                "\nPROMPT: select keywords from given query and cached keywords "
                "However, the selected cached keywords must have stong relevance to the qeury"
                "Output the keywords **ONLY** in the following format: "
                "\n[\"keyword1\", \"keyword2\", \"keyword3\", ...] (A JSON array of strings). "
                "DO NOT include any extra text, explanation, or formatting!!"
                "Note: Do not include 'chapter' as a keyword."
            ) 
        keywords_extracted = send(
                f"QUERY: {query}" +"Here are some cached keywords"+keywords_extracted+
                "\nPROMPT: Extract proper keywords from given query and cached keywords "
                "Output the keywords **ONLY** in the following format: "
                "\n[\"keyword1\", \"keyword2\", \"keyword3\", ...] (A JSON array of strings). "
                "DO NOT include any extra text, explanation, or formatting!!"
                "Note: Do not include 'chapter' as a keyword."
            )  
    
    else:
        print("\033[33m" + "A detailed question detected!" + "\033[0m")

        keywords_extracted = send(
                f"QUERY: {query}" 
                "\nPROMPT: Extract proper keywords from given query and cached keywords "
                "Output the keywords **ONLY** in the following format: "
                "\n[\"keyword1\", \"keyword2\", \"keyword3\", ...] (A JSON array of strings). "
                "DO NOT include any extra text, explanation, or formatting!!"
                "Note: Do not include 'chapter' as a keyword."
            )    
    print("here are keywords extracted from the query:")
    print("\033[33m" + keywords_extracted + "\033[0m")
    
    #得到topk个句子
    logging.debug(recall_index)
    top_k_sentences=get_top_k_sentences(sentences_list=sentences_list,question=keywords_extracted,recall_index=recall_index,literary_stopwords=temp_literary_stopwords)
    cnt=0
    # for item in top_k_sentences:
    #     cnt+=1
    #     print(item)
    #     if(cnt>=5):
    #         break
    
    print("get top k sentence done!")
    # 遍历所有 topk sentences
    Max=top_k_sentences[0][2]
    first_class_index =1

    for sid, _, similarity, common_words in top_k_sentences:
        if similarity < Max :
            break
        elif first_class_index >= deep_search_index:
            break
        first_class_index += 1
    # for i in range(first_class_index+first_class_index):
    #         # Convert the tuple to a list
    #         temp_index=top_k_sentences[i][0]
    #         # Modify the second element of the list
    #         temp = (
    #             "\n!!!PAY MORE ATTENTION TO THIS IMPORTANT SENTENCE!!!\n<START OF THIS SENTENCE>\n"
    #             + top_k_sentences[i][1]
    #             + "\n<END OF THIS SENTENCE>\n"
    #         )

    #         sentences_list[temp_index-1]['sentence']=temp
    deep_search_neighbour_num = (deep_search_num // first_class_index - 1) // 2
    logging.debug(f"first_class_index:{first_class_index}")
    logging.debug(f"deep neighbour num:{deep_search_neighbour_num}")
    chapter_list=get_top_k_sentences(sentences_list,"CHAPTER",temp_literary_stopwords,100)
    chapter_list=clean_chapter_list(chapter_list)
    #插入chapter信息到所有句子中
    all_sentence_1=extract_neighboring_sentences(sentences_list, top_k_sentences[:first_class_index], k_neighbors=deep_search_neighbour_num)#加深搜索
    all_sentence_2=extract_neighboring_sentences(sentences_list, top_k_sentences[first_class_index:], k_neighbors=neighbor_num)#正常搜索
    all_sentences=all_sentence_1+all_sentence_2
    #all_sentences_sorted = sorted(all_sentences, key=lambda x: x['SID'])  # 按照 SID 排序
    # 用一个字典来去重，确保每个 SID 只保留一个句子（保留第一个出现的）
    unique_sentences_dict = {}
    for sentence in all_sentences:
        sid = sentence['SID']
        if sid not in unique_sentences_dict:
            unique_sentences_dict[sid] = sentence

    # 然后再按 SID 升序排序
    all_sentences_sorted = sorted(unique_sentences_dict.values(), key=lambda x: x['SID'])

    all_sentences_sorted=insert_chapter_information(chapter_list=chapter_list,all_sentences_sorted=all_sentences_sorted)
    retrieve_data = "SentenceID indicates the position of the sentence within the entire context."
    for entry in all_sentences_sorted:
        if len(entry['sentence']) <=3:
            pass
        else:
            retrieve_data += "SentenceID" + str(entry['SID']) + ":" + entry['sentence']   # 提取 'sentence' 并加上换行符
    end_retrieve_time = time.time()
    retrieve_time=end_retrieve_time-start_retrieve_time
    print("---retrieve done!---")
    resp={
        'retrieve_data':retrieve_data,
        'prepare_time':preparetime,
        'retrieve_time':retrieve_time,
        'keywords_extracted':keywords_extracted,
    }
    return resp
