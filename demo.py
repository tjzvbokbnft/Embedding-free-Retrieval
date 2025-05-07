import os
import src.core_functions as core_functions
import src.utils as utils
import src.local_config as local_config
import time
# 配置文件路径
# txtdir = "In put your txt document path here"

# # start_file = "Wuthering Heights.txt"

# # # 读取小说文本
# # files = sorted(os.listdir(txtdir))
# # start_index = files.index(start_file) if start_file in files else 0
# # file = files[start_index]

# with open(os.path.join(txtdir, file), "r") as f:
#     context = f.read()


txt_path = "Input your txt document path here"  # 例如: "data/Wuthering_Heights.txt"

# 获取文件名用于后续展示
file = os.path.basename(txt_path)

# 读取小说文本内容
with open(txt_path, "r", encoding="utf-8") as f:
    context = f.read()


time_stamp=time.time()
DEMO_LOG_DIR=f"DEMO_LOG/{local_config.common_model}+{time_stamp}"
os.makedirs(DEMO_LOG_DIR, exist_ok=True)




# 交互式循环
print(f"\n🧠 Loaded base context: {file}")
print("💬 Ask questions about the base context. Type 'exit' to quit.\n")

cold_start_query_list = [
    "What core themes and genres best describe this novel?",
    "What is the central narrative hook or premise introduced early in the story?",
    "How would you describe the author's writing style and tone, based on the opening chapters?",

]


temp_memory=""
cache_keywords=""
cold_start_index=len(cold_start_query_list)
round=0
while True:
    if round<cold_start_index:
        query=cold_start_query_list[round]
        print(query)
    else:
        query = input("You: ")
        if query.lower() in ['exit', 'quit']:
            print("👋 Exiting.")
            break
    new_keywords=""
    # 检索相关文本
    if len(cache_keywords)!= 0:
        query = query
    if len(temp_memory)>=1000:
        resp = core_functions.retrieve_useful(text_input=temp_memory, query=query,cached_keywords=cache_keywords)
        memory_retrieval = resp['retrieve_data']
        new_keywords+=resp["keywords_extracted"]
    else:
        memory_retrieval=temp_memory
    resp = core_functions.retrieve_useful(text_input=context, query=query,cached_keywords=cache_keywords)
    context_retrieval = resp['retrieve_data']
    new_keywords+=resp["keywords_extracted"]
  
    # 构造最终输入
    final_input = "RETRIEVAL from context"+context_retrieval+"\n\nRETRIVAL FROM AGENT MEMORY:\n\n"+memory_retrieval  + "\n\nBASED on the context retrieval and agent memory above, response to the user's query or request:" + query

    # 模型作答
    final_response = core_functions.send(final_input)
    temp_memory+=final_response
    mem_keywords=core_functions.send(
            f"Context: {query}" 
            "\nPROMPT: Extract proper keywords/element/entity/location from given context "
            "Output the keywords **ONLY** in the following format: "
            "\n[\"keyword1\", \"keyword2\", \"keyword3\", ...] (A JSON array of strings). "
            "DO NOT include any extra text, explanation, or formatting!!"
            "Note: Do not include 'chapter' as a keyword."
        )   
    new_keywords+=mem_keywords
    cache_keywords+=new_keywords
    print("\n🤖 Agent: \033[1;32m" + "="*20 + "\033[0m\n")
    print("\n🤖 Agent: \033[1;32m" + final_response.strip() + "\033[0m\n")
    print("\n🤖 Agent: \033[1;32m" + "="*20 + "\033[0m\n")
    # ✅ 写入 mem_log.txt 文件
    with open(f"{DEMO_LOG_DIR}/agent_mem.txt", "a", encoding="utf-8") as log_file:
        log_file.write(f"\n\n==== Round {round} ====\n")
        log_file.write(f"🧍‍♂️ User: {query.strip()}\n")
        log_file.write(f"🧠 Agent Memory(newly updated):\n{final_response.strip()}\n")
        # ✅ 写入 mem_log.txt 文件
    with open(f"{DEMO_LOG_DIR}/context_retrieval_log.txt", "a", encoding="utf-8") as log_file:
        log_file.write(f"\n\n==== Round {round} ====\n")
        log_file.write(f"🧍‍♂️ User: {query.strip()}\n")
        log_file.write(f"🤖 context retrieval:\n{context_retrieval.strip()}\n")
    with open(f"{DEMO_LOG_DIR}/mem_retrieval_log.txt", "a", encoding="utf-8") as log_file:
        log_file.write(f"\n\n==== Round {round} ====\n")
        log_file.write(f"🧍‍♂️ User: {query.strip()}\n")
        log_file.write(f"🤖 mem retrieval:\n{memory_retrieval}\n")
    with open(f"{DEMO_LOG_DIR}/keywords_cached.txt", "a", encoding="utf-8") as log_file:
        log_file.write(f"\n\n==== Round {round} ====\n")
        log_file.write(f"🧍‍♂️ User: {query.strip()}\n")
        log_file.write(f"🤖 keywords cached:\n{new_keywords}\n")
    round+=1



    
       
