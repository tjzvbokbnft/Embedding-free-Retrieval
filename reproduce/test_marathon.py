from src.dataloader import dataloader
import src.local_config as local_config
import src.prompt as prompt
import os
import src.core_functions as core_functions
import src.utils as utils
import time
import logging
import json

if __name__ == "__main__":

    os.makedirs(local_config.res_mc_dir, exist_ok=True)
    os.makedirs(local_config.log_directory, exist_ok=True)
    os.makedirs(local_config.output_folder, exist_ok=True)
    data=dataloader(dataset=local_config.dataset)
    results = []
    TFlist=[]#存储每本书
    start_index=0
    for item in data[start_index:]:#book level
        print('------------------')
        file_id=item['id']
        print("file_id:",file_id)
        domain=item['type']
        # print(file_id,domain)
        context=item['context']
       
        onlyQuery=item['question']
        queryWithOption=onlyQuery
        value=item["options"].get('A')
        if value is not None:
            optionA=f"\nOptionA:{item['options']['A']}\n"
            queryWithOption+=optionA
        value=item["options"].get('B')
        if value is not None:
            optionB=f"OptionB:{item['options']['B']}\n"
            queryWithOption+=optionB
        value=item["options"].get('C')
        if value is not None:
            optionC=f"OptionC:{item['options']['C']}\n"
            queryWithOption+=optionC
        value=item["options"].get('D')
        if value is not None:
            optionD=f"OptionD:{item['options']['D']}\n"
            queryWithOption+=optionD
        # correct_ans=item['answer']
        response = []#存储每本书的响应答案列表
        confidence_list=[]
        correctness_list=[]
        importance_list=[]
        DTI_list=[]
        BTS_list=[]
        retrieve_time_list=[]
        generation_time_list=[]
        log_file_path = os.path.join(local_config.log_directory, f'{file_id}.log')
        core_functions.setup_logger(log_file_path)  # 调用日志配置函数
        query=queryWithOption
        print(len(context))
        print(query)
        # print(queryWithOption)
        #Get retrieve
        #print("----retrieve---")
        resp=core_functions.retrieve(text_input=context, query=query)
        retrieval=resp['retrieve_data']
        retrieve_time_list.append(resp['retrieve_time'])
        prepare_time=resp['prepare_time']
        keywords_extracted=resp['keywords_extracted']
        #importance = utils.compute_imp(query, retrieval)
        #回答
        print(prepare_time)
        print("retrieve_length_:",len(retrieval))

        print("@answering...")
        resp_Answer=utils.Answer(final_input=utils.construct_context(retrieval=retrieval,queryWithOption=queryWithOption),correctAns="A")
        #保存
        generation_time_list.append(resp_Answer['generateTime'])
        print("generation time:",resp_Answer['generateTime'])
        response.append(resp_Answer['ans'])
        TFlist.append(resp_Answer['tfValue'])
        confidence_list.append(resp_Answer['confidenceRate'])
        correctness_list.append(resp_Answer['correctnessRate'])
        #importance_list.append(importance)
        DTI_list.append(resp_Answer['DTI'])
        BTS_list.append(resp_Answer['BTS'])
        #打印
        logging.info(f"""
            ========================================
            book name:{file_id}
            -----------
            query:
            {query}
            -----------------
            all sample ans are :{resp_Answer['all_ans']}
            final ans :{resp_Answer['ans']}
            ------------
            correctness_this_question:{resp_Answer['tfValue']}
            -----------
            confidence:{resp_Answer['confidenceRate']}
            correctness_rate{resp_Answer['correctnessRate']}
            -----------
            keywords_extracted:
            {keywords_extracted}
            -----------
            time:
            retrieve time{retrieve_time_list}
            generate time{generation_time_list}
            total generate time{sum(retrieve_time_list)}
            total retrieve time{sum(generation_time_list)}
            -----------
            retrieved_data_for_this_question:
            {retrieval }
            ------------
            thinking process:
            {resp_Answer['all_cot']}
            
            $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
        """)
        
        print("LLM ans   :",response)
        # print("TFlist    :",TFlist)
        # print("voter正确率:",resp_Answer['correctnessRate'])
        # print("DTIlsit   :",resp_Answer['DTI'])
        # print("BTS_list  :",resp_Answer['BTS'])
            
        temp={"id":file_id,"answer":resp_Answer['ans']}
        results.append(temp)
        
        output_filename = f"{local_config.output_folder}/0_temp_result_{file_id}.json"
        with open(output_filename, "w") as f_output:
            json.dump(results, f_output)
        data_to_store = {
                file_id: {
                    
                    "context_length":len(context),
                    "TFlist": TFlist,
                    "TFvalue":resp_Answer['tfValue'],
                    
                    "confidence_list":confidence_list,
                    "correctness_list":correctness_list,
                    "importance_list":importance_list,

                    'preparing_time':prepare_time,
                    "retrieve_time_list":retrieve_time_list,
                    "generation_time_list":generation_time_list,

                    'avg_retrieve_time':sum(retrieve_time_list)/len(retrieve_time_list),
                    'avg_generation_time':sum(generation_time_list)/len(retrieve_time_list)

                    
                }
            }
        output_filename = f"{local_config.output_folder}/{file_id}_Lists.json"
        with open(output_filename, "w", encoding="utf-8") as f_output:
            json.dump(data_to_store, f_output, ensure_ascii=False, indent=4)
        output_filename = f"{local_config.res_mc_dir}/res_mc.json"
        with open(output_filename, "w") as f_output:
                json.dump(results, f_output)
 

                    



