from ollama import Client
#Config LLM parameters here
recall_index=30
neighbor_num=1
deep_search_index=5
deep_search_num=10
voter_num=1
num_ctx=5000

#config ollama
url="http://localhost:11434/api/generate"
common_model = "llama3.1:latest"
embedder_config = {
    "model": "nomic-embed-text",
    "ollama_base_url": "http://localhost:11434"
}
embedder = Client(host=embedder_config["ollama_base_url"])

#novelqa data
dataset='marathon'#novelQA, marathon
novelQAtxtdir="nvQA/book/PublicDomain"
novelQAjsondir="nvQA/data/PublicDomain"
novelQAansdir="nvQA/0_questions_with_correct_answer/CorrectAnswers/res_mc.json"
# dataset='longbench-v2'
longBenchjsondir='newData/LongBench_v2.json'
#marathon
marathonjsondir='newData/marathon.json'

#stopwords
literary_stopwords ={
    "the","is","am","are","was","were","be","been","being","he","she","it","they",
    "a","an","to","of","for","in","on","with","at","by","and","but","or","so","if","that",
    "this","those","these","i","you","me","we","us","them","there","then","what","which",
    "as","from","up","down","out","over","under","again","further","about","above","below",
    "between","into","through","during","before","after","right","left","just","their",
    'did', 'doing', 'do', 'never', 'yes', '[', ']', 'how', 'many', 'times', 'that', 'the', 'author', 
    'mentioned', 'implication', 'metaphor', 'described', 'have', 'symbol', "'", '*', '.', ',', '?', 
    "''", 'in', '``', 'so', 'there', 'are', 'and', "'s", 'has', 'this', 'happened', 'being', 'novel', 
    'plot', '**', '***', 'was', 'be', 'being', 'is', 'were', 'i', 'but', 'for', 'my', '(', 'which', 
    'past', 'its', 'it', 'had', 'from', 'with', ')', 'of', 'to', 'a','once','1','2','3','4',':', 'now', 'you','me',
    'into','emun',"his","him","her","someone","who","said","few","three","yes","no",'himself',"where","not","all","hundred",
    "would","when",
}

#config history folders
matching_method=f'Gglobal+{voter_num}voters+commonword+cot+2x{neighbor_num}neighbours+deep_search{deep_search_num}+CTX_{num_ctx}'
history_folder=f'History/{dataset}+{common_model}+{matching_method}@{recall_index}'#root folder
res_mc_dir=f'{history_folder}/res_mc'#reslt folder
log_directory = f'{history_folder}/logs'#logging folder
output_folder = f'{history_folder}/Test_results'#question with answer folder