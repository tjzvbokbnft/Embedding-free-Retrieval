

text_prompt = "You are an expert in reading and analysis of long texts. Please read the following text carefully. After reading, you will answer multiple choice questions based on the content of the text. Make sure to pay attention to the details. Here is the text:"

question_prompt = (
    "Based on your understanding of the previous long text, please choose the correct answer to the following multiple-choice question. "
    "You **MUST** respond with ONLY a single uppercase letter from A, B, C, or D. Do not include any other text, explanation, or punctuation. "
    "For example, if you think the answer is option B, you should reply with 'B' only. "
    "If the options only contain A or B, you MUST not respond with C or D. "
    "If you are unsure of the correct answer, please make your best guess and still choose one of the options. "
    "If the retrieve-retrieved information is not useful, ignore it and rely on your own knowledge and understanding of the novel to answer."
)
question_prompt_cot = ("""Based on the retrieved data, make your own choose from the given options and explain why.
                          If you are unsure of the correct answer, please make your best guess and still choose one of the options. 
                          YOU MUST OUTPUT the option you choose at the HEAD of your response!!!
                          your answer shoul look like this: your answer/the option you choose + your explaination of why you choose this option/answer
                       """)
question_prompt_cot_final=("""
                           "out put ONLY the option character that this man choose, for example: A,B,C or D" 
                           "You **MUST** respond with ONLY a single uppercase letter from A, B, C, or D. Do not include any other text, explanation, or punctuation.'
                            """)
need_option=("""You will be given a query along with a set of options. Your task is to evaluate whether the information
              provided in the options is valuable for answering the query. If the options contain useful or relevant 
             information that could help in formulating an answer, respond with "yes". If the options do not provide 
             any meaningful or relevant information, respond with "no".
             No extra information , only reply with yes or no""")
filter_prompt="""
                Important Instructions: 
                Not all retrieved information is relevant to answering the question. 
                You must carefully evaluate the retrieved data and filter out any information that does not directly support answering the question. 
                Focus only on the most relevant and useful information for your analysis.
                """
format_prompt= """Output the keywords **ONLY** in the following format: 
            \n[\"keyword1\", \"keyword2\", \"keyword3\", ...] (A JSON array of strings). 
            DO NOT include ANY EXTRA text, explanation, or formatting!!
            Note: Do not include 'chapter' as a keyword."""