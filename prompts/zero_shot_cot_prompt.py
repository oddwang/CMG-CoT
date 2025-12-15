def create_cot_true_false_prompt(question):
    prompt_template = """<s>[INST]<<SYS>>
               You are a helpful, respectful and honest assistant. You are to answer the last question by first providing a reasoning followed by the answer. The final answer in the format: The answer is True/False.
               <</SYS>>

               """
    prompt_template += 'Q: ' + question + "\nA: Let's think step by step." + '[/INST]'

    return prompt_template

def create_cot_prompt(question):
    prompt_template = """<s>[INST]<<SYS>>
               You are a helpful, respectful and honest assistant. You are to answer the last question by first providing a reasoning followed by the answer option. The final answer in the format: The answer is (X).
               <</SYS>>

               """
    # 添加当前问题
    prompt_template += 'Q: ' + question + "\nA: Let's think step by step." + '[/INST]'
    return prompt_template