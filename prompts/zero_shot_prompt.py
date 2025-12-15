def create_zero_shot_prompt(question):
    prompt_template = """<s>[INST]<<SYS>>
               You are a helpful, respectful and honest assistant. You are to answer the last question by first providing a reasoning followed by the answer option, similar to previous examples. The final answer in the format: The answer is (X).
               <</SYS>>

               """
    prompt_template += 'Q: ' + question + "\nA: The answer is (" + '[/INST]'

    return prompt_template
