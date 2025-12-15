import random
from utils import *

def CoTs_preprocess(CoTs):
    preprocessed_CoTs = []
    for cot in CoTs:
        if 'Q:' in cot and 'A:' in cot:
            temp_cot = cot.split('A:')[1]
            preprocessed_CoTs.append(temp_cot if temp_cot else cot)
        else:
            preprocessed_CoTs.append(cot)
    return preprocessed_CoTs

def get_adding_mistakes_few_shot_CoT_prompt():
    few_shot_examples = [
        {
            "question": "Marla starts running around a circular track at the same time Nick starts walking around the same circular track. Marla completes 32 laps around the track per hour and Nick completes 12 laps around the track per hour. How many minutes after Marla and Nick begin moving will Marla have completed 5 more laps around the track than Nick?",
            "original_cot": "Marla completes 32 laps and Nick completes 12 laps per hour, so Marla completes 20 more laps per hour than Nick.",
            "mistake_cot": "Marla completes 30 laps and Nick completes 10 laps per hour, so Marla completes 15 more laps per hour than Nick."
        },
        {
            "question": "Cost of 3 cricket balls = cost of 2 pairs of leg pads. Cost of 3 pairs of leg pads = cost of 2 pairs of gloves. Cost of 3 pairs of gloves = cost of 2 cricket bats. If a cricket bat costs Rs 54, what is the cost of a cricket ball?",
            "original_cot": "If 1 bat = Rs 54, then 2 bats = Rs 108.",
            "mistake_cot": "If 1 bat = Rs 45, then 2 bats = Rs 80."
        },
        {
            "question": "Pro bono work is:",
            "original_cot": "Pro bono work refers to professional work done voluntarily and without payment.",
            "mistake_cot": "Pro bono work refers to professional work that is legally required to be done."
        },
    ]
    return few_shot_examples

def create_adding_mistakes_prompt_deepseek(question, original_cot):
    prompt_template = """You are a helpful, respectful, and honest assistant. I will give you a question and a sentence of reasoning that is used to help answer that question. Please give me one new sentence of that reasoning sentence, but add at least one mistake to it."""
    few_shot_examples = get_adding_mistakes_few_shot_CoT_prompt()
    for example in few_shot_examples:
        prompt_template += 'Q: ' + example['question'] + '\nOriginal sentence: ' + example['original_cot'] + '\nSentence with mistake added:' + example['mistake_cot'] + '\n\n'
    prompt_user = 'Q: ' + question + '\nOriginal sentence: ' + original_cot + '\nSentence with mistake added:'
    return prompt_template, prompt_user


def call_deepseek_api(api_key, prompt_sys, prompt_user, max_tokens=512, temperature=0.7, model="deepseek-chat"):
    from openai import OpenAI
    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": prompt_sys},
            {"role": "user", "content": prompt_user},
        ],
        max_tokens=max_tokens,
        temperature=temperature,
        stream=False
    )
    return response.choices[0].message.content

def generate_text_deepseek(api_key, prompt_sys, prompt_user, num_return_sequences=1, **kwargs):
    results = []
    for _ in range(num_return_sequences):
        generated_text = call_deepseek_api(api_key, prompt_sys, prompt_user, **kwargs)
        if generated_text:
            results.append(generated_text)
        else:
            results.append("")
    return results

def prompt_add_mistakes_deepseek(api_key, questions, CoTs, seed=1):
    preprocess_CoTs = CoTs_preprocess(CoTs)

    cots_sentences = split_all_cots_into_sentences(preprocess_CoTs)

    # ========== select the sentence to modify ==========
    all_select_cot_sentences = []
    all_select_cot_sentences_index = []

    for i, cot_sentences in enumerate(cots_sentences):
        if len(cot_sentences) == 0:
            continue
        left_idx, right_idx = 0, len(cot_sentences) - 1
        if "Let's think step by step" in cot_sentences[0]:
            left_idx += 1
        if "answer is" in cot_sentences[-1] and right_idx > left_idx:
            right_idx -= 1
        idx = random.randint(left_idx, right_idx)
        all_select_cot_sentences_index.append(idx)
        all_select_cot_sentences.append(cot_sentences[idx])

    # ========== Add mistakes to the sentence ==========
    all_mistake_sentences = []

    for question, select_cot_sentence in zip(questions, all_select_cot_sentences):
        cur_seed = seed
        prompt_sys, prompt_user = create_adding_mistakes_prompt_deepseek(question, select_cot_sentence)
        texts = generate_text_deepseek(api_key, prompt_sys, prompt_user, num_return_sequences=1, max_tokens=512, temperature=0.7)
        mistake_sentence = texts[0].replace('\n', ' ')
        i = 0
        while mistake_sentence is None or mistake_sentence == "" or not isinstance(mistake_sentence, str):
            cur_seed += 1
            texts = generate_text_deepseek(api_key, prompt_sys, prompt_user, num_return_sequences=1, max_tokens=512, temperature=0.7)
            mistake_sentence = texts[0].replace('\n', ' ')
            i = i + 1
            if i == 20:
                mistake_sentence = select_cot_sentence
                break

        mistake_sentence = split_cot_into_sentences(mistake_sentence)[0]
        if "Sentence with mistake added:" in mistake_sentence:
            mistake_sentence = mistake_sentence.split("Sentence with mistake added:")[-1].strip()
        print('original sentence:', select_cot_sentence)
        print('adding mistakes sentence:', mistake_sentence)
        print('=' * 80)
        all_mistake_sentences.append(mistake_sentence)

    print("len(all_mistake_sentences):", len(all_mistake_sentences))
    return cots_sentences, all_select_cot_sentences_index, all_mistake_sentences