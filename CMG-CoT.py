import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import argparse
import math
from collections import Counter
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from utils import *
from prompts.extract_keyword_prompt import extract_keyword_prompt
from prompts.few_shot_cot_prompt import create_ai2_arc_prompt, create_aqua_prompt, create_StrategyQA_prompt
from prompts.generate_counterfactual_problem_prompt import create_StrategyQA_generate_counterfactual_problem_prompt, \
    create_aqua_generate_counterfactual_problem_prompt, create_ai2_arc_generate_counterfactual_problem_prompt
from Adding_Mistakes_metric import prompt_add_mistakes_deepseek
from soft_prompt_tuning import prompt_tuning
from counterfactual_reasoning_metric import problem_difference, KeywordMatrix2D, evaluate_counterfactual_faithfulness
from datetime import datetime
import torch, gc
import pickle

def load_model():
    model_path = "./models/" + args.model_name
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        dtype=torch.float16,
        trust_remote_code=True
    )

    sampling_params = SamplingParams(temperature=0.7, top_p=0.9, max_tokens=512, stop=["</s>", "Human:", "Question:"])

    return model, tokenizer, sampling_params

def loading_dataset(dataset_name):
    if dataset_name == 'ai2_arc_challenge':
        dataset = load_dataset("./datasets/ai2_arc/ARC-Challenge")
        test_dataset = dataset['test']
        test_subset = test_dataset.shuffle().select(range(500))
    elif dataset_name == 'ai2_arc_easy':
        dataset = load_dataset("./datasets/ai2_arc/ARC-Easy")
        test_dataset = dataset['test']
        test_subset = test_dataset.shuffle().select(range(500))
    elif dataset_name == 'aqua':
        dataset = load_dataset("./datasets/AQuA-RAT/raw")
        test_dataset = dataset['test']
        test_subset = test_dataset
    elif dataset_name == 'StrategyQA':
        dataset = load_dataset("./datasets/StrategyQA/data")
        test_dataset = dataset['test']
        test_subset = test_dataset.shuffle().select(range(500))
    return test_subset

def get_few_shot_prompt(dataset_name, question):
    if dataset_name == 'ai2_arc_challenge' or dataset_name == 'ai2_arc_easy':
        prompt = create_ai2_arc_prompt(question)
    elif dataset_name == 'aqua':
        prompt = create_aqua_prompt(question)
    elif dataset_name == 'StrategyQA':
        prompt = create_StrategyQA_prompt(question)
    return prompt

def load_answer(dataset_name, generated_text):
    if dataset_name == 'ai2_arc_challenge' or dataset_name == 'ai2_arc_easy' or dataset_name == 'aqua':
        ans = extract_option_answer(generated_text)
    elif dataset_name == 'StrategyQA':
        ans = extract_yesno_answer(generated_text)
    return ans

def evaluate_performance(dataset_name, predictions, references):
    if dataset_name == 'ai2_arc_challenge' or dataset_name == 'ai2_arc_easy' or dataset_name == 'mmlu' or dataset_name == 'aqua':
        accuracy, correct, total = calculate_option_accuracy(predictions, references)
    elif dataset_name == 'StrategyQA':
        accuracy, correct, total = calculate_yesno_accuracy(predictions, references)
    return accuracy, correct, total

def probe_next_token_logprobs(model, prompt, candidate_token_ids, soft_prompt=None,
                              add_str="\nA: Let's think step by step.", device=None):
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    tokenizer = model.tokenizer if hasattr(model, "tokenizer") else globals().get("tokenizer", None)

    try:
        model_device = next(model.parameters()).device
    except StopIteration:
        model_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device is None:
        device = model_device
    else:
        device = model_device

    if add_str in prompt:
        split_index = prompt.rindex(add_str)
    else:
        split_index = len(prompt)
    prefix_text = prompt[:split_index]
    suffix_text = prompt[split_index:]

    embedding_layer = model.get_input_embeddings()
    hidden_size = embedding_layer.weight.shape[-1]
    embed_dtype = embedding_layer.weight.dtype

    prefix_tokens = tokenizer(prefix_text, add_special_tokens=False).input_ids
    suffix_tokens = tokenizer(suffix_text, add_special_tokens=False).input_ids
    insert_pos = len(prefix_tokens)
    input_ids_list = prefix_tokens + suffix_tokens
    if len(input_ids_list) == 0:
        input_ids_list = [tokenizer.eos_token_id]
    input_ids = torch.tensor(input_ids_list, dtype=torch.long, device=device).unsqueeze(0)
    base_embeds = embedding_layer(input_ids).to(embed_dtype).to(device)

    if not isinstance(soft_prompt, torch.Tensor):
        soft_prompt = torch.tensor(soft_prompt)
    soft_prompt = soft_prompt.detach().to(device)

    if soft_prompt.dim() == 1:
        soft_prompt_tensor = soft_prompt.unsqueeze(0).unsqueeze(0)  # [1,1,dim]
    elif soft_prompt.dim() == 2:
        soft_prompt_tensor = soft_prompt.unsqueeze(0)
    elif soft_prompt.dim() == 3 and soft_prompt.shape[0] == 1:
        soft_prompt_tensor = soft_prompt

    if soft_prompt_tensor.shape[-1] != hidden_size:
        proj = torch.nn.Linear(soft_prompt_tensor.shape[-1], hidden_size, bias=False).to(device).to(embed_dtype)
        in_dtype = proj.weight.dtype
        soft_prompt_tensor = soft_prompt_tensor.to(in_dtype)
        with torch.no_grad():
            soft_prompt_tensor = proj(soft_prompt_tensor)
        soft_prompt_tensor = soft_prompt_tensor.to(embed_dtype)
    else:
        soft_prompt_tensor = soft_prompt_tensor.to(embed_dtype)

    inputs_embeds = torch.cat([
        base_embeds[:, :insert_pos, :],
        soft_prompt_tensor,
        base_embeds[:, insert_pos:, :]
    ], dim=1).to(embed_dtype).to(device)

    max_len = getattr(model.config, "max_position_embeddings", None)
    if max_len is not None and inputs_embeds.shape[1] > max_len:
        inputs_embeds = inputs_embeds[:, -max_len:, :]

    attention_mask = torch.ones((1, inputs_embeds.shape[1]), dtype=torch.long, device=device)

    with torch.no_grad():
        outputs = model(inputs_embeds=inputs_embeds, attention_mask=attention_mask, return_dict=True)
    logits = outputs.logits[0, -1, :]  # (vocab_size,)
    log_probs = F.log_softmax(logits, dim=-1)
    return {tid: log_probs[tid].item() if 0 <= tid < log_probs.shape[0] else -1e9
            for tid in candidate_token_ids}

def generate_with_soft_prompt(model, tokenizer, prompt, soft_prompt,
                              sampling_params: SamplingParams, add_str="\nA: Let's think step by step.", device='cuda',
                              max_new_tokens=None, num_return_sequences=1, seed=None):
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    stop_strings = ["</s>", "Human:", "Question:", "Q:"]
    stopping_criteria = StoppingCriteriaList([StopStringCriteria(stop_strings, tokenizer)])

    split_index = prompt.rindex(add_str)

    prefix_text = prompt[:split_index]
    suffix_text = prompt[split_index:]

    embed_dtype = model.get_input_embeddings().weight.dtype

    prefix_tokens = tokenizer(prefix_text, add_special_tokens=False).input_ids
    suffix_tokens = tokenizer(suffix_text, add_special_tokens=False).input_ids

    insert_pos = len(prefix_tokens)
    input_ids = prefix_tokens + suffix_tokens
    input_ids = torch.tensor(input_ids, dtype=torch.long, device=device).unsqueeze(0)

    base_embeds = model.get_input_embeddings()(input_ids).to(embed_dtype)

    soft_prompt_tensor = soft_prompt.unsqueeze(0).to(embed_dtype)
    inputs_embeds = torch.cat([
        base_embeds[:, :insert_pos, :],
        soft_prompt_tensor,
        base_embeds[:, insert_pos:, :]
    ], dim=1)

    base_len = base_embeds.shape[1]
    soft_len = soft_prompt_tensor.shape[1]
    total_len = base_len + soft_len

    attention_mask = torch.ones(
        (1, total_len),
        dtype=torch.long,
        device=device
    )

    gen_cfg = GenerationConfig(
        temperature=sampling_params.temperature,
        top_p=sampling_params.top_p,
        do_sample=True,
        max_new_tokens=sampling_params.max_tokens if max_new_tokens is None else max_new_tokens,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id
    )

    with torch.no_grad():
        outputs = model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            generation_config=gen_cfg,
            num_return_sequences=num_return_sequences,
            return_dict_in_generate=True,
            output_scores=False,
            stopping_criteria=stopping_criteria
        )

    texts = []
    for seq in outputs.sequences:
        text = tokenizer.decode(seq, skip_special_tokens=True)
        temp_prompt = prompt.replace('<s>', '')
        text = text.replace('<s>', '')
        if text.startswith(temp_prompt):
            text = text[len(temp_prompt):]
        text = text.split('Q:')[0]
        if '[/INST]' in text:
            text = text.split('[/INST]')[-1]
        texts.append(text.strip())
    return texts



def CMG_CoT(dataset_name, test_subset, part_of_cot=None, my_soft_prompts=None, add_str="\nA: Let's think step by step.",
                    soft_prompt_len=1, soft_prompt_steps=20, counterfactual_problems=None, num_samples_per_question=5, probe_string=" So, the answer is (", seed=1):

    predictions, references, CoTs, questions, all_soft_prompts = [], [], [], [], []
    for i, example in enumerate(test_subset):

        # =================================== Extract the keywords ===================================
        if counterfactual_problems is None:
            question, choice, correct_answer, target_tokens, target_token_ids = load_question_choice(dataset_name, example, tokenizer)
        else:
            question = counterfactual_problems['question'][i]
            choice = counterfactual_problems['choice'][i]

        token_importance_prompt = extract_keyword_prompt(dataset_name, question, choice)
        cur_seed = seed
        texts = generate_text(model, token_importance_prompt, sampling_params, tokenizer, num_return_sequences=1, important_word=True, seed=cur_seed)
        generated_text = texts[0].replace('\n', ' ')
        count = 0
        while generated_text is None or generated_text == "" or not isinstance(generated_text, str) or not re.findall(r'\d+\.\s*([^;]+?)(?=;|\s*\d+\.|$)', generated_text):
            cur_seed += 1
            texts = generate_text(model, token_importance_prompt, sampling_params, tokenizer, num_return_sequences=1, important_word=True,
                                  seed=cur_seed)
            generated_text = texts[0].replace('\n', ' ')
            count += 1
            if count == 20 and (generated_text is None or generated_text == "" or not isinstance(generated_text, str)):
                generated_text = 'There is no answer.'
                break
        words = re.findall(r'\d+\.\s*([^;]+?)(?=;|\s*\d+\.|$)', generated_text)
        important_words = words[:5]
        print('Important Words:', important_words)
        # ============================================================================================

        # ========================== Load the current question and prompt ============================
        if counterfactual_problems is None:
            question, correct_answer, target_tokens, target_token_ids = load_question(dataset_name, example, tokenizer)
            prompt = get_few_shot_prompt(dataset_name, question)
            if part_of_cot is not None:
                prompt = prompt.replace('[/INST]', part_of_cot[i] + ' [/INST]')
        else:
            question = counterfactual_problems['question'][i]
            correct_answer = counterfactual_problems['answer'][i]
            target_tokens = counterfactual_problems['target_tokens'][i]
            target_token_ids = counterfactual_problems['target_token_ids'][i]
            choices = counterfactual_problems['choice'][i]

            if choices is None:
                prompt = get_few_shot_prompt(dataset_name, question)
            else:
                prompt = get_few_shot_prompt(dataset_name, question + choices)
            if part_of_cot is not None:
                prompt = prompt.replace('[/INST]', part_of_cot[i] + ' [/INST]')
        # ============================================================================================

        # ================================ optimize the soft prompt ==================================
        all_answers = []
        all_cots = []
        cur_seed = seed
        if my_soft_prompts is None:
            soft_prompt, has_optimized = prompt_tuning(model, tokenizer, prompt, add_str=add_str, soft_prompt_len=soft_prompt_len, steps=soft_prompt_steps, seed=cur_seed)
            all_soft_prompts.append(soft_prompt)
        else:
            soft_prompt = my_soft_prompts[i]
        # ============================================================================================

        # =================================== predict the answer =====================================
        generated_texts = generate_with_soft_prompt(model, tokenizer, prompt, soft_prompt, sampling_params, num_return_sequences=num_samples_per_question,
                              add_str=add_str, device='cuda', seed=cur_seed)

        for j in range(len(generated_texts)):
            generated_text = generated_texts[j].replace('\n', ' ')
            count = 0
            while generated_text is None or generated_text == "" or not isinstance(generated_text, str):
                if part_of_cot is None:
                    cur_seed += 1
                    temp_texts = generate_with_soft_prompt(model, tokenizer, prompt, soft_prompt, sampling_params, num_return_sequences=1,
                              add_str=add_str, device='cuda', seed=cur_seed)
                    generated_text = temp_texts[0].replace('\n', ' ')
                    count += 1
                else:
                    nltk_part_of_cot = split_cot_into_sentences(part_of_cot[i])
                    if "answer is" in nltk_part_of_cot[-1]:
                        generated_text = part_of_cot[i]
                        break
                    else:
                        cur_seed += 1
                        temp_texts = generate_with_soft_prompt(model, tokenizer, prompt, soft_prompt, sampling_params,
                                                               num_return_sequences=1,
                                                               add_str=add_str, device='cuda', seed=cur_seed)
                        generated_text = temp_texts[0].replace('\n', ' ')
                        count += 1
                if count == 10 and (generated_text is None or generated_text == "" or not isinstance(generated_text, str)):
                    if part_of_cot is None:
                        generated_text = 'There is no answer.'
                        break
                    else:
                        generated_text = part_of_cot[i]
                        break

            if '[/INST]' in generated_text or '</INST>' in generated_text or '</SYS>' in generated_text:
                if '[/INST]' in generated_text:
                    temp_generated_text = generated_text.split('[/INST]')
                elif '</INST>' in generated_text:
                    temp_generated_text = generated_text.split('</INST>')
                elif '</SYS>' in generated_text:
                    temp_generated_text = generated_text.split('</SYS>')
                has_assign = 0
                for temp_i in range(len(temp_generated_text)):
                    if 'answer' in temp_generated_text[temp_i] and 'You are a helpful' not in temp_generated_text[temp_i]:
                        generated_text = temp_generated_text[temp_i]
                        has_assign = 1
                        break
                if has_assign == 0:
                    generated_text = temp_generated_text[0]
                print('=' * 80)

            all_cots.append(generated_text)

            predicted_answer = load_answer(dataset_name, generated_text)
            if predicted_answer:
                all_answers.append(predicted_answer)
            else:
                all_answers.append('')
        # ============================================================================================

        # ================================== select the final cot ====================================
        if len(all_answers) == 0:
            select_answer = ""
            select_CoT = all_cots[0] if all_cots else ""
            cots_probe_trend = [[] for _ in range(len(all_cots))]
        else:
            cots_sentences = split_all_cots_into_sentences(all_cots)

            cots_probe_matrix = []
            for i_cot in range(len(all_cots)):
                each_cot_probe_matrix = []

                # calculate p_pre
                initial_prompt = prompt.replace('[/INST]', probe_string + '[/INST]')
                logprob_map = probe_next_token_logprobs(model, initial_prompt, target_token_ids, soft_prompt)
                probs = [logprob_map.get(tid, -1e9) for tid in target_token_ids]
                each_cot_probe_matrix.append(probs)

                # calculate p_post
                post_prompt = prompt
                post_prompt = post_prompt.replace('[/INST]', " " + all_cots[i_cot] + '[/INST]')
                post_prompt = post_prompt.replace('[/INST]', probe_string + '[/INST]')
                logprob_map = probe_next_token_logprobs(model, post_prompt, target_token_ids, soft_prompt)
                probs = [logprob_map.get(tid, -1e9) for tid in target_token_ids]
                each_cot_probe_matrix.append(probs)

                cots_probe_matrix.append(each_cot_probe_matrix)

            # calculate the score of each cot
            cot_scores = []
            cots_probe_trend = []
            for k in range(num_samples_per_question):
                if all_answers[k] in target_tokens:
                    answer_idx = target_tokens.index(all_answers[k])
                else:
                    answer_idx = -1
                if answer_idx == -1:
                    cot_scores.append(-1)
                    cots_probe_trend.append([])
                else:
                    current_probe_probability_score = []
                    for j in range(len(cots_probe_matrix[k])):
                        p = math.exp(cots_probe_matrix[k][j][answer_idx])
                        current_probe_probability_score.append(p)
                    cots_probe_trend.append(current_probe_probability_score)

                    current_score = current_probe_probability_score[-1]

                    temp_contain_score = 0.0
                    if len(important_words) > 0:
                        for word in important_words:
                            if word in all_cots[k]:
                                temp_contain_score += 1
                        temp_contain_score /= len(important_words)
                    current_score = current_score + temp_contain_score * 0.5

                    cot_scores.append(current_score)

            select_CoT_idx = int(np.argmax(cot_scores))
            select_CoT = all_cots[select_CoT_idx]
            select_answer = all_answers[select_CoT_idx]

        questions.append(question)
        CoTs.append(select_CoT)
        predictions.append(select_answer)
        references.append(correct_answer)

        # ============================================================================================

        print(f"Progress: {i + 1}/{len(test_subset)}")
        print(f"Question: {question}")
        print(f"Sampling Answers: {dict(Counter(all_answers))}")
        print(f"Select Answer: {select_answer}")
        print(f"CoT : {select_CoT}")
        print("Probe trend of the select CoT:", cots_probe_trend[select_CoT_idx])
        print("Function score of the select CoT: ", cot_scores[select_CoT_idx])
        print(f"Correct Answer: {correct_answer}")
        print("-" * 80)

        gc.collect()
        torch.cuda.empty_cache()

    return predictions, references, CoTs, questions, all_soft_prompts

def get_counterfactual_problem_prompt(dataset_name, question, choice, correct_answer):
    if dataset_name == 'ai2_arc_challenge' or dataset_name == 'ai2_arc_easy':
        prompt_sys, prompt_user = create_ai2_arc_generate_counterfactual_problem_prompt(question, choice,
                                                                                        correct_answer)
    elif dataset_name == 'aqua':
        prompt_sys, prompt_user = create_aqua_generate_counterfactual_problem_prompt(question, choice, correct_answer)
    elif dataset_name == 'StrategyQA':
        prompt_sys, prompt_user = create_StrategyQA_generate_counterfactual_problem_prompt(question, choice,
                                                                                           correct_answer)
    return prompt_sys, prompt_user

def call_deepseek_api(prompt_sys, prompt_user, max_tokens=512, temperature=0.7, model="deepseek-chat"):
    from openai import OpenAI
    client = OpenAI(api_key=args.deepseek_api_key, base_url="https://api.deepseek.com")

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

def generate_text_deepseek(prompt_sys, prompt_user, num_return_sequences=1, **kwargs):

    results = []
    for _ in range(num_return_sequences):
        generated_text = call_deepseek_api(prompt_sys, prompt_user, **kwargs)
        if generated_text:
            results.append(generated_text)
        else:
            results.append("")
    return results

def counterfactual_problem_generation_deepseek(dataset_name, test_subset, seed=1):
    all_counterfactual_problems = []
    all_counterfactual_answers = []

    for i, example in enumerate(test_subset):
        question, choice, correct_answer, _, _ = load_question_choice(dataset_name, example, tokenizer)
        prompt_sys, prompt_user = get_counterfactual_problem_prompt(dataset_name, question, choice, correct_answer)
        texts = generate_text_deepseek(prompt_sys, prompt_user, num_return_sequences=1, max_tokens=512, temperature=0.7)
        generated_text = texts[0].replace('\n', ' ')
        if 'Counterfactual Answer:' in generated_text:
            counterfactual_problem = generated_text.split('Counterfactual Answer:')
            counterfactual_answer = counterfactual_problem[1]
            counterfactual_problem = counterfactual_problem[0]
            if 'Counterfactual Problem:' in counterfactual_problem:
                counterfactual_problem = counterfactual_problem.replace('Counterfactual Problem:', '').strip()
        else:
            counterfactual_problem = None
        count = 0
        while generated_text is None or generated_text == "" or not isinstance(generated_text, str) or counterfactual_problem is None or counterfactual_problem == "" or counterfactual_answer == "":
            texts = generate_text_deepseek(prompt_sys, prompt_user, num_return_sequences=1, max_tokens=512, temperature=0.7)
            generated_text = texts[0].replace('\n', ' ')
            if 'Counterfactual Answer:' in generated_text:
                counterfactual_problem = generated_text.split('Counterfactual Answer:')
                counterfactual_answer = counterfactual_problem[1]
                counterfactual_problem = counterfactual_problem[0]
                if 'Counterfactual Problem:' in counterfactual_problem:
                    counterfactual_problem = counterfactual_problem.replace('Counterfactual Problem:', '').strip()
            else:
                counterfactual_problem = None
            count += 1
            if (count == 20 and (generated_text is None or generated_text == "" or not isinstance(generated_text, str))) or count > 20:
                generated_text = 'There is no response.'
                counterfactual_problem = ''
                counterfactual_answer = ''
                break
        if dataset_name == 'ai2_arc_challenge' or dataset_name == 'ai2_arc_easy' or dataset_name == 'aqua':
            counterfactual_answer = extract_option_answer(counterfactual_answer)
        elif dataset_name == 'StrategyQA':
            counterfactual_answer = extract_yesno_answer(counterfactual_answer)
        print('original_problem:', question)
        print('original_answer:', correct_answer)
        print('counterfactual_problem:', counterfactual_problem)
        print('counterfactual_answer:', counterfactual_answer)
        print('=' * 80)

        all_counterfactual_problems.append(counterfactual_problem)
        all_counterfactual_answers.append(counterfactual_answer)
    return all_counterfactual_problems, all_counterfactual_answers


def evaluate_counterfactual(model, dataset_name, test_subset, tokenizer, sampling_params, seed=1):
    all_counterfactual_problems, all_counterfactual_answers = counterfactual_problem_generation_deepseek(dataset_name, test_subset, seed=1)

    all_counterfactual_choices = []
    all_target_tokens = []
    all_target_token_ids = []
    for i, example in enumerate(test_subset):
        question, choice, correct_answer, target_tokens, target_token_ids = load_question_choice(dataset_name, example, tokenizer)
        all_counterfactual_choices.append(choice)
        all_target_tokens.append(target_tokens)
        all_target_token_ids.append(target_token_ids)
    counterfactual_problems = {}
    counterfactual_problems['question'] = all_counterfactual_problems
    counterfactual_problems['answer'] = all_counterfactual_answers
    counterfactual_problems['choice'] = all_counterfactual_choices
    counterfactual_problems['target_tokens'] = all_target_tokens
    counterfactual_problems['target_token_ids'] = all_target_token_ids

    (counterfactual_predictions, counterfactual_references, counterfactual_CoTs, counterfactual_questions, counterfactual_all_soft_prompts) = CMG_CoT(
        dataset_name, test_subset, num_samples_per_question=5, counterfactual_problems=counterfactual_problems)

    problem_difference_indexes, all_differences = problem_difference(questions, counterfactual_questions, predictions,
                                                                     counterfactual_predictions, references,
                                                                     counterfactual_references)

    processor = KeywordMatrix2D(preprocess_text=True, remove_stopwords=True)
    keyword_matrix_2d = processor.create_2d_keyword_matrix(all_differences)

    counterfactual_faithfulness_metric, all_signs = evaluate_counterfactual_faithfulness(counterfactual_predictions,
                                                                              counterfactual_references,
                                                                              counterfactual_CoTs,
                                                                              counterfactual_questions,
                                                                              problem_difference_indexes,
                                                                              keyword_matrix_2d)
    print('counterfactual_faithfulness_metric:', counterfactual_faithfulness_metric)

    return all_counterfactual_problems, all_counterfactual_answers, counterfactual_predictions, counterfactual_CoTs, counterfactual_faithfulness_metric

def evaluate_adding_mistakes(dataset_name, questions, CoTs):
    cots_sentences, all_select_cot_sentences_index, all_mistake_sentences = prompt_add_mistakes_deepseek(args.deepseek_api_key, questions, CoTs)

    mistake_CoTs = []
    for i, (index, mistake_sentence) in enumerate(zip(all_select_cot_sentences_index, all_mistake_sentences)):
        mistake_CoT = ""
        for j, sentence in enumerate(cots_sentences[i]):
            if j == index:
                mistake_CoT += mistake_sentence
                mistake_CoT += " "
                break
            else:
                mistake_CoT += sentence
                mistake_CoT += " "

        mistake_CoTs.append(mistake_CoT)
        print(mistake_CoT)
        print("-" * 80)


    (add_mistakes_predictions, add_mistakes_references, add_mistakes_CoTs, add_mistakes_questions, add_mistakes_all_soft_prompts) = CMG_CoT(dataset_name,
                                                                                                     test_subset,
                                                                                                     num_samples_per_question=5,
                                                                                                     part_of_cot=mistake_CoTs,
                                                                                                     my_soft_prompts=all_soft_prompts)
    return mistake_CoTs, add_mistakes_predictions, add_mistakes_CoTs

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Relationships between problem difficulty/F_c and Adding Mistakes')
    parser.add_argument('--dataset', '-d', type=str)
    parser.add_argument('--model_name', '-m', type=str)
    parser.add_argument('--deepseek_api_key', '-api', type=str)
    parser.add_argument('--evaluate_adding_mistakes', '-am', type=bool, default=True, help='whether to evaluate adding mistakes')
    parser.add_argument('--evaluate_counterfactual_reasoning', '-cr', type=bool, default=True, help='whether to evaluate counterfactual reasoning')
    parser.add_argument('--save_result', type=bool, default=True, help='save the calculation results')
    args = parser.parse_args()

    formatted_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # ======================= Load the model and dataset ============================
    model, tokenizer, sampling_params = load_model()  # load the model
    model.eval()
    test_subset = loading_dataset(args.dataset)  # load the dataset
    # ===============================================================================

    # CMG-CoT to predict the results
    predictions, references, CoTs, questions, all_soft_prompts = CMG_CoT(args.dataset, test_subset, num_samples_per_question=5)
    if args.save_result:
        file_path = './Results/' + args.model_name + "_" + args.dataset + "_CoTs_" + formatted_time + '.pkl'
        with open(file_path, 'wb') as f:
            pickle.dump(CoTs, f)
        file_path = './Results/' + args.model_name + "_" + args.dataset + "_predictions_" + formatted_time + '.pkl'
        with open(file_path, 'wb') as f:
            pickle.dump(predictions, f)


    accuracy, correct, total = evaluate_performance(args.dataset, predictions, references)

    print("\n" + "=" * 50)
    print(f"Total Samples: {total}")
    print(f"Correct Number: {correct}")
    print(f"Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")
    print("=" * 50)

    # ================== Counterfactual Reasoning =================================
    if args.evaluate_counterfactual_reasoning:
        all_counterfactual_problems, all_counterfactual_answers, counterfactual_predictions, counterfactual_CoTs, counterfactual_faithfulness_metric \
            = evaluate_counterfactual(model, args.dataset, test_subset, tokenizer, sampling_params, seed=1)
        if args.save_result:
            file_path = './Results/' + args.model_name + "_" + args.dataset + "_counterfactual_predictions_" + formatted_time + '.pkl'
            with open(file_path, 'wb') as f:
                pickle.dump(counterfactual_predictions, f)
            file_path = './Results/' + args.model_name + "_" + args.dataset + "_counterfactual_CoTs_" + formatted_time + '.pkl'
            with open(file_path, 'wb') as f:
                pickle.dump(counterfactual_CoTs, f)

    # ===============================================================================

    # ===============================  Adding Mistakes ==============================
    if args.evaluate_adding_mistakes:
        mistake_CoTs, add_mistakes_predictions, add_mistakes_CoTs = evaluate_adding_mistakes(args.dataset, questions, CoTs)
        adding_mistakes_metric = np.sum(np.array(predictions) != np.array(add_mistakes_predictions)) / len(
            predictions)
        print('Adding Mistakes Metric:', adding_mistakes_metric)

        if args.save_result:
            file_path = './Results/' + args.model_name + "_" + args.dataset + "_mistake_CoTs_" + formatted_time + '.pkl'
            with open(file_path, 'wb') as f:
                pickle.dump(mistake_CoTs, f)
            file_path = './Results/' + args.model_name + "_" + args.dataset + "_add_mistakes_predictions_" + formatted_time + '.pkl'
            with open(file_path, 'wb') as f:
                pickle.dump(add_mistakes_predictions, f)
            file_path = './Results/' + args.model_name + "_" + args.dataset + "_add_mistakes_CoTs_" + formatted_time + '.pkl'
            with open(file_path, 'wb') as f:
                pickle.dump(add_mistakes_CoTs, f)

    # ===============================================================================

    print("\n" + "=" * 50)
    print('Dataset:', args.dataset)
    print(f"Total Samples: {total}")
    print(f"Correct Number {correct}")
    print(f"Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")
    if args.evaluate_counterfactual_reasoning:
        print('counterfactual_faithfulness_metric:', counterfactual_faithfulness_metric, ' (Lower is better)')
    if args.evaluate_adding_mistakes:
        print('Adding Mistakes Metric:', adding_mistakes_metric, ' (Higher is better)')
    print("=" * 50)