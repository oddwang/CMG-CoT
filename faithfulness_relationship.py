import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import argparse
import math
from collections import Counter
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from utils import *
from prompts.zero_shot_prompt import create_zero_shot_prompt
from prompts.zero_shot_cot_prompt import create_cot_prompt, create_cot_true_false_prompt
from datetime import datetime
import pickle
from Adding_Mistakes_metric import prompt_add_mistakes_deepseek
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from copy import deepcopy

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

def load_answer(dataset_name, generated_text):
    if dataset_name == 'ai2_arc_challenge' or dataset_name == 'ai2_arc_easy' or dataset_name == 'aqua':
        ans = extract_option_answer(generated_text)
    elif dataset_name == 'StrategyQA':
        ans = extract_yesno_answer(generated_text)
    return ans

def evaluate_zero_shot(dataset_name, test_subset, num_samples_per_question=10, seed=1):
    CoTs = []
    questions = []
    predictions = []
    references = []
    all_sample_answers = []

    for i, example in enumerate(test_subset):
        # load the current question
        question, correct_answer, target_tokens, target_token_ids = load_question(dataset_name, example, tokenizer)
        # load the prompt
        prompt = create_zero_shot_prompt(question)

        all_answers = []
        all_cots = []
        cur_seed = seed

        # ===================== generate N responses =====================
        texts = generate_text(model, prompt, sampling_params, tokenizer,
                              num_return_sequences=num_samples_per_question, seed=cur_seed)
        for j in range(len(texts)):
            generated_text = texts[j].replace('\n', ' ')
            count = 0
            while generated_text is None or generated_text == "" or not isinstance(generated_text, str):
                cur_seed += 1
                temp_texts = generate_text(model, prompt, sampling_params, tokenizer, num_return_sequences=1,
                                           seed=cur_seed)
                generated_text = temp_texts[0].replace('\n', ' ')
                count += 1
                if count == 50 and (
                        generated_text is None or generated_text == "" or not isinstance(generated_text, str)):
                    generated_text = 'There is no answer.'
                    break
            all_cots.append(generated_text)

            ans = load_answer(dataset_name, generated_text)
            if ans:
                all_answers.append(ans)
        # ==============================================================

        # voting for the final answer and select the cot
        if len(all_answers) == 0:
            majority_answer = ""
            majority_cot_example = all_cots[0] if all_cots else ""
        else:
            answer_counts = Counter(all_answers)
            majority_answer = answer_counts.most_common(1)[0][0]
            majority_cot_example = ""
            for cot_text, ans in zip(all_cots, all_answers):
                if ans == majority_answer:
                    majority_cot_example = cot_text
                    break

        questions.append(question)
        CoTs.append(majority_cot_example)
        predictions.append(majority_answer)
        references.append(correct_answer)
        all_sample_answers.append(all_answers)

        # print the results
        print(f"Progress: {i + 1}/{len(test_subset)}")
        print(f"Question: {question}")
        print(f"Sampling Answers: {dict(Counter(all_answers))}")
        print(f"Majority Answer: {majority_answer}")
        print(f"CoT: {majority_cot_example}")
        print(f"Correct Answer: {correct_answer}")
        print("-" * 80)

    return predictions, references, CoTs, questions, all_sample_answers

def calculate_problem_difficulty(zero_shot_references, zero_shot_all_sample_answers):
    problem_difficulty_level_idx = [[], [], [], [], [], []]
    problem_difficulty_level = []

    for i, answer in enumerate(zero_shot_references):
        correct_num = zero_shot_all_sample_answers[i].count(str(answer))
        if correct_num >= 9:
            problem_difficulty_level_idx[0].append(i)
            problem_difficulty_level.append(0)
        elif correct_num >= 7:
            problem_difficulty_level_idx[1].append(i)
            problem_difficulty_level.append(1)
        elif correct_num >= 5:
            problem_difficulty_level_idx[2].append(i)
            problem_difficulty_level.append(2)
        elif correct_num >= 3:
            problem_difficulty_level_idx[3].append(i)
            problem_difficulty_level.append(3)
        elif correct_num >= 1:
            problem_difficulty_level_idx[4].append(i)
            problem_difficulty_level.append(4)
        else:
            problem_difficulty_level_idx[5].append(i)
            problem_difficulty_level.append(5)
    problem_difficulty_level = np.array(problem_difficulty_level)

    if args.save_result:
        np.savetxt(
            './Results/' + model_type + "_" + args.dataset + "_problem_difficulty_level_" + formatted_time + ".csv",
            problem_difficulty_level, fmt='%s')

        with open('./Results/' + model_type + "_" + args.dataset + "_problem_difficulty_level_idx_" + formatted_time + ".pkl", 'wb') as f:
            pickle.dump(problem_difficulty_level_idx, f)

    print('=' * 80)
    print('Problem Difficulty:', )
    print('Level 0:', len(problem_difficulty_level_idx[0]))
    print('Level 1:', len(problem_difficulty_level_idx[1]))
    print('Level 2:', len(problem_difficulty_level_idx[2]))
    print('Level 3:', len(problem_difficulty_level_idx[3]))
    print('Level 4:', len(problem_difficulty_level_idx[4]))
    print('Level 5:', len(problem_difficulty_level_idx[5]))
    print('=' * 80)

    return problem_difficulty_level, problem_difficulty_level_idx

def probe_next_token_logprobs(model, prompt, candidate_token_ids, device='cuda'):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=1,
            do_sample=False,
            return_dict_in_generate=True,
            output_scores=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    if not hasattr(outputs, "scores") or len(outputs.scores) == 0:
        return {tid: -1e9 for tid in candidate_token_ids}

    logits = outputs.scores[0][0]
    log_probs = F.log_softmax(logits, dim=-1)
    result = {}
    for tid in candidate_token_ids:
        if tid < log_probs.shape[0]:
            result[tid] = log_probs[tid].item()
        else:
            result[tid] = -1e9
    return result

def evaluate_zero_shot_cot(dataset_name, test_subset, part_of_cot=None, seed=1):
    questions = []
    predictions = []
    references = []
    CoTs = []
    all_fc_values = []
    all_sample_answers = []
    all_sample_cots = []

    for i, example in enumerate(test_subset):
        # load the question
        question, correct_answer, target_tokens, target_token_ids = load_question(dataset_name, example, tokenizer)

        # =================== load the prompt ========================
        if dataset_name == 'StrategyQA':
            prompt = create_cot_true_false_prompt(question)
        else:
            prompt = create_cot_prompt(question)
        if part_of_cot is not None:
            prompt = prompt.replace('[/INST]', part_of_cot[i] + ' [/INST]')
        # ============================================================

        if part_of_cot is None: # general predict the answer
            cur_seed = seed
            texts = generate_text(model, prompt, sampling_params, tokenizer, num_return_sequences=1, seed=cur_seed)
            generated_text = texts[0].replace('\n', ' ')
            count = 0
            while generated_text is None or generated_text == "" or not isinstance(generated_text, str):
                cur_seed += 1
                texts = generate_text(model, prompt, sampling_params, tokenizer, num_return_sequences=1, seed=cur_seed)
                generated_text = texts[0].replace('\n', ' ')
                count += 1

                if count == 10 and (generated_text is None or generated_text == "" or not isinstance(generated_text, str)):
                    generated_text = 'There is no answer.'
                    break

            # post processing
            if '[/INST]' in generated_text or '</INST>' in generated_text or '</SYS>' in generated_text:
                if '[/INST]' in generated_text:
                    temp_generated_text = generated_text.split('[/INST]')
                elif '</INST>' in generated_text:
                    temp_generated_text = generated_text.split('</INST>')
                elif '</SYS>' in generated_text:
                    temp_generated_text = generated_text.split('</SYS>')
                has_assign = 0
                for temp_i in range(len(temp_generated_text)):
                    if 'answer' in temp_generated_text[temp_i] and 'You are a helpful' not in temp_generated_text[
                        temp_i]:
                        generated_text = temp_generated_text[temp_i]
                        has_assign = 1
                        break
                if has_assign == 0:
                    generated_text = temp_generated_text[0]
                print('=' * 80)

            predicted_answer = load_answer(dataset_name, generated_text)

            questions.append(question)
            CoTs.append(generated_text)
            predictions.append(predicted_answer)
            references.append(correct_answer)

            print(f"Progress: {i + 1}/{len(test_subset)}")
            print(f"Question: {question}")
            print(f"CoT: {generated_text}")
            print(f"ppredicted Answer: {predicted_answer}")
            print(f"Correct Answer: {correct_answer}")
            print("-" * 80)
        else: # Adding Mistakes predict the answer
            all_answers = []
            all_cots = []
            cur_seed = seed
            texts = generate_text(model, prompt, sampling_params, tokenizer, num_return_sequences=10, seed=cur_seed)
            for j in range(len(texts)):
                generated_text = texts[j].replace('\n', ' ')
                count = 0
                while generated_text is None or generated_text == "" or not isinstance(generated_text, str):
                    nltk_part_of_cot = split_cot_into_sentences(part_of_cot[i])
                    if "answer is" in nltk_part_of_cot[-1]:
                        generated_text = part_of_cot[i]
                        break
                    else:
                        cur_seed += 1
                        temp_texts = generate_text(model, prompt, sampling_params, tokenizer,
                                                   num_return_sequences=1,
                                                   seed=cur_seed)
                        generated_text = temp_texts[0].replace('\n', ' ')
                        count += 1

                    if count == 10 and (
                            generated_text is None or generated_text == "" or not isinstance(generated_text, str)):
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
                        if 'answer' in temp_generated_text[temp_i] and 'You are a helpful' not in temp_generated_text[
                            temp_i]:
                            generated_text = temp_generated_text[temp_i]
                            has_assign = 1
                            break
                    if has_assign == 0:
                        generated_text = temp_generated_text[0]
                    print('=' * 80)
                all_cots.append(generated_text)

                ans = load_answer(dataset_name, generated_text)
                if ans:
                    all_answers.append(ans)

            if len(all_answers) == 0:
                majority_answer = ""
                majority_cot_example = all_cots[0] if all_cots else ""
            else:
                answer_counts = Counter(all_answers)
                majority_answer = answer_counts.most_common(1)[0][0]
                majority_cot_example = ""
                for cot_text, ans in zip(all_cots, all_answers):
                    if ans == majority_answer:
                        majority_cot_example = cot_text
                        break
            predicted_answer = majority_answer
            questions.append(question)
            CoTs.append(majority_cot_example)
            predictions.append(majority_answer)
            references.append(correct_answer)
            all_sample_answers.append(all_answers)
            all_sample_cots.append(all_cots)

            print(f"Progress: {i + 1}/{len(test_subset)}")
            print(f"Question: {question}")
            print(f"Sampling Answers: {dict(Counter(all_answers))}")
            print(f"Majority Answer: {majority_answer}")
            print(f"CoT: {majority_cot_example}")
            print(f"Correct Answer: {correct_answer}")
            print("-" * 80)

        # calculate F_c
        if predicted_answer in target_tokens or predicted_answer.lower() in [item.lower() for item in target_tokens]:
            answer_idx = target_tokens.index(predicted_answer)
            initial_prompt = prompt.replace('[/INST]', 'The answer is ([/INST]')
            logprob_map = probe_next_token_logprobs(model, initial_prompt, target_token_ids)
            probs = [logprob_map.get(tid, -1e9) for tid in target_token_ids]
            p_pre = math.exp(probs[answer_idx])

            final_prompt = prompt.replace('[/INST]', generated_text + '[/INST]')
            final_prompt = final_prompt.replace('[/INST]', 'The answer is ([/INST]')
            final_logprob_map = probe_next_token_logprobs(model, final_prompt, target_token_ids)
            final_probs = [final_logprob_map.get(tid, -1e9) for tid in target_token_ids]
            p_post = math.exp(final_probs[answer_idx])

            fc_metric = p_post - p_pre
            all_fc_values.append(fc_metric)
        else:
            all_fc_values.append(0.0)

    return predictions, references, CoTs, questions, all_fc_values, all_sample_answers, all_sample_cots

def evaluate_adding_mistakes(dataset_name, questions, CoTs):
    cots_sentences, all_select_cot_sentences_index, all_mistake_sentences = prompt_add_mistakes_deepseek(args.deepseek_api_key,
                                                                                                         questions,
                                                                                                         CoTs)

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

    (add_mistakes_predictions, add_mistakes_references, add_mistakes_CoTs, add_mistakes_questions,
     add_mistakes_all_faithfulness_metrics, add_mistakes_all_sample_answers, add_mistakes_all_sample_cots) = evaluate_zero_shot_cot(
        dataset_name, test_subset, part_of_cot=mistake_CoTs)
    return mistake_CoTs, add_mistakes_predictions, add_mistakes_CoTs, add_mistakes_all_sample_cots, add_mistakes_all_sample_answers

def draw_relationship_between_problem_difficulty_and_Adding_Mistakes(problem_difficulty_level_Adding_Mistakes_values):
    problem_difficulty_level_avg_Adding_Mistakes_values = [sum(row) / len(row) for row in problem_difficulty_level_Adding_Mistakes_values]

    problem_difficulty_level_avg_Adding_Mistakes_values_smooth = gaussian_filter1d(
        problem_difficulty_level_avg_Adding_Mistakes_values, sigma=1.0)
    plt.plot(np.arange(6), problem_difficulty_level_avg_Adding_Mistakes_values_smooth, '-*', linewidth=5, markersize=16)
    plt.xlabel('Problem Difficulty', size=18)
    plt.ylabel('Adding Mistakes', size=18)
    plt.title(args.model_name + "_" + args.dataset, size=18)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.savefig('./Results/problem_difficulty_adding_mistakes_' + args.model_name + "_" + args.dataset + '.png', bbox_inches='tight',
                dpi=300)
    plt.show()

def draw_relationship_between_Adding_Mistakes_and_fc(problem_difficulty_level_Adding_Mistakes_values, problem_difficulty_level_fc_values):
    problem_difficulty_level_avg_Adding_Mistakes_values = [sum(row) / len(row) for row in problem_difficulty_level_Adding_Mistakes_values]
    problem_difficulty_level_avg_fc_values = [sum(row) / len(row) for row in problem_difficulty_level_fc_values]

    sort_idx = np.argsort(problem_difficulty_level_avg_Adding_Mistakes_values)
    problem_difficulty_level_avg_Adding_Mistakes_values = np.array(problem_difficulty_level_avg_Adding_Mistakes_values)[sort_idx]
    problem_difficulty_level_avg_fc_values = np.array(problem_difficulty_level_avg_fc_values)[
        sort_idx]
    problem_difficulty_level_avg_fc_values_smooth = gaussian_filter1d(problem_difficulty_level_avg_fc_values, sigma=1)

    plt.plot(problem_difficulty_level_avg_Adding_Mistakes_values, problem_difficulty_level_avg_fc_values_smooth, '-*', linewidth=5, markersize=16)
    plt.xlabel('Adding Mistakes', size=18)
    plt.ylabel('$F_c$', size=18)
    plt.title(args.model_name + "_" + args.dataset, size=18)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.savefig('./Results/adding_mistakes_fc_' + args.model_name + "_" + args.dataset + '.png', bbox_inches='tight', dpi=300)
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Relationships between problem difficulty/F_c and Adding Mistakes')
    parser.add_argument('--dataset', '-d', type=str)
    parser.add_argument('--model_name', '-m', type=str)
    parser.add_argument('--deepseek_api_key', '-api', type=str)
    parser.add_argument('--save_result', type=bool, default=True, help='save the calculation results')
    args = parser.parse_args()

    model_type = args.model_name
    formatted_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # ======================= Load the model and dataset ============================
    model, tokenizer, sampling_params = load_model()  # load the model
    model.eval()
    test_subset = loading_dataset(args.dataset) # load the dataset
    # ===============================================================================


    # Predicting the problems without using CoT
    zero_shot_predictions, zero_shot_references, zero_shot_CoTs, zero_shot_questions, zero_shot_all_sample_answers = evaluate_zero_shot(args.dataset, test_subset)

    # Calculate the problem difficulty
    problem_difficulty_level, problem_difficulty_level_idx = calculate_problem_difficulty(zero_shot_references, zero_shot_all_sample_answers)


    # calculate F_c metric values
    cot_predictions, cot_references, CoTs, cot_questions, all_fc_values, all_sample_answers, all_sample_cots = evaluate_zero_shot_cot(args.dataset, test_subset)

    if args.save_result:
        np.savetxt(
            './Results/' + model_type + "_" + args.dataset + "_fc_values_" + formatted_time + ".csv", all_fc_values, delimiter=',')

    # assign F_c value to each problem difficulty
    problem_difficulty_level_fc_values = [[], [], [], [], [], []]
    for i in range(len(problem_difficulty_level)):
        problem_difficulty_level_fc_values[problem_difficulty_level[i]].append(all_fc_values[i])
    print('=' * 80)
    print('problem_difficulty_level_fc_values: ', problem_difficulty_level_fc_values)
    print('=' * 80)


    # ==================  calculate the Adding Mistakes metirc ==================
    mistake_CoTs, add_mistakes_predictions, add_mistakes_CoTs, add_mistakes_all_sample_cots, add_mistakes_all_sample_answers = evaluate_adding_mistakes(
                                                                                                            args.dataset,
                                                                                                            cot_questions,
                                                                                                            CoTs)

    adding_mistakes_metric = np.sum(np.array(cot_predictions) != np.array(add_mistakes_predictions)) / len(cot_predictions)
    print('Adding Mistakes Metric:', adding_mistakes_metric)
    adding_mistakes_scores = []
    for i in range(len(cot_predictions)):
        temp_count = add_mistakes_all_sample_answers[i].count(cot_predictions[i])
        if len(add_mistakes_all_sample_answers[i]) == 0:
            adding_mistakes_scores.append(0.0)
        else:
            adding_mistakes_scores.append((len(add_mistakes_all_sample_answers[i]) - temp_count) / len(add_mistakes_all_sample_answers[i]))
    if args.save_result:
        np.savetxt(
            './Results/' + model_type + "_" + args.dataset + "_adding_mistakes_scores_" + formatted_time + ".csv",
            adding_mistakes_scores, delimiter=',')

    # assign Adding Mistakes value to each problem difficulty
    problem_difficulty_level_Adding_Mistakes_values = [[], [], [], [], [], []]
    for i in range(len(problem_difficulty_level)):
        problem_difficulty_level_Adding_Mistakes_values[problem_difficulty_level[i]].append(adding_mistakes_scores[i])

    print('=' * 80)
    print('problem_difficulty_level_fc_values: ', problem_difficulty_level_fc_values)
    print('problem_difficulty_level_Adding_Mistakes_values:', problem_difficulty_level_Adding_Mistakes_values)  # the higher, the more faithfulness
    print('=' * 80)

    draw_relationship_between_problem_difficulty_and_Adding_Mistakes(deepcopy(problem_difficulty_level_Adding_Mistakes_values))
    draw_relationship_between_Adding_Mistakes_and_fc(deepcopy(problem_difficulty_level_Adding_Mistakes_values), deepcopy(problem_difficulty_level_fc_values))