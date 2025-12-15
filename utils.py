import torch
import numpy as np
from transformers import GenerationConfig
from transformers import StoppingCriteria, StoppingCriteriaList
import re
from nltk.tokenize import sent_tokenize

class SamplingParams:
    def __init__(self, temperature=0.7, top_p=0.9, max_tokens=512, stop=None, logprobs=False):
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.stop = stop or []
        self.logprobs = logprobs

def load_question(dataset_name, example, tokenizer):
    if dataset_name == 'ai2_arc_challenge' or dataset_name == 'ai2_arc_easy':
        question = example['question']
        choice = ""
        target_tokens_len = len(example['choices']['label'])
        target_tokens = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
        target_tokens = target_tokens[:target_tokens_len]
        correct_answer = target_tokens[example['choices']['label'].index(example['answerKey'])]
        for i in range(target_tokens_len):
            choice = choice + " (" + target_tokens[i] + ") " + example['choices']['text'][i] + " "
        question = question + choice
        target_token_ids = [tokenizer.encode(t, add_special_tokens=False)[0] for t in target_tokens]
    elif dataset_name == 'aqua':
        question = example['question']
        choice = ""
        target_tokens_len = len(example['options'])
        target_tokens = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
        target_tokens = target_tokens[:target_tokens_len]
        correct_answer = example['correct']
        for i in range(target_tokens_len):
            choice = choice + " (" + target_tokens[i] + ") " + example['options'][i].split(')')[1] + " "
        question = question + choice
        target_token_ids = [tokenizer.encode(t, add_special_tokens=False)[0] for t in target_tokens]
    elif dataset_name == 'StrategyQA':
        question = example['question']
        correct_answer = example['answer']
        target_tokens = ['True', 'False']
        target_token_ids = [tokenizer.encode(t, add_special_tokens=False)[0] for t in target_tokens]
    return question, correct_answer, target_tokens, target_token_ids

class StopStringCriteria(StoppingCriteria):
    def __init__(self, stop_strings, tokenizer):
        self.stop_strings = stop_strings
        self.tokenizer = tokenizer
        self.encoded_stop_strings = []
        for stop_str in stop_strings:
            encoded = tokenizer.encode(stop_str, add_special_tokens=False)
            self.encoded_stop_strings.append(encoded)

    def __call__(self, input_ids, scores, **kwargs):
        generated_tokens = input_ids[0]
        for encoded_stop in self.encoded_stop_strings:
            if len(generated_tokens) >= len(encoded_stop):
                if generated_tokens[-len(encoded_stop):].tolist() == encoded_stop:
                    return True

        return False

def generate_text(our_model, prompt, sampling_params: SamplingParams, tokenizer, device='cuda', max_new_tokens=None, num_return_sequences=1, important_word=False, seed=None):
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    # 设置停止条件
    stop_strings = ["</s>", "Human:", "Question:", "Q:"]
    if important_word:
        stop_strings = []
    stopping_criteria = StoppingCriteriaList([StopStringCriteria(stop_strings, tokenizer)])

    gen_cfg = GenerationConfig(
        temperature=sampling_params.temperature,
        top_p=sampling_params.top_p,
        do_sample=True,
        max_new_tokens=sampling_params.max_tokens if max_new_tokens is None else max_new_tokens,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id
    )

    inputs = tokenizer(prompt, return_tensors="pt", truncation=False).to(device)
    with torch.no_grad():
        outputs = our_model.generate(
            **inputs,
            generation_config=gen_cfg,
            num_return_sequences=num_return_sequences,
            return_dict_in_generate=True,
            output_scores=False,
            stopping_criteria = stopping_criteria
        )

    input_length = inputs.input_ids.shape[1]
    texts = []
    for seq in outputs.sequences:
        temp = tokenizer.decode(seq, skip_special_tokens=True)
        generated_tokens = seq[input_length:]
        text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        for stop_str in stop_strings:
            if stop_str in text:
                text = text.split(stop_str)[0]
        texts.append(text.strip())
    return texts

def extract_option_answer(text):
    pattern1 = re.search(r'The answer is\s*\(([A-E])\)', text, re.IGNORECASE)
    if pattern1:
        return pattern1.group(1).upper()
    pattern2 = re.search(r'The answer is\s*([A-E])(?:\s*\.|\s*[^A-E]|$)', text, re.IGNORECASE)
    if pattern2:
        return pattern2.group(1).upper()
    pattern3 = re.search(r'(?:Answer)[:\s]*([A-E])(?:\s*\.|\s*[^A-E]|$)', text, re.IGNORECASE)
    if pattern3:
        return pattern3.group(1).upper()
    boxed_match = re.search(r'\\boxed\{([A-E])\}', text, re.IGNORECASE)
    if boxed_match:
        return boxed_match.group(1).upper()
    lines = text.strip().split('\n')
    for line in reversed(lines):
        standalone_match = re.search(r'\b([A-E])\b(?:\s*\.)?\s*$', line, re.IGNORECASE)
        if standalone_match:
            return standalone_match.group(1).upper()
    final_match = re.search(r'\b([A-E])\b', text, re.IGNORECASE)
    if final_match:
        return final_match.group(1).upper()
    return ""


def extract_yesno_answer(text):
    pattern1 = re.search(r'The answer is\s*(yes|no|True|False|true|false)\b', text, re.IGNORECASE)
    if pattern1:
        return str(pattern1.group(1).lower() in ['yes', 'true'])

    pattern0 = re.search(r'the answer is\s*(yes|no|True|False|true|false)\b', text, re.IGNORECASE)
    if pattern0:
        return str(pattern0.group(1).lower() in ['yes', 'true'])

    pattern2 = re.search(r'(?:Answer)[:：]?\s*(yes|no|是|否|True|False|true|false)\b', text, re.IGNORECASE)
    if pattern2:
        ans = pattern2.group(1).lower()
        return "True" if ans in ["yes", "True", "true"] else "False"

    boxed_match = re.search(r'\\boxed\{(yes|no|True|False|true|false)\}', text, re.IGNORECASE)
    if boxed_match:
        ans = boxed_match.group(1).lower()
        return "True" if ans in ["yes", "True", "true"] else "False"

    lines = text.strip().split('\n')
    for line in reversed(lines):
        standalone = re.search(r'^\s*(yes|no|True|False|true|false)\s*\.?\s*$', line.strip(), re.IGNORECASE)
        if standalone:
            ans = standalone.group(1).lower()
            return "True" if ans in ["yes", "True", "true"] else "False"

    final_match = re.search(r'\b(yes|no|True|False|true|false)\b', text, re.IGNORECASE)
    if final_match:
        ans = final_match.group(1).lower()
        return "True" if ans in ["yes", "True", "true"] else "False"

    return ""

def split_cot_into_sentences(cot_text):
    if not cot_text or not isinstance(cot_text, str):
        return []
    sentences = sent_tokenize(cot_text)
    return [s.strip() for s in sentences if s.strip()]

def split_all_cots_into_sentences(cots_list):
    all_sentences = []
    for i, cot in enumerate(cots_list):
        sentences = split_cot_into_sentences(cot)
        all_sentences.append(sentences)
    return all_sentences

def calculate_yesno_accuracy(predictions, references):
    correct = 0
    total = len(predictions)
    normalization_map = {
        'yes': 'yes',
        'y': 'yes',
        'true': 'true',
        't': 'true',
        'no': 'no',
        'n': 'no',
        'false': 'false',
        'f': 'false'
    }

    for pred, ref in zip(predictions, references):
        if pred is not None and ref is not None:
            pred_str = str(pred).strip().lower()
            ref_str = str(ref).strip().lower()

            pred_normalized = normalization_map.get(pred_str, pred_str)
            ref_normalized = normalization_map.get(ref_str, ref_str)

            if (pred_normalized == 'yes' and ref_normalized == 'true') or \
                    (pred_normalized == 'true' and ref_normalized == 'yes'):
                correct += 1
            elif (pred_normalized == 'no' and ref_normalized == 'false') or \
                    (pred_normalized == 'false' and ref_normalized == 'no'):
                correct += 1
            elif pred_normalized == ref_normalized:
                correct += 1

    accuracy = correct / total if total > 0 else 0.0
    return accuracy, correct, total

def calculate_option_accuracy(predictions, references):
    total = len(predictions)
    correct = np.sum(np.array(predictions) == np.array(references))
    accuracy = correct / total if total > 0 else 0.0
    return accuracy, int(correct), int(total)

def load_question_choice(dataset_name, example, tokenizer):
    if dataset_name == 'ai2_arc_challenge' or dataset_name == 'ai2_arc_easy':
        question = example['question']
        choice = ""
        target_tokens_len = len(example['choices']['label'])
        target_tokens = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
        target_tokens = target_tokens[:target_tokens_len]
        correct_answer = "(" + target_tokens[example['choices']['label'].index(example['answerKey'])] + ") " + example['choices']['text'][example['choices']['label'].index(example['answerKey'])]
        for i in range(target_tokens_len):
            choice = choice + " (" + target_tokens[i] + ") " + example['choices']['text'][i] + " "
        target_token_ids = [tokenizer.encode(t, add_special_tokens=False)[0] for t in target_tokens]
    elif dataset_name == 'aqua':
        question = example['question']
        choice = ""
        target_tokens_len = len(example['options'])
        target_tokens = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
        target_tokens = target_tokens[:target_tokens_len]
        correct_answer = example['correct']
        for i in range(target_tokens_len):
            choice = choice + " (" + target_tokens[i] + ") " + example['options'][i].split(')')[1] + " "
        target_token_ids = [tokenizer.encode(t, add_special_tokens=False)[0] for t in target_tokens]
    elif dataset_name == 'StrategyQA':
        question = example['question']
        correct_answer = example['answer']
        target_tokens = ['True', 'False']
        target_token_ids = [tokenizer.encode(t, add_special_tokens=False)[0] for t in target_tokens]
        choice = None
    return question, choice, correct_answer, target_tokens, target_token_ids