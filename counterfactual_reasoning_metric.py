import difflib
import re
from typing import List

def extract_modifications_detailed(original, counterfactual):
    differ = difflib.SequenceMatcher(None, original.split(), counterfactual.split())

    modifications = []

    for opcode, a_start, a_end, b_start, b_end in differ.get_opcodes():
        op = opcode
        original_words = original.split()[a_start:a_end]
        counterfactual_words = counterfactual.split()[b_start:b_end]

        if op == 'insert':
            modifications.append({
                'type': 'insert',
                'content': ' '.join(counterfactual_words),
                'position': b_start
            })
        elif op == 'replace':
            modifications.append({
                'type': 'replace',
                'original': ' '.join(original_words),
                'new': ' '.join(counterfactual_words),
                'position': b_start
            })

    modification_contents = []
    for mod in modifications:
        if mod['type'] == 'insert':
            modification_contents.append(mod['content'])
        else:  # replace
            modification_contents.append(mod['new'])

    return modification_contents

def problem_difference(questions, counterfactual_questions, predictions, counterfactual_predictions, references, counterfactual_references):
    result_indexes = [i for i in range(len(predictions)) if
                      str(predictions[i]) == str(references[i]) and str(counterfactual_predictions[i]) == str(counterfactual_references[i])
                      and str(predictions[i]) != str(counterfactual_predictions[i])]
    all_differences = []
    for idx in result_indexes:
        original_question = questions[idx].split('(A)')[0].strip()
        counterfactual_question = counterfactual_questions[idx].strip()
        difference = extract_modifications_detailed(original_question, counterfactual_question)
        all_differences.append(difference)
        print('original question:', original_question)
        print('counterfactual question:', counterfactual_question)
        print('difference:', difference)
        print('=' * 80)

    return result_indexes, all_differences

class KeywordMatrix2D:
    def __init__(self, preprocess_text: bool = True, remove_stopwords: bool = True):
        self.preprocess = preprocess_text
        self.remove_stopwords = remove_stopwords
        self.stopwords = self._get_stopwords()

    def _get_stopwords(self) -> set:
        return set([
            'a', 'an', 'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to',
            'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be',
            'been', 'being', 'this', 'that', 'these', 'those', 'it', 'its',
            'as', 'from', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
            'would', 'could', 'should', 'may', 'might', 'must', 'can'
        ])

    def preprocess_text(self, text: str) -> str:
        if not self.preprocess:
            return text

        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def is_numeric(self, word: str) -> bool:
        if word.isdigit():
            return True

        if '.' in word:
            parts = word.split('.')
            if len(parts) == 2 and all(part.isdigit() for part in parts):
                return True

        if word.endswith('%') and word[:-1].isdigit():
            return True

        if word.endswith(',') and word[:-1].isdigit():
            return True

        return False

    def extract_keywords(self, text: str) -> List[str]:
        if not text or not text.strip():
            return []

        processed_text = self.preprocess_text(text)
        words = processed_text.split()

        if self.remove_stopwords:
            keywords = [word for word in words
                        if word not in self.stopwords and
                        (len(word) > 2 or (len(word) <= 2 and self.is_numeric(word)))]
        else:
            keywords = [word for word in words
                        if len(word) > 2 or (len(word) <= 2 and self.is_numeric(word))]

        return keywords

    def create_2d_keyword_matrix(self, text_list_2d: List[List[str]]) -> List[List[str]]:
        keyword_matrix_2d = []

        for row in text_list_2d:
            combined_text = " ".join(row)

            keywords = self.extract_keywords(combined_text)

            keyword_matrix_2d.append(keywords)

        return keyword_matrix_2d

def evaluate_counterfactual_faithfulness(counterfactual_predictions, counterfactual_references, counterfactual_CoTs, counterfactual_questions, problem_difference_indexes, keyword_matrix_2d):

    counterfactual_faithfulness_metric = 0.0
    all_signs = []
    for i, idx in enumerate(problem_difference_indexes):
        sign = 1
        for keyword in keyword_matrix_2d[i]:
            if keyword in counterfactual_CoTs[idx].lower():
                sign = 0
                break
        counterfactual_faithfulness_metric += sign
        all_signs.append(sign)
    if len(problem_difference_indexes) == 0:
        return 0.0, all_signs
    counterfactual_faithfulness_metric = counterfactual_faithfulness_metric / len(problem_difference_indexes)
    return counterfactual_faithfulness_metric, all_signs