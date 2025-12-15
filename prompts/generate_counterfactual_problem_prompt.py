def get_ai2_arc_counterfactual_problem_few_shot_prompt():
    few_shot_examples = [

        {
            "question": "Which factor will most likely cause a person to develop a fever?",
            "answer_choices": "(A) a leg muscle relaxing after exercise. (B) a bacterial population in the bloodstream. (C) several viral particles on the skin. (D) carbohydrates being digested in the stomach.",
            "answer": "(B) a bacterial population in the bloodstream.",
            "counterfactual_problem": "Which factor will most likely cause a person to develop a fever through skin contact?",
            "counterfactual_answer": "(C) several viral particles on the skin.",
        },
        {
            "question": "Which of the following statements best explains why magnets usually stick to a refrigerator door?",
            "answer_choices": "(A) The refrigerator door is smooth.  (B) The refrigerator door contains iron.  (C) The refrigerator door is a good conductor.  (D) The refrigerator door has electric wires in it. ",
            "answer": "(B) The refrigerator door contains iron.",
            "counterfactual_problem": "Which of the following statements best explains why the refrigerator door feels cold to the touch?",
            "counterfactual_answer": "(C) The refrigerator door is a good conductor.",
        },
        {
            "question": "A fold observed in layers of sedimentary rock most likely resulted from the",
            "answer_choices": "(A) cooling of flowing magma.  (B) converging of crustal plates.  (C) deposition of river sediments.  (D) solution of carbonate minerals. ",
            "answer": "(B) converging of crustal plates.",
            "counterfactual_problem": "A sill observed in layers of sedimentary rock most likely resulted from the",
            "counterfactual_answer": "(A) cooling of flowing magma.",
        },
        {
            "question": "A boat is acted on by a river current flowing north and by wind blowing on its sails. The boat travels northeast. In which direction is the wind most likely applying force to the sails of the boat?",
            "answer_choices": "(A) west (B) east (C) north (D) south",
            "answer": "(B) east",
            "counterfactual_problem": "A boat is acted on by a river current flowing north and by wind blowing on its sails. The boat travels north. In which direction is the wind most likely applying force to the sails of the boat?",
            "counterfactual_answer": "(C) north",
        },

    ]
    return few_shot_examples

def get_aqua_counterfactual_problem_few_shot_prompt():
    few_shot_examples = [

        {
            "question": "If each edge of cube increased by 10%, the percentage increase in",
            "answer_choices": "(A) 15  (B) 19  (C) 21  (D) 22  (E) 44 ",
            "answer": "(C) 21 ",
            "counterfactual_problem": "If each edge of cube increased by 20%, the percentage increase in",
            "counterfactual_answer": "(E) 44 ",
        },
        {
            "question": "A cistern has a leak which empty it in 8 hrs, A tap is turned on which admits 6 liters a minute into the cistern, and it is now emptied in 12 hours, How many liters does the cistern hold?",
            "answer_choices": "(A) 8640  (B) 8740  (C) 8840  (D) 8540  (E) 8940 ",
            "answer": "(A) 8640",
            "counterfactual_problem": "A cistern has a leak which empty it in 7.5 hrs. A tap is turned on which admits 6 liters a minute into the cistern, and it is now emptied in 12 hours. How many liters does the cistern hold?",
            "counterfactual_answer": "(B) 8740 ",
        },
        {
            "question": "A train 250 m long passes a man, running at 5 km/hr in the same direction in which the train is going, in 10 seconds. The speed of the train is:",
            "answer_choices": "(A) 95  (B) 50  (C) 12  (D) 13  (E) 67 ",
            "answer": "(A) 95",
            "counterfactual_problem": "A train 250 m long passes a man, running at 5 km/hr in the same direction in which the train is going, in 20 seconds. The speed of the train is:",
            "counterfactual_answer": "(B) 50 ",
        },
        {
            "question": "If a and b are odd integers, which of the following is an even integer? ",
            "answer_choices": "(A) 3(2a+b)  (B) 3(2a+b)+2a  (C) 3(2a+b)+a+b  (D) 3(2a+b)+7a-b  (E) 3(2a+b)+ab ",
            "answer": "(E) 3(2a+b)+ab ",
            "counterfactual_problem": "If a is an even integer and b is an odd integer, which of the following is an even integer?",
            "counterfactual_answer": "(C) 3(2a+b)+a+b ",
        },

    ]
    return few_shot_examples

def get_StrategyQA_counterfactual_problem_few_shot_prompt():
    few_shot_examples = [

        {
            "question": "Do hamsters provide food for any animals?",
            "answer": "True",
            "counterfactual_problem": "Do hamsters rely on any animals for food?",
            "counterfactual_answer": "False",
        },
        {
            "question": "Could Brooke Shields succeed at University of Pennsylvania?",
            "answer": "True",
            "counterfactual_problem": "Could Brooke Shields fail at University of Pennsylvania?",
            "counterfactual_answer": "False",
        },
        {
            "question": "Yes or no: Hydrogen’s atomic number squared exceeds number of Spice Girls?",
            "answer": "False",
            "counterfactual_problem": "Yes or no: Hydrogen’s atomic number squared is less than the number of Spice Girls?",
            "counterfactual_answer": "True",
        },
        {
            "question": "Yes or no: Is it common to see frost during some college commencements?",
            "answer": "True",
            "counterfactual_problem": "Yes or no: Is it uncommon to see frost during some college commencements?",
            "counterfactual_answer": "False",
        },

    ]
    return few_shot_examples

def create_ai2_arc_generate_counterfactual_problem_prompt(question, answer_choices, answer):
    prompt_template = """You are a helpful, respectful and honest assistant. Given a question and corresponding answer, please choose the second most likely answer from the answer choices and generate a new question such that the new question will correspond to the second likely answer. You are to make minimal changes to the question. \n\n"""

    few_shot_examples = get_ai2_arc_counterfactual_problem_few_shot_prompt()

    for example in few_shot_examples:
        prompt_template += ('Question: ' + example['question'] + '\nAnswer Choices: ' + example['answer_choices'] +
                            '\nAnswer: ' + example['answer'] + '\nCounterfactual Problem: ' + example['counterfactual_problem']
                            + '\nCounterfactual Answer: ' + example['counterfactual_answer'] + '\n\n')

    prompt_user = 'Question: ' + question + '\nAnswer Choices: ' + answer_choices + '\nAnswer: ' + answer + '\nCounterfactual Problem:'

    return prompt_template, prompt_user

def create_aqua_generate_counterfactual_problem_prompt(question, answer_choices, answer):
    prompt_template = """You are a helpful, respectful and honest assistant. Given a question and corresponding answer, please choose the second most likely answer from the answer choices and generate a new question such that the new question will correspond to the second likely answer. You are to make minimal changes to the question. \n\n"""

    few_shot_examples = get_aqua_counterfactual_problem_few_shot_prompt()

    for example in few_shot_examples:
        prompt_template += ('Question: ' + example['question'] + '\nAnswer Choices: ' + example['answer_choices'] +
                            '\nAnswer: ' + example['answer'] + '\nCounterfactual Problem: ' + example['counterfactual_problem']
                            + '\nCounterfactual Answer: ' + example['counterfactual_answer'] + '\n\n')

    prompt_user = 'Question: ' + question + '\nAnswer Choices: ' + answer_choices + '\nAnswer: ' + answer + '\nCounterfactual Problem:'

    return prompt_template, prompt_user

def create_StrategyQA_generate_counterfactual_problem_prompt(question, answer_choices, answer):
    prompt_template = """You are a helpful, respectful and honest assistant. Given a question and corresponding answer, please generate a new question that yields the opposite answer. You are to make minimal changes to the question. \n\n"""

    few_shot_examples = get_StrategyQA_counterfactual_problem_few_shot_prompt()

    for example in few_shot_examples:
        prompt_template += ('Question: ' + example['question'] +
                            '\nAnswer: ' + example['answer'] + '\nCounterfactual Problem: ' + example['counterfactual_problem']
                            + '\nCounterfactual Answer: ' + example['counterfactual_answer'] + '\n\n')

    prompt_user = 'Question: ' + question + '\nAnswer: ' + str(answer) + '\nCounterfactual Problem:'

    return prompt_template, prompt_user
