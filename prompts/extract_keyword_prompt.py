def get_arc_keyword_few_shot_prompt():
    few_shot_examples = [

        {
            "question": "Which factor will most likely cause a person to develop a fever?",
            "answer_options": "(A) a leg muscle relaxing after exercise. (B) a bacterial population in the bloodstream. (C) several viral particles on the skin. (D) carbohydrates being digested in the stomach.",
            "rank_result": "1. fever; 2. cause; 3. develop; 4. factor; 5. most;",
        },
        {
            "question": "Which of the following statements best explains why magnets usually stick to a refrigerator door?",
            "answer_options": "(A) The refrigerator door is smooth.  (B) The refrigerator door contains iron.  (C) The refrigerator door is a good conductor.  (D) The refrigerator door has electric wires in it.",
            "rank_result": "1. magnets; 2. refrigerator; 3. door; 4. stick; 5. iron;",
        },
        {
            "question": "A fold observed in layers of sedimentary rock most likely resulted from the",
            "answer_options": "(A) cooling of flowing magma.  (B) converging of crustal plates.  (C) deposition of river sediments.  (D) solution of carbonate minerals.",
            "rank_result": "1. fold; 2. sedimentary; 3. rock; 4. layers; 5. resulted;",
        },
        {
            "question": "A boat is acted on by a river current flowing north and by wind blowing on its sails. The boat travels northeast. In which direction is the wind most likely applying force to the sails of the boat?",
            "answer_options": "(A) west (B) east (C) north (D) south",
            "rank_result": "1. boat; 2. northeast; 3. wind; 4. sails; 5. current;",
        },

    ]
    return few_shot_examples

def get_aqua_keyword_few_shot_prompt():
    few_shot_examples = [

        {
            "question": "Suresh started a business, investing Rs.18000. After 3 months and 4 months respectively, Rohan and Sudhir joined him with capitals of 12000 and 9000. At the end of the year the total profit was Rs.4048. What is the difference between Rohan’s and Sudhir’s share in the profit?",
            "answer_options": "(A) s.345  (B) s.350  (C) s.352  (D) s.362  (E) s.368 ",
            "rank_result": "1. 4048; 2. 12000; 3. 9000; 4. 18000; 5. profit;",
        },
        {
            "question": "A cistern has a leak which empty it in 8 hrs, A tap is turned on which admits 6 liters a minute into the cistern, and it is now emptied in 12 hours, How many liters does the cistern hold ?",
            "answer_options": "(A) 8640  (B) 8740  (C) 8840  (D) 8540  (E) 8940 ",
            "rank_result": "1. 6; 2. 12; 3. 8; 4. liters; 5. minute;",
        },
        {
            "question": "A family consists of two grandparents, two parents, and four grandchildren. The average age of the grandparents is 66 years, the average age of the parents is 40 years, and the average age of the grandchildren is 7 years. What is the average age (in years) of the family?",
            "answer_options": "(A) 26  (B) 28  (C) 30  (D) 32  (E) 34 ",
            "rank_result": "1. 66; 2. 40; 3. 7; 4. average; 5. age;",
        },
        {
            "question": "A doctor prescribed 18 cubic centimeters of a certain drug to a patient whose body weight was 135 pounds. If the typical dosage is 2 cubic centimeters per 15 pounds of the body weight, by what percent was the prescribed dosage greater than the typical dosage?",
            "answer_options": "(A) 8%  (B) 9%  (C) 11%  (D) 12.5%  (E) 14.8% ",
            "rank_result": "1. 135; 2. 18; 3. 15; 4. 2; 5. greater;",
        },
        {
            "question": "When x is divided by 3, the quotient is y and the remainder is 1. When x is divided by 5, the quotient is z and the remainder is 6. Which of the following is the value of y in terms of z?",
            "answer_options": "(A) 4z/7 + 5  (B) (7z + 5  (C) (6z + 7  (D) (5z + 5  (E) (4z + 6 ",
            "rank_result": "1. 3; 2. 1; 3. 5; 4. 6; 5. y;",
        },

    ]
    return few_shot_examples

def get_StrategyQA_keyword_few_shot_prompt():
    few_shot_examples = [

        {
            "question": "Would a modern central processing unit circuit chip fit on a housekey?",
            "rank_result": "1. central; 2. processing; 3. unit; 4. circuit; 5. chip;",
        },
        {
            "question": "Would the average Hawaiian male experience more days on Earth compared to a wild cane toad?",
            "rank_result": "1. average; 2. Hawaiian; 3. male; 4. male; 5. male;",
        },
        {
            "question": "Could Aretha Franklin vote for a president when her second child was born?",
            "rank_result": "1. vote; 2. president; 3. Aretha; 4. Franklin; 5. born;",
        },
        {
            "question": "Did François Mitterrand serve under Napoleon Bonapart in the French army?",
            "rank_result": "1. Mitterrand; 2. Napoleon; 3. François; 4. Bonapart; 5. serve;",
        },
        {
            "question": "Would Elon Musk be more likely to know about astrology than physics?",
            "rank_result": "1. Musk; 2. Elon; 3. physics; 4. astrology; 5. know;",
        },

    ]
    return few_shot_examples

def extract_keyword_prompt(dataset_name, question, choice):
    if dataset_name == 'ai2_arc_easy' or dataset_name == 'ai2_arc_challenge':
        few_shot_examples = get_arc_keyword_few_shot_prompt()
    elif dataset_name == 'aqua':
        few_shot_examples = get_aqua_keyword_few_shot_prompt()
    elif dataset_name == 'StrategyQA':
        few_shot_examples = get_StrategyQA_keyword_few_shot_prompt()
    examples_text = ""
    for i, example in enumerate(few_shot_examples, 1):
        examples_text += f"Example {i}:\n"
        examples_text += f"Question: {example['question']}\n"
        if dataset_name != 'StrategyQA':
            examples_text += f"Answer Options: {example['answer_options']}\n"
        examples_text += f"Rank Result: {example['rank_result']}\n\n"
    if dataset_name != 'StrategyQA':
        prompt_template = """<s>[INST]
    You are an AI assistant that analyzes the importance of words in questions. Your task is to evaluate all words from the question and rank them by their importance for answering the question correctly.

    # INSTRUCTIONS:
    1. Analyze each word's importance for understanding and answering the question
    2. Consider semantic meaning, contextual relevance, and relevance to answer options
    3. Select and rank ONLY the TOP 5 most important words

    # CRITICAL REQUIREMENTS:
    - Output EXACTLY 5 words - no more, no less
    - All words MUST come from the question text
    - Each word appears ONLY ONCE
    - NO explanations, comments, or additional text
    - Begin directly with the ranked list

    # OUTPUT FORMAT (STRICT):
    1. <Most important word>;
    2. <Second most important word>;
    3. <Third most important word>;
    4. <Fourth most important word>;
    5. <Fifth most important word>;

    # FEW-SHOT EXAMPLES:
    {examples}

    # CURRENT TASK:
    Question: {question}
    Answer Options: {choices}
    Rank result:
    [/INST]"""
    else:
        prompt_template = """<s>[INST]
    You are an AI assistant that analyzes the importance of words in questions. Your task is to evaluate all words from the question and rank them by their importance for answering the question correctly.

    # INSTRUCTIONS:
    1. Analyze each word's importance for understanding and answering the question
    2. Consider semantic meaning, contextual relevance, and relevance to answer options
    3. Select and rank ONLY the TOP 5 most important words

    # CRITICAL REQUIREMENTS:
    - Output EXACTLY 5 words - no more, no less
    - All words MUST come from the question text
    - Each word appears ONLY ONCE
    - NO explanations, comments, or additional text
    - Begin directly with the ranked list

    # OUTPUT FORMAT (STRICT):
    1. <Most important word>;
    2. <Second most important word>;
    3. <Third most important word>;
    4. <Fourth most important word>;
    5. <Fifth most important word>;

    # FEW-SHOT EXAMPLES:
    {examples}

    # CURRENT TASK:
    Question: {question}
    Rank result: 
[/INST]"""

    prompt_template = prompt_template.format(
        examples=examples_text.strip(),
        question=question,
        choices=choice,
    )

    return prompt_template