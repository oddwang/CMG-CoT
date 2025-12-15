def get_ai2_arc_reasoning_tasks_few_shot_CoT_prompt():
    few_shot_examples = [
        {
            "question": "George wants to warm his hands quickly by rubbing them. Which skin surface will produce the most heat? (A) dry palms. (B) wet palms. (C) palms covered with oil. (D) palms covered with lotion.",
            "answer": "Let's think step by step. Dry surfaces will more likely cause more friction via rubbing than other smoother surfaces, hence dry palms will produce the most heat. The answer is (A)."
        },
        {
            "question": "Which factor will most likely cause a person to develop a fever? (A) a leg muscle relaxing after exercise. (B) a bacterial population in the bloodstream. (C) several viral particles on the skin. (D) carbohydrates being digested in the stomach. ",
            "answer": "Let's think step by step. Option (B), bacterial population is the most likely cause for a person developing fever. The answer is (b)."
        },
        {
            "question": "Which change in the state of water particles causes the particles to become arranged in a fixed position? (A) boiling. (B) melting. (C) freezing. (D) evaporating.",
            "answer": "When water is freezed, the particles are arranged in a fixed position; the particles are still moving for all other options. The answer is (C)."
        },
        {
            "question": "When a switch is used in an electrical circuit, the switch can (A) cause the charge to build. (B) increase and decrease the voltage. (C) cause the current to change direction. (D) stop and start the flow of current.",
            "answer": "Let's think step by step. The function of a switch is to start and stop the flow of a current. The answer is (D)."
        },

    ]
    return few_shot_examples

def get_aqua_few_shot_CoT_prompt():
    few_shot_examples = [
        {
            "question": "If each edge of cube increased by 10%, the percentage increase in (A) 15  (B) 19  (C) 21  (D) 22  (E) 25 ",
            "answer": "Let's think step by step. 100 × (110)/100 × (110)/100 = 121. 121 = 21%. So, the answer is (C)."
        },
        {
            "question": "A cistern has a leak which empty it in 8 hrs, A tap is turned on which admits 6 liters a minute into the cistern, and it is now emptied in 12 hours, How many liters does the cistern hold ? (A) 8640  (B) 8740  (C) 8840  (D) 8540  (E) 8940 ",
            "answer": "Let's think step by step. Inflow of water in 12hrs with rate 6 lit/min= 6*60*12= 4320. But, the same volume is emptied in 12 hrs, so the total capacity = total inflow + total outflow = 4320+4320 = 8640. So, the answer is (A)."
        },
        {
            "question": "A train 250 m long passes a man, running at 5 km/hr in the same direction in which the train is going, in 10 seconds. The speed of the train is: (A) 95  (B) 50  (C) 12  (D) 13  (E) 67 ",
            "answer": "Let's think step by step. Speed of the train relative to man = (250/10) m/sec = (25) m/sec. [(25) * (18/5)] km/hr = 90 km/hr. Let the speed of the train be x km/hr. Then, relative speed = (x - 5) km/hr. x - 5 = 90. x = 95 km/hr. So, the answer is (A)."
        },
        {
            "question": "If a and b are odd integers, which of the following is an even integer? (A) 3(2a+b)  (B) 3(2a+b)+2a  (C) 3(2a+b)+a+b  (D) 3(2a+b)+7a-b  (E) 3(2a+b)+ab ",
            "answer": "Let's think step by step. 3(2a+b)+ab=odd(odd)+odd=odd+odd=even. So, the answer is (E)."
        },

    ]
    return few_shot_examples

def get_StrategyQA_few_shot_CoT_prompt():
    few_shot_examples = [
        {
            "question": "Do hamsters provide food for any animals?",
            "answer": "Let's think step by step. Hamsters are prey animals. Prey are food for predators. Thus, hamsters provide food for some animals. So, the answer is True."
        },
        {
            "question": "Could Brooke Shields succeed at University of Pennsylvania?",
            "answer": "Let's think step by step. Brooke Shields went to Princeton University. Princeton University is about as academically rigorous as the University of Pennsylvania. Thus, Brooke Shields could also succeed at the University of Pennsylvania. So, the answer is True."
        },
        {
            "question": "Yes or no: Hydrogen’s atomic number squared exceeds number of Spice Girls?",
            "answer": "Let's think step by step. Hydrogen has an atomic number of 1. 1 squared is 1. There are 5 Spice Girls. Thus, Hydrogen’s atomic number squared is less than 5. So the answer is False."
        },
        {
            "question": "Yes or no: Is it common to see frost during some college commencements?",
            "answer": "Let's think step by step. College commencement ceremonies can happen in December, May, and June. December is in the winter, so there can be frost. Thus, there could be frost at some commencements. So, the answer is True."
        },
        {
            "question": "Yes or no: Could a llama birth twice during War in Vietnam (1945-46)?",
            "answer": "Let's think step by step. The War in Vietnam was 6 months. The gestation period for a llama is 11 months, which is more than 6 months. Thus, a llama could not give birth twice during the War in Vietnam. So, the answer is False."
        },
        {
            "question": "Yes or no: Would a pear sink in water?",
            "answer": "Let's think step by step. The density of a pear is about 0.6g/cm^3, which is less than water. Objects less dense than water float. Thus, a pear would float. So, the answer is False."
        },
    ]
    return few_shot_examples


def create_ai2_arc_prompt(question):
    prompt_template = """<s>[INST]<<SYS>>
           You are a helpful, respectful and honest assistant. You are to answer the last question by first providing a reasoning followed by the answer option, similar to previous examples. The final answer in the format: The answer is (X).
           <</SYS>>

           """

    few_shot_examples = get_ai2_arc_reasoning_tasks_few_shot_CoT_prompt()

    for example in few_shot_examples:
        prompt_template += 'Q: ' + example['question'] + '\nA: ' + example['answer'] + '\n\n'

    prompt_template += 'Q: ' + question + "\nA:" + '[/INST]'

    return prompt_template

def create_aqua_prompt(question):
    prompt_template = """<s>[INST]<<SYS>>
           You are a helpful, respectful and honest assistant. You are to answer the last question by first providing a reasoning followed by the answer option, similar to previous examples. The final answer in the format: So, the answer is (X).
           <</SYS>>

           """

    few_shot_examples = get_aqua_few_shot_CoT_prompt()

    for example in few_shot_examples:
        prompt_template += 'Q: ' + example['question'] + '\nA: ' + example['answer'] + '\n\n'

    prompt_template += 'Q: ' + question + "\nA: Let's think step by step." + '[/INST]'

    return prompt_template

def create_StrategyQA_prompt(question):
    prompt_template = """<s>[INST]<<SYS>>
           You are a helpful, respectful and honest assistant. You are to answer the last question by first providing a reasoning followed by the answer the question, similar to previous examples. The final answer in the format: So, the answer is True/False.
           <</SYS>>

           """

    few_shot_examples = get_StrategyQA_few_shot_CoT_prompt()

    for example in few_shot_examples:
        prompt_template += 'Q: ' + example['question'] + '\nA: ' + example['answer'] + '\n\n'

    prompt_template += 'Q: ' + question + "\nA: Let's think step by step." + '[/INST]'

    return prompt_template