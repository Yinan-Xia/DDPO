import glob
import sys
import json
import re
import random
from . import bbeh
assert len(sys.argv) > 1, 'You need to pass the directory'
path = sys.argv[1]
random.seed(12345)



def extract_answer(text, level):
    if level == 'l1':
        pattern = r"answer is \(?([A-J])\)?"
        match = re.search(pattern, text)
        if match:
            return match.group(1)
        else:
            return None
    elif level == 'l2':
        pattern = r"answer is \(?([A-J])\)?"
        match = re.search(pattern, text)
        if match:
            return match.group(1)
        else:
            return extract_again(text)


def extract_again(text):
    match = re.search(r'.*[aA]nswer:\s*([A-J])', text)
    if match:
        return match.group(1)
    else:
        return extract_final(text)
    

def extract_final(text):
    pattern = r"\b[A-J]\b(?!.*\b[A-J]\b)"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(0)
    else:
        return None

# def compute_score(pred,gt):
#     # pred = extract_answer(e['model_outputs'], 'l1')
#     # if pred is None:
#     #     pred = extract_answer(e['model_outputs'], 'l2')
#     pred = bbeh.preprocess_sample(pred)
#     if pred is None or pred is "":
#         pred = random.choice(["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"])
#     # Remove the None cases
#     if pred == gt:
#         return True
#     else:
#         return False

def compute_score(pred,gt):
    pred = bbeh.preprocess_sample(pred)
    if pred == gt:
        return True
    else:
        return False