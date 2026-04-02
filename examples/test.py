from verl.utils.reward_score.prime_math.before import grade_answer as grade_answer_before
from verl.utils.reward_score.prime_math import grade_answer as grade_answer_now

def _last_boxed_only_string(string):
    idx = string.rfind("\\boxed")
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    left_brace_idx = None
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
            if left_brace_idx is None:
                left_brace_idx = i
        elif string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break

        i += 1

    if left_brace_idx is None or right_brace_idx is None:
        return None

    return string[left_brace_idx + 1 : right_brace_idx].strip()




def match_answer(response):
    is_matched = False
    for ans_marker in ["answer:", "answer is", "answers are"]:
        ans_idx = response.lower().rfind(ans_marker)
        if ans_idx != -1:
            is_matched = True
            response = response[ans_idx + len(ans_marker) :].strip()
            if response.endswith("\n"):
                response = response[:-2]

    for ans_marker in ["is answer", "is the answer", "are answers", "are the answers"]:
        ans_idx = response.lower().rfind(ans_marker)
        if ans_idx != -1:
            is_matched = True
            response = response[:ans_idx].strip()
            if response.endswith("\n"):
                response = response[:-2]

    # Find boxed
    ans_boxed = _last_boxed_only_string(response)
    if ans_boxed:
        is_matched = True
        response = ans_boxed

    if ". " in response:
        dot_idx = response.lower().rfind(". ")
        if dot_idx != -1:
            response = response[:dot_idx].strip()

    for ans_marker in ["be ", "is ", "are ", "=", ": ", "get ", "be\n", "is\n", "are\n", ":\n", "get\n"]:
        ans_idx = response.lower().rfind(ans_marker)
        if ans_idx != -1:
            is_matched = True
            response = response[ans_idx + len(ans_marker) :].strip()
            if response.endswith("\n"):
                response = response[:-2]

    is_matched = is_matched if any([c.isdigit() for c in response]) else False  # answer must have a digit
    # Grade
    return is_matched, response





def compute_score(model_output: str, ground_truth: str) -> bool:
    model_output = str(model_output)
    ground_truth = str(ground_truth)

    is_matched, extracted_model_output = match_answer(model_output)
    format_correctness = "Step 2:" in model_output and "\\box" in model_output

    # print(is_matched, extracted_model_output, format_correctness)

    # grade simple algebra questions. if succeeded, return; otherwise, proceed to more complex grading
    
    print(1, grade_answer_before(extracted_model_output, ground_truth))
    print(2, grade_answer_now(extracted_model_output, ground_truth))
    # print(grade_answer_before(extracted_model_output, ground_truth)) false
    # print(grade_answer_now(extracted_model_output, ground_truth)) true
    if grade_answer_now(extracted_model_output, ground_truth):
        return True, True, extracted_model_output

    try:
        if "\\pi" in extracted_model_output or "\\pi" in ground_truth:
            equivs = []
            for pi in [math.pi, 3.14]:
                equivs.append(math_equal(extracted_model_output, ground_truth, timeout=True, pi=pi))
            is_correct = any(equivs)
        else:
            is_correct = math_equal(extracted_model_output, ground_truth, timeout=True)
    except Exception:
        is_correct = False

    return is_correct, format_correctness, extracted_model_output




model_output = """leading to some points being uncovered. Therefore, the minimal number of planes required is 3n.\n\nTherefore, the answer should be \\boxed{3n}.\n\n**Final Answer**\n\\boxed{3n}"""

ground_truth = """$3 n$"""

print(compute_score(model_output, ground_truth))