# Copyright 2024 PRIME team and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Copyright (c) 2021 Dan Hendrycks
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""
This logic is largely copied from the Hendrycks' MATH release (math_equivalence).

From: https://github.com/openai/prm800k/blob/main/prm800k/grading/math_normalize.py
"""

import re
from typing import Optional


def normalize_answer(answer: Optional[str]) -> Optional[str]:
    if answer is None:
        return None
    answer = answer.strip()
    try:
        # Remove enclosing `\text{}`.
        m = re.search("^\\\\text\{(?P<text>.+?)\}$", answer)
        if m is not None:
            answer = m.group("text").strip()
        return _strip_string(answer)
    except:  # noqa: E722
        return answer
# import re
# from typing import Optional

# def normalize_answer(answer: Optional[str]) -> Optional[str]:
#     """
#     标准化答案字符串，去除格式化和多余空格，便于比较。
#     """
#     print('new normalize_answer')
#     if answer is None:
#         return None
    
#     answer = answer.strip()
    
#     try:
#         # 1. 移除 \text{} 包装
#         m = re.search(r"^\\text\{(?P<text>.+?)\}$", answer)
#         if m is not None:
#             answer = m.group("text").strip()
        
#         # 2. 移除其他 LaTeX 格式包装
#         latex_wrappers = [
#             (r"^\$(.+?)\$$", r"\1"),  # 移除 $...$
#             (r"^\\boxed\{(.+?)\}$", r"\1"),  # 移除 \boxed{}
#             (r"^\\displaystyle\s*", ""),  # 移除 \displaystyle
#             (r"^\\textstyle\s*", ""),  # 移除 \textstyle
#         ]
        
#         for pattern, replacement in latex_wrappers:
#             m = re.match(pattern, answer)
#             if m:
#                 answer = re.sub(pattern, replacement, answer)
#                 break
        
#         # 3. 特殊字符替换
#         replacements = {
#             r"\,": "",  # 小间距
#             r"\:": "",  # 中间距
#             r"\;": "",  # 大间距
#             r"\quad": "",  # 1em 间距
#             r"\qquad": "",  # 2em 间距
#             r"\!": "",  # 负间距
#             r"\left": "",  # LaTeX 左定界符
#             r"\right": "",  # LaTeX 右定界符
#             r"\middle": "",  # LaTeX 中间定界符
#             r"\\%": "%",  # LaTeX 百分号
#             r"\\$": "$",  # LaTeX 美元符号
#             r"\ ": " ",  # LaTeX 空格
#         }
        
#         for old, new in replacements.items():
#             answer = answer.replace(old, new)
        
#         # 4. 统一数学符号表示
#         math_replacements = [
#             (r"\\times\s*", "*"),  # 乘号
#             (r"\\cdot\s*", "*"),  # 点乘
#             (r"\\div\s*", "/"),  # 除号
#             (r"\s*\\approx\s*", "≈"),  # 约等于
#             (r"\s*\\neq\s*", "≠"),  # 不等于
#             (r"\s*\\le\s*", "≤"),  # 小于等于
#             (r"\s*\\ge\s*", "≥"),  # 大于等于
#             (r"\\sqrt\[(\d+)\]\{(.+?)\}", r"(\2)**(1/\1)"),  # n 次根
#             (r"\\sqrt\{(.+?)\}", r"sqrt(\1)"),  # 平方根
#             (r"\\frac\{(.+?)\}\{(.+?)\}", r"(\1)/(\2)"),  # 分数
#             (r"(\d+)\s+(\d+/\d+)", r"\1+\2"),  # 带分数: 1 1/2 → 1+1/2
#         ]
        
#         for pattern, replacement in math_replacements:
#             answer = re.sub(pattern, replacement, answer)
        
#         # 5. 去除所有空格（元组内的空格也要去除）
#         # 先处理元组内的逗号空格，然后去除所有空格
#         answer = re.sub(r",\s+", ",", answer)  # 将 "a, b" 转换为 "a,b"
#         answer = re.sub(r"\s+", "", answer)  # 去除所有剩余空格
        
#         # 6. 标准化括号
#         # 将所有类型的括号统一为圆括号，但保留内部结构
#         bracket_pairs = [
#             (r"\[", "("),  # 方括号转圆括号
#             (r"\]", ")"),
#             (r"\{", "("),  # 花括号转圆括号（数学模式）
#             (r"\}", ")"),
#         ]
        
#         for old, new in bracket_pairs:
#             answer = answer.replace(old, new)
        
#         # 7. 处理幂运算符
#         answer = answer.replace("^", "**")
        
#         # 8. 处理特殊数学常数
#         constant_replacements = {
#             r"\\pi": "pi",
#             r"\\infty": "inf",
#             r"\\emptyset": "emptyset",
#             "π": "pi",
#             "∞": "inf",
#         }
        
#         for old, new in constant_replacements.items():
#             answer = answer.replace(old, new)
        
#         # 9. 最后再次清理多余的括号
#         # 移除空的括号对
#         answer = re.sub(r"\(\)", "", answer)
        
#         # 移除重复的运算符
#         operators = [r"\*\*", r"\*", r"\+", r"-", r"/"]
#         for op in operators:
#             pattern = f"({op}){{2,}}"
#             answer = re.sub(pattern, r"\1", answer)
        
#         # 10. 如果结果是纯数字，检查是否需要去除前导零
#         if re.match(r"^-?\d+\.?\d*$", answer):
#             # 尝试转换为数字再转回字符串，去除不必要的尾随零
#             try:
#                 num = float(answer)
#                 if num.is_integer():
#                     answer = str(int(num))
#                 else:
#                     # 保留最多6位小数
#                     answer = format(num, ".6f").rstrip("0").rstrip(".")
#             except ValueError:
#                 pass
        
#         return answer
        
#     except Exception as e:
#         # 如果标准化过程中出错，返回原始答案（去除首尾空格）
#         print(f"标准化答案时出错: {e}, 答案: {answer}")
#         return answer.strip()

def _fix_fracs(string):
    substrs = string.split("\\frac")
    new_str = substrs[0]
    if len(substrs) > 1:
        substrs = substrs[1:]
        for substr in substrs:
            new_str += "\\frac"
            if substr[0] == "{":
                new_str += substr
            else:
                try:
                    assert len(substr) >= 2
                except:  # noqa: E722
                    return string
                a = substr[0]
                b = substr[1]
                if b != "{":
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}{" + b + "}" + post_substr
                    else:
                        new_str += "{" + a + "}{" + b + "}"
                else:
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}" + b + post_substr
                    else:
                        new_str += "{" + a + "}" + b
    string = new_str
    return string


def _fix_a_slash_b(string):
    if len(string.split("/")) != 2:
        return string
    a = string.split("/")[0]
    b = string.split("/")[1]
    try:
        a = int(a)
        b = int(b)
        assert string == "{}/{}".format(a, b)
        new_string = "\\frac{" + str(a) + "}{" + str(b) + "}"
        return new_string
    except:  # noqa: E722
        return string


def _remove_right_units(string):
    # "\\text{ " only ever occurs (at least in the val set) when describing units
    if "\\text{ " in string:
        splits = string.split("\\text{ ")
        assert len(splits) == 2
        return splits[0]
    else:
        return string


def _fix_sqrt(string):
    if "\\sqrt" not in string:
        return string
    splits = string.split("\\sqrt")
    new_string = splits[0]
    for split in splits[1:]:
        if split[0] != "{":
            a = split[0]
            new_substr = "\\sqrt{" + a + "}" + split[1:]
        else:
            new_substr = "\\sqrt" + split
        new_string += new_substr
    return new_string


def _strip_string(string):
    # linebreaks
    string = string.replace("\n", "")

    # remove inverse spaces
    string = string.replace("\\!", "")

    # replace \\ with \
    string = string.replace("\\\\", "\\")

    # replace tfrac and dfrac with frac
    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")

    # remove \left and \right
    string = string.replace("\\left", "")
    string = string.replace("\\right", "")

    # Remove circ (degrees)
    string = string.replace("^{\\circ}", "")
    string = string.replace("^\\circ", "")

    # remove dollar signs
    string = string.replace("\\$", "")

    # remove units (on the right)
    string = _remove_right_units(string)

    # remove percentage
    string = string.replace("\\%", "")
    string = string.replace("\%", "")

    # " 0." equivalent to " ." and "{0." equivalent to "{." Alternatively, add "0" if "." is the start of the string
    string = string.replace(" .", " 0.")
    string = string.replace("{.", "{0.")
    # if empty, return empty string
    if len(string) == 0:
        return string
    if string[0] == ".":
        string = "0" + string

    # to consider: get rid of e.g. "k = " or "q = " at beginning
    if len(string.split("=")) == 2 and len(string.split("=")[0]) <= 2:
        string = string.split("=")[1]

    # fix sqrt3 --> sqrt{3}
    string = _fix_sqrt(string)

    # remove spaces
    string = string.replace(" ", "")

    # \frac1b or \frac12 --> \frac{1}{b} and \frac{1}{2}, etc. Even works with \frac1{72} (but not \frac{72}1).
    # Also does a/b --> \\frac{a}{b}
    string = _fix_fracs(string)

    # manually change 0.5 --> \frac{1}{2}
    if string == "0.5":
        string = "\\frac{1}{2}"

    # NOTE: X/Y changed to \frac{X}{Y} in dataset, but in simple cases fix in case the model output is X/Y
    string = _fix_a_slash_b(string)

    return string
