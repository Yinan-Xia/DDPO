# Copyright 2024 Bytedance Ltd. and/or its affiliates
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

from collections import defaultdict
from typing import Any

import torch
import math as math_o

from verl.utils.reward_score import default_compute_score
from verl import DataProto
from verl.workers.reward_manager import register
from verl.workers.reward_manager.abstract import AbstractRewardManager

import os
import json
import uuid
from datetime import datetime
import random
import math as math_o

@register("grpo_lead")
class LEADRewardManager(AbstractRewardManager):
    """The reward manager."""

    def __init__(self, tokenizer, num_examine, compute_score=None, reward_fn_key="data_source") -> None:
        """
        Initialize the NaiveRewardManager instance.

        Args:
            tokenizer: The tokenizer used to decode token IDs into text.
            num_examine: The number of batches of decoded responses to print to the console for debugging purpose.
            compute_score: A function to compute the reward score. If None, `default_compute_score` will be used.
            reward_fn_key: The key used to access the data source in the non-tensor batch data. Defaults to
                "data_source".
        """
        self.tokenizer = tokenizer  # Store the tokenizer for decoding token IDs
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.compute_score = compute_score or default_compute_score
        
        self.reward_fn_key = reward_fn_key  # Store the key for accessing the data source

    def get_reweight_rewards(self, rewards, completions_length, index):
        """
        rewards: list of rewards (1-d tensor)
        completions_length: list of completions length (1-d tensor)
        index: list indicating which question each sample belongs to
        """

        unique_questions = set(index)
        
        for q in unique_questions:
            group_indices = [i for i, q_idx in enumerate(index) if q_idx == q]

            # 初始化统计量
            sum_len = 0
            var_len = 0
            num_correct = 0
            min1, min2, min3 = float('inf'), float('inf'), float('inf')
            
            for i in group_indices:
                L = completions_length[i]
                if L < min1:
                    min3 = min2
                    min2 = min1
                    min1 = L
                elif L < min2:
                    min3 = min2
                    min2 = L
                elif L < min3:
                    min3 = L
                if rewards[i] > 0:
                    num_correct += 1
                    sum_len += L

            avg_min = (min1 + min2 + min3) / 3
            if avg_min <= 4000:
                continue
            
            avg_len = sum_len / num_correct if num_correct > 0 else 0.0
            std_len = 1.0
            if num_correct > 1:
                for i in group_indices:
                    if rewards[i] > 0:
                        var_len += (completions_length[i] - avg_len) ** 2
                val_len = var_len / (num_correct - 1)
                std_len = math_o.sqrt(val_len)
                
            alpha = 0.05
            
            min_three_avg = (min1 + min2 + min3) / 3
            
            for i in group_indices:
                if rewards[i] > 0:
                    L_i = completions_length[i]
                    epsilon = 1e-6
                    z_i = (L_i - avg_len) / (std_len + epsilon)
                    rewards[i] = rewards[i] * math_o.exp(-alpha * z_i)
                    
        return rewards

    def __call__(self, data: DataProto, return_dict: bool = False) -> torch.Tensor | dict[str, Any]:
        """We will expand this function gradually based on the available datasets"""

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if 'rm_scores' in data.batch.keys():
            return data.batch['rm_scores']


        already_print_data_sources = {}

        from concurrent.futures import ThreadPoolExecutor
        from typing import Dict, Any
        #import threading
        # Thread-safe dict for tracking printed data sources
        # print_lock = threading.Lock()
        
        def process_item(args):
            i, data_item, already_print_data_sources = args
            prompt_ids = data_item.batch['prompts']
            prompt_length = prompt_ids.shape[-1]
            
            valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch['responses'] 
            valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            sequences = torch.cat((valid_prompt_ids, valid_response_ids))
            sequences_str = self.tokenizer.decode(sequences)
            # print(sequences_str)

            ground_truth = data_item.non_tensor_batch['reward_model']['ground_truth']

            # select rm_score
            data_source = data_item.non_tensor_batch['data_source']
            # compute_score_fn = prime_compute_score
            # res = compute_score_fn(model_output=sequences_str, ground_truth=ground_truth)
            data_source = data_item.non_tensor_batch[self.reward_fn_key]
            extra_info = data_item.non_tensor_batch.get("extra_info", {})
            score = self.compute_score(
                data_source=data_source,
                solution_str=sequences_str,
                ground_truth=ground_truth,
                extra_info=extra_info,
            )
            if isinstance(score, dict):
                score = score["score"]
                # Store the information including original reward
                for key, value in score.items():
                    reward_extra_info[key].append(value)
            else:
                score = score
            # if score < 1:
            #     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")  
            #     unique_id = uuid.uuid4().hex 
            #     file_name = f"{Failed_RESULTS_DIR}/failed_{timestamp}_{unique_id}.json"

            #     data_to_save = {
            #         "sequences_str": sequences_str,
            #         "ground_truth": ground_truth
            #     }

            #     with open(file_name, "w", encoding="utf-8") as f:
            #         json.dump(data_to_save, f, ensure_ascii=False, indent=4)
            
            # if score >= 1:

            #     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")  
            #     unique_id = uuid.uuid4().hex 
            #     file_name = f"{Correct_RESULTS_DIR}/corrected_{timestamp}_{unique_id}.json"

            #     data_to_save = {
            #         "sequences_str": sequences_str,
            #         "ground_truth": ground_truth
            #     }

            #     with open(file_name, "w", encoding="utf-8") as f:
            #         json.dump(data_to_save, f, ensure_ascii=False, indent=4)
            
            
            return i, score, valid_response_length

        # Process items in parallel using ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=96) as executor:
            args = [(i, data[i], already_print_data_sources) for i in range(len(data))]
            results = list(executor.map(process_item, args))


        # Fill reward tensor with results
        rewards = []
        acc_reward=[]
        completions_length = []
        skipped_idx = []

        # if self.test:
        #     for i, score, valid_response_length in results:
        #         if score < 0:
        #             score = 0.0
        #             # skipped_idx.append(i)
        #             # continue
        #         rewards.append(score)
        #         completions_length.append(valid_response_length)
        #     reweight_rewards = rewards
        # else:
        for i, score, valid_response_length in results:
            acc_reward.append(score)
            if score == -1:
                rewards.append(0.0)
            elif score == 0:
                rewards.append(-1) # control the reward of negative answer
            # elif score == -2:
            #     rewards.append(-1.5) # format penalty, if there are repetition.
            else:
                rewards.append(score)
            completions_length.append(valid_response_length)
        index = data.non_tensor_batch['uid']
        # print(rewards)
        reweight_rewards = self.get_reweight_rewards(rewards, completions_length, index)
        # reweight_rewards = rewards # no reweight

        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)

        mask = torch.ones(reward_tensor.shape[0], dtype=torch.bool)
        mask[skipped_idx] = False
        reward_tensor = reward_tensor[mask]

        skipped_idx = set(skipped_idx)
        cnt_skipped = 0
        for i, score, valid_response_length in results:
            if i in skipped_idx:
                cnt_skipped += 1
                continue
            reward_tensor[i-cnt_skipped, valid_response_length - 1] = reweight_rewards[i-cnt_skipped]
        
        if return_dict:
            return {
                "acc":acc_reward,
                "length":completions_length,
                "reward_tensor": reward_tensor
            }
        else:
            return reward_tensor
        # return reward_tensor
