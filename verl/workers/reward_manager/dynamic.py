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
import collections
import torch
import math as math_o
from verl import DataProto
from verl.utils.reward_score import default_compute_score
from verl.workers.reward_manager import register
from verl.workers.reward_manager.abstract import AbstractRewardManager


@register("dynamic")
class DynamicRewardManager(AbstractRewardManager):
    """The reward manager."""

    def __init__(self, tokenizer, num_examine, compute_score=None, reward_fn_key="data_source") -> None:
        """
        Initialize the DynamicRewardManager instance.

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
    def cal_acc(self,
            token_level_rewards: torch.Tensor,
            index: torch.Tensor,
            input_ids,
            ground_truth,
            completions_length,
            epsilon: float = 1e-6):
        """
        Compute accuracy for every rollouts.
        Args:
            token_level_rewards: `(torch.Tensor)`
                shape: (bs, response_length)
        
        Returns:
            advantages: `(torch.Tensor)`
                shape: (bs, response_length)
            Returns: `(torch.Tensor)`
                shape: (bs, response_length)
        """
        # Modified advatages, reweight based on correct ratio.
        response_length = token_level_rewards.shape[-1]
        non_zero_mask = (token_level_rewards != 0)
        scores = (token_level_rewards * non_zero_mask).sum(dim=-1)

        id2score = defaultdict(list)
        id2correct_num = defaultdict(list)
        id2input = {}
        id2ground_truth= {}
        id2correct_ratio = {}
        id2mean = {}
        id2std = {}

        with torch.no_grad():
            bsz = scores.shape[0]
            for i in range(bsz):
                id2score[index[i]].append(scores[i])
                id2correct_num[index[i]].append(int(scores[i] > 0))
                if index[i] not in id2input:
                    id2input[index[i]] = input_ids[i]
                    id2ground_truth[index[i]] = ground_truth[i]
            for idx in id2score:
                if len(id2score[idx]) == 1:
                    id2mean[idx] = torch.tensor(0.0)
                    id2std[idx] = torch.tensor(1.0)
                elif len(id2score[idx]) > 1:
                    id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))
                    id2std[idx] = torch.std(torch.tensor(id2score[idx]))
                else:
                    raise ValueError(f"no score in prompt index: {idx}")
            for index_i in id2correct_num:
                group_size = len(id2correct_num[index_i])
                id2correct_ratio[index_i] = torch.tensor(sum(id2correct_num[index_i]) / group_size).to(scores.device)
            return id2correct_ratio
    def get_reweight_rewards(self, rewards, completions_length, max_response_len,index,id2correct_ratio):
        """
        rewards: list of rewards (1-d tensor)
        completions_length: list of completions length (1-d tensor)
        index: list indicating which question each sample belongs to
        """
        reweight_rewards = [0]*len(rewards)
        unique_questions = set(index)
        length_penalty=[0]*len(rewards)
        length_reward=[0]*len(rewards)
        length_exceed_penalty=[0]*len(rewards)
        sum_len_per_acc=collections.defaultdict(int)
        num_per_acc=collections.defaultdict(int)
        var_per_acc=collections.defaultdict(int)
        std_per_acc=collections.defaultdict(int)
        
        #计算不同acc难度问题的length_sum和num
        for q in unique_questions:
            group_indices = [i for i, q_idx in enumerate(index) if q_idx == q]
            acc_q = id2correct_ratio[q]
            for i in group_indices:
                L = completions_length[i]
                if rewards[i] > 0:
                    sum_len_per_acc[acc_q]+=L
                    num_per_acc[acc_q]+=1
                if acc_q==0:
                    sum_len_per_acc[acc_q]+=L
                    num_per_acc[acc_q]+=1

        # # 计算std
        # for q in unique_questions:
        #     group_indices = [i for i, q_idx in enumerate(index) if q_idx == q]
        #     acc_q = id2correct_ratio[q]
        #     avg_len = sum_len_per_acc[acc_q] / num_per_acc[acc_q] if num_per_acc[acc_q] > 0 else 0.0
        #     var_len = 0
        #     if num_per_acc[acc_q]>1:
        #         for i in group_indices:
        #             if rewards[i] > 0:
        #                 var_per_acc[acc_q] += (completions_length[i] - avg_len) ** 2
        #             if acc_q==0:
        #                 var_per_acc[acc_q] += (completions_length[i] - avg_len) ** 2
        # for acc_q in var_per_acc.keys():
        #     if num_per_acc[acc_q] > 1:
        #         var_per_acc[acc_q] = var_per_acc[acc_q] / (num_per_acc[acc_q]-1)
        #         std_per_acc[acc_q] = math_o.sqrt(var_per_acc[acc_q])
        #     else:
        #         var_per_acc[acc_q] = 0.0
        #         std_per_acc[acc_q] = 1.0

        for q in unique_questions:
            group_indices = [i for i, q_idx in enumerate(index) if q_idx == q]
            acc_q = id2correct_ratio[q]
            # # 初始化统计量
            # sum_len = 0
            # var_len = 0
            # num_correct = 0
            
            # 计算本样本的中正确回答长度的mean
            if acc_q>0:
                right_length=0
                right_cnt=0
                for i in group_indices:
                    L = completions_length[i]
                    if rewards[i] > 0:
                        right_length+=L
                        right_cnt+=1
                avg_q_len = right_length/right_cnt
            else:
                avg_q_len=0

            # 计算每个 acc 的 mean 和 std
            avg_len = sum_len_per_acc[acc_q]/num_per_acc[acc_q] if num_per_acc[acc_q] > 0 else 0.
            similar_acc = float(1/len(group_indices))
            avg_len_similar = sum_len_per_acc[similar_acc]/num_per_acc[similar_acc] if num_per_acc[similar_acc] > 0 else 0
            sum_acc=0
            num_acc=0
            for acc in sum_len_per_acc.keys():
                sum_acc+=sum_len_per_acc[acc]
                num_acc+=num_per_acc[acc]
            avg_len_all = sum_acc/num_acc if num_acc>0 else 0

            # std_len = std_per_acc[acc_q]

            # 本prompt中所有正确回答的平均长度和标准差
            # avg_len = sum_len / num_correct if num_correct > 0 else 0.0
            # std_len = 1.0
            # if num_correct > 1:
            #     for i in group_indices:
            #         if rewards[i] > 0:
            #             var_len += (completions_length[i] - avg_len) ** 2
            #     val_len = var_len / (num_correct - 1)
            #     std_len = math_o.sqrt(val_len)
                
            # alpha = 0.05
            threshold = 0.2
            alpha_penalty = acc_q
            alpha_reward = threshold - acc_q
            # alpha_penalty = 1
            # alpha_reward = 1
            # min_three_avg = (min1 + min2 + min3) / 3
            
            for i in group_indices:
                L_i = completions_length[i]
                reweight_rewards[i] = rewards[i]
                if acc_q >= threshold: ## threshold = 0.2
                    # z_i = (L_i - avg_len_all) / max_response_len
                    z_i = (L_i - avg_len) / max_response_len

                    lower_bar = 2500
                    if L_i >= lower_bar:
                        reweight_rewards[i] = rewards[i] - alpha_penalty * z_i
                        # length_penalty[i] = alpha_penalty * z_i
                    else:
                        # reweight_rewards[i] = rewards[i] + alpha_penalty * (L_i - 2500) / max_response_len
                        # reweight_rewards[i] = rewards[i] + alpha_penalty * z_i
                        reweight_rewards[i] = rewards[i]
                else:
                    upper_bar = 10000   #超出bar的给penalty，越接近bar的reward越大
                    # z_i = L_i / max_response_len #v2
                    # z_i = (L_i - avg_len) / max_response_len # v1
                    if acc_q!=0:
                        z_i = L_i - avg_len / max_response_len #v2
                        # z_i = (L_i - avg_len_all) / max_response_len #0.02左右
                    else:
                        z_i = L_i -  avg_len_similar/ max_response_len
                        # z_i = L_i -  avg_len_all/ max_response_len
                    if L_i < upper_bar:
                        #ablation study
                        # reweight_rewards[i] = rewards[i] + alpha_reward * (L_i - avg_len) / max_response_len
                        reweight_rewards[i] = rewards[i] + alpha_reward * z_i
                    else:
                        # reweight_rewards[i] = rewards[i] - alpha_reward * z_i
                        # reweight_rewards[i] = rewards[i] - alpha_reward * (10000 - L_i) / max_response_len
                        reweight_rewards[i] = rewards[i]

        return reweight_rewards,length_penalty,length_reward,length_exceed_penalty

    def __call__(self, data: DataProto, global_steps: int = 0,return_dict: bool = False) -> torch.Tensor | dict[str, Any]:
        """We will expand this function gradually based on the available datasets"""

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if "rm_scores" in data.batch.keys():
            if return_dict:
                reward_extra_keys = data.meta_info.get("reward_extra_keys", [])
                reward_extra_info = {key: data.non_tensor_batch[key] for key in reward_extra_keys}
                return {"reward_tensor": data.batch["rm_scores"], "reward_extra_info": reward_extra_info}
            else:
                return data.batch["rm_scores"]
        
        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info = defaultdict(list)

        already_print_data_sources = {}
        completions_length=[]
        rewards = []
        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem

            prompt_ids = data_item.batch["prompts"]

            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch["responses"]
            valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
            completions_length.append(valid_response_length.item())
            valid_response_ids = response_ids[:valid_response_length]
            
            # decode
            prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)

            ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]
            data_source = data_item.non_tensor_batch[self.reward_fn_key]
            extra_info = data_item.non_tensor_batch.get("extra_info", {})
            num_turns = data_item.non_tensor_batch.get("__num_turns__", None)
            extra_info["num_turns"] = num_turns

            score = self.compute_score(
                data_source=data_source,
                solution_str=response_str,
                ground_truth=ground_truth,
                extra_info=extra_info,
            )

            if isinstance(score, dict):
                reward = score["score"]
                # Store the information including original reward
                for key, value in score.items():
                    reward_extra_info[key].append(value)
            else:
                reward = score
            rewards.append(reward)
            
            reward_tensor[i, valid_response_length - 1] = reward
            

            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print("[prompt]", prompt_str)
                print("[response]", response_str)
                print("[ground_truth]", ground_truth)
                if isinstance(score, dict):
                    for key, value in score.items():
                        print(f"[{key}]", value)
                else:
                    print("[score]", score)
        index = data.non_tensor_batch['uid']
        input_ids = data.batch['prompts'] #added for saveing correctness
        max_response_len = data.batch['responses'].shape[-1]
        ground_truth = [
            item.non_tensor_batch.get("reward_model", {}).get("ground_truth", None) for item in data
        ]
        id2correct = self.cal_acc(reward_tensor, index,input_ids,ground_truth,completions_length)
        reweight_rewards,length_penalty,length_reward,length_exceed_penalty = self.get_reweight_rewards(rewards, completions_length, max_response_len,index,id2correct)
        reweight_reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        for i in range(len(data)):
            valid_response_length = completions_length[i]
            reweight_reward_tensor[i, valid_response_length - 1] = reweight_rewards[i]
        self._step_counter = 0
        if global_steps !=0:
            self._step_counter = global_steps

        print("step_now:",self._step_counter)
        print("self.num_examine:",self.num_examine)

        if return_dict:
            return {
                "acc":rewards,
                "length":completions_length,
                "reward_tensor": reweight_reward_tensor,
                "reward_extra_info": reward_extra_info,
            }
        else:
            return reweight_reward_tensor

