"""
ppo 相关的数据集实现
"""
from torch.utils.data import Dataset

class RolloutDataset(Dataset):
  def __init__(self,
  input_tokens_list,
  old_logprobs_list,
  old_values_list,
  old_rewards_list,
  query_tensors_list,
  input_tokens_only=False
  ) -> None:
    self.input_tokens_only = input_tokens_only
    if not self.input_tokens_only:
      assert len(input_tokens_list) == len(old_logprobs_list)
      assert len(old_logprobs_list) == len(old_values_list)
      assert len(old_values_list) == len(old_rewards_list)
      assert len(old_rewards_list) == len(query_tensors_list)
    self.input_tokens_list = input_tokens_list
    self.old_logprobs_list = old_logprobs_list
    self.old_values_list = old_values_list
    self.old_rewards_list = old_rewards_list
    self.query_tensors_list = query_tensors_list

  
  def __getitem__(self, i):
    if not self.input_tokens_only:
      return {
        'input_tokens': self.input_tokens_list[i],
        'old_logprobs': self.old_logprobs_list[i],
        'old_values': self.old_values_list[i],
        'old_rewards': self.old_rewards_list[i],
        'query_tensors':self.query_tensors_list[i]
      }
    else:
      # print('datasetxxxx',self.input_tokens_list[i].shape)
      return {
        'input_tokens': self.input_tokens_list[i]
      }

  def __len__(self):
    return len(self.input_tokens_list)



    