"""
ppo相关函数
"""
import torch
from megatron import print_rank_0
import time


def generate(model, input_tensor, max):
  """利用模型与输入生成样本

  Args:
      model (_type_): _description_
      input_tensor (_type_): _description_
  """
  pass


def make_experience(self,
                    prompt_iterator,
                    model,
                    num_rollouts: int = 1024,
                    iter_count: int = 0):  # noqa:
  """Make experiences

    Takes `chunk_size` number of prompts from `prompt_iterator`, samples
    from the model and then computes the KL against a reference model. Finally it
    then appends PPOElements to trainer's `store`.

    Args:
        num_rollouts: Number of rollouts to generate
        iter_count: Total number of updates run (i.e. number of updates run for all batches & epochs)
    """
  print_rank_0("Collecting rollouts")
  # tbar = logging.tqdm(
  #     total=num_rollouts,
  #     disable=os.environ.get("RANK", 0) != "0",
  #     desc=f"[rollout 0 / {num_rollouts}]",
  #     # Lower progress bar by 1 if we're in WARNING mode or above to avoid hiding high priority progress
  #     # bars (e.g. loss progress in trainers)
  #     position=logging.get_verbosity() >= logging.WARNING,
  #     # Leave progress bar if we're in INFO mode or lower to avoid spamming in suppressed verbosity levels
  #     leave=logging.get_verbosity() < logging.WARNING,
  # )

  ppo_rl_elements = []
  stats = {}
  # clock = Clock()

  while len(ppo_rl_elements) < num_rollouts:
    # Get next batch in prompt dataset and refresh if exhausted
    # TOOD (jon-tow): Make `prompt_dataloader` a cyclic/infinite DataLoader to not require manually
    # "refreshing" the contents of the `prompt_iterator`

    # list[dict]
    prompt_batch = next(prompt_iterator)
    assert len(prompt_batch) == 1
    prompt_batch = prompt_batch[0]

    # try:
    #     batch: PromptBatch = next(self.prompt_iterator)
    # except StopIteration:
    #     self.prompt_iterator = iter(self.prompt_dataloader)
    #     batch = next(self.prompt_iterator)

    exp_generate_time = time()

    # Generate samples from the language model (similar to using HuggingFace `generate` method)
    samples = generate(**batch)
    stats["time/exp_generate"] = time() - exp_generate_time

    prompt_tensors = batch.input_ids
    device = samples.device

    prompt_sizes = torch.tensor([prompt_tensors.shape[1]] *
                                len(prompt_tensors),
                                device=device)
    padded_samples = self.accelerator.pad_across_processes(
        samples, dim=1, pad_index=self.tokenizer.eos_token_id, pad_first=False)
    padded_prompts = self.accelerator.pad_across_processes(
        prompt_tensors,
        dim=1,
        pad_index=self.tokenizer.eos_token_id,
        pad_first=False)
    gathered_samples = self.accelerator.gather(padded_samples)
    gathered_prompts = self.accelerator.gather(padded_prompts)
    gathered_prompt_sizes = self.accelerator.gather(prompt_sizes)

    if self.accelerator.is_main_process:
      all_str_samples, all_str_prompts, all_str_outputs = self.decode(
          gathered_prompts, gathered_samples, gathered_prompt_sizes)

      exp_score_time = time()
      all_scores = torch.tensor(
          self.reward_fn(
              samples=all_str_samples,
              prompts=all_str_prompts,
              outputs=all_str_outputs,
          ),
          dtype=torch.float,
          device=device,
      )
      stats["time/exp_score"] = time() - exp_score_time

      all_scores = list(
          all_scores.reshape(self.accelerator.num_processes, -1).unbind())
    else:
      all_scores = None

    if torch.distributed.is_initialized():
      scores = torch.empty(len(samples), device=device)
      torch.distributed.scatter(scores, all_scores)
    else:
      scores = torch.tensor(all_scores[0])

    str_samples, str_prompts, str_outputs = self.decode(
        prompt_tensors, samples)

    # Pad the sample outputs
    outputs = self.tokenizer(str_outputs).input_ids
    if self.config.model.model_arch_type == "seq2seq":
      # add <pad> to the start of the output
      for i in range(len(outputs)):
        outputs[i] = [self.tokenizer.pad_token_id] + outputs[i]

    outputs = list(map(torch.LongTensor, outputs))
    maxsize = max(map(len, outputs))
    outputs = [
        F.pad(
            output,
            (0, maxsize - len(output)),
            value=self.tokenizer.pad_token_id,
        ) for output in outputs
    ]
    sample_outputs = torch.vstack(outputs).to(device)

    # store statistics of the initial rollout as reference
    if self.ref_mean is None:
      self.ref_mean, self.ref_std = scores.mean(), scores.std()
    all_scores_mean, all_scores_std = self.running_moments.update(scores)
    stats["exp_scores/mean"] = all_scores_mean
    stats["exp_scores/std"] = all_scores_std
    stats["exp_scores/running_mean"] = self.running_moments.mean
    stats["exp_scores/running_std"] = self.running_moments.std

    if self.config.method.scale_reward == "running":
      scores /= self.running_moments.std
    elif self.config.method.scale_reward == "ref":
      scores /= self.ref_std

    clip_reward = self.config.method.cliprange_reward
    if clip_reward:
      scores = torch.clip(scores, -clip_reward, clip_reward)

    # Precompute logprobs, values
    if self.config.model.model_arch_type == "seq2seq":
      attention_mask = batch.attention_mask.to(device)
      prompt_tensors = batch.input_ids.to(device)
      decoder_attention_mask = sample_outputs.not_equal(
          self.tokenizer.pad_token_id)
      decoder_attention_mask[:, 0] = 1
      with torch.no_grad():
        outputs = self.model(
            input_ids=prompt_tensors,
            attention_mask=attention_mask,
            decoder_input_ids=sample_outputs,
            decoder_attention_mask=decoder_attention_mask,
        )
        logits = outputs.logits
        values = outputs.value
        if hasattr(self.model, "frozen_head"):
          ref_logits = self.model.forward_hydra(
              input_ids=prompt_tensors,
              attention_mask=attention_mask,
              decoder_input_ids=sample_outputs,
              decoder_attention_mask=decoder_attention_mask,
              return_dict=True,
          ).logits
        else:
          ref_logits = self.ref_model(
              input_ids=prompt_tensors,
              attention_mask=attention_mask,
              decoder_input_ids=sample_outputs,
              decoder_attention_mask=decoder_attention_mask,
              return_dict=True,
          ).logits
    else:
      all_tokens = torch.cat((prompt_tensors.to(device), sample_outputs),
                             dim=1)
      attention_mask = all_tokens.not_equal(
          self.tokenizer.pad_token_id).long().to(device)
      with torch.no_grad():
        logits, *_, values = self.model(
            all_tokens,
            attention_mask=attention_mask,
        )
        # TODO(dahoas): When hydra model works need to also support generation on hydra head
        if hasattr(self.model, "frozen_head"):
          ref_logits = self.model.forward_hydra(
              all_tokens,
              attention_mask=attention_mask,
              return_dict=True,
          ).logits
        else:
          ref_logits = self.ref_model(
              all_tokens,
              attention_mask=attention_mask,
              return_dict=True,
          ).logits
          ref_logits = ref_logits.to(device)

    if self.config.model.model_arch_type == "seq2seq":
      logprobs = logprobs_of_labels(logits[:, :-1, :], sample_outputs[:, 1:])
      ref_logprobs = logprobs_of_labels(ref_logits[:, :-1, :],
                                        sample_outputs[:, 1:])
    else:
      logprobs = logprobs_of_labels(logits[:, :-1, :], all_tokens[:, 1:])
      ref_logprobs = logprobs_of_labels(ref_logits[:, :-1, :], all_tokens[:,
                                                                          1:])

    n_samples: int = samples.shape[0]
    logprobs = logprobs.cpu()
    ref_logprobs = ref_logprobs.cpu()
    prompt_tensors = prompt_tensors.cpu()
    sample_outputs = sample_outputs.cpu()

    # Estimate the KL divergence between the model and reference model
    if self.config.model.model_arch_type == "seq2seq":
      values = values.cpu()[:, :-1]
      start = 0

      # Get the number of non-padding tokens for each sample
      # This assumes all padding is on the right side
      padding_token: int = 0
      ends = (sample_outputs[:, start:] != padding_token).sum(1)

      # Get the logprobs and values, for tokens that are not padding
      # or beginning of sequences tokens. These are from the model
      # (not the reference model)
      all_logprobs = [logprobs[ix, start:ends[ix]] for ix in range(n_samples)]
      all_values = [values[ix, start:ends[ix]] for ix in range(n_samples)]

      kl_divergence_estimate: List[torch.Tensor] = [
          -self.kl_ctl.value *
          (logprobs[sample_idx, start:ends[sample_idx]] -
           ref_logprobs[sample_idx, start:ends[sample_idx]])
          for sample_idx in range(n_samples)
      ]

    # Else if not seq2seq (i.e. causal)
    else:
      values = values.cpu()[:, :-1]
      start = prompt_tensors.shape[1] - 1
      ends = start + attention_mask[:, start:].sum(1)
      all_values = [values[ix, start:ends[ix]] for ix in range(n_samples)]
      all_logprobs = [logprobs[ix, start:ends[ix]] for ix in range(n_samples)]

      kl_divergence_estimate = -self.kl_ctl.value * (logprobs - ref_logprobs)
      kl_divergence_estimate = [
          rs[start:ends[ix]] for ix, rs in enumerate(kl_divergence_estimate)
      ]

    rollout_count = 0

    for sample_idx in range(n_samples):
      sample_kl_divergence_estimate = kl_divergence_estimate[sample_idx]

      if len(sample_kl_divergence_estimate) == 0 or len(
          all_logprobs[sample_idx]) == 0:
        continue

      rewards = sample_kl_divergence_estimate
      rewards[-1] += scores[sample_idx].cpu()

      ppo_rl_elements.append(
          PPORLElement(
              query_tensor=prompt_tensors[sample_idx],
              response_tensor=sample_outputs[sample_idx],
              logprobs=all_logprobs[sample_idx],
              values=all_values[sample_idx],
              rewards=rewards,
          ))

      rollout_count += 1
    exp_time = clock.tick()
    tbar.set_description(f"[rollout {len(ppo_rl_elements)} / {num_rollouts}]")
    tbar.update(min(rollout_count, num_rollouts))
  tbar.close()

  stats["kl_ctl_value"] = self.kl_ctl.value
  stats["time/exp"] = exp_time

  if not ray.is_initialized():
    self.accelerator.log(stats, step=iter_count)

  # Push samples and rewards to trainer's rollout storage
  self.push_to_store(ppo_rl_elements)
