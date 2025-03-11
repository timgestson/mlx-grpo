import re
import wandb
import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load
from mlx_lm.tuner.trainer import grad_checkpoint
from mlx_lm.models import cache as kv_cache
import mlx.optimizers as optim
from mlx.utils import tree_flatten
from datasets import load_dataset

max_tokens = 1024
epsilon = 1e-6
temperature = 0.9
batch_size = 2
generations = 16

system_prompt = """A conversation between User and Assistant. The user asks a question, and the Assistant solves it.
The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think>
<answer> answer here </answer>
"""


def format_reward(text, _answer):
    think_count = text.count("<think>")
    answer_count = text.count("<answer>")
    has_correct_structure = think_count == 1 and answer_count == 1
    pattern = r"^.*?<think>.*?</think>.*?<answer>.*?</answer>.*?$"
    matches_format = bool(re.search(pattern, text, re.DOTALL))

    return 0.5 if (has_correct_structure and matches_format) else 0


def answer_reward(text, answer):
    match = re.search(rf"<answer>\s*(.*?)\s*</answer>", text, re.DOTALL)
    return (
        1
        if match and match.group(1).strip() == answer
        else 0.5 if match and answer in match.group(1).strip() else 0
    )


def sample(logits, temp, top_p):
    softmax_logits = mx.softmax(logits, axis=-1)
    if temp == 0:
        tokens = mx.argmax(logits, axis=-1, keepdims=True)
    else:
        if top_p > 0 and top_p < 1.0:
            tokens = top_p_sampling(logits, top_p, temp)
        else:
            scaled_logits = logits * (1 / temp)
            tokens = mx.random.categorical(logits * (1 / temp), axis=-1)
            if scaled_logits.ndim > 1:
                tokens = mx.expand_dims(tokens, axis=-1)

    probs = softmax_logits[0, tokens]
    return tokens.squeeze(), probs


def top_p_sampling(logprobs, temp, top_k):
    logprobs = logprobs * (1 / temp)
    mask_idx = mx.argpartition(-logprobs, kth=top_k - 1, axis=-1)[..., top_k:]
    masked_logprobs = mx.put_along_axis(
        logprobs, mask_idx, mx.array(-float("inf"), logprobs.dtype), axis=-1
    )
    return mx.random.categorical(masked_logprobs, axis=-1)


def selective_softmax(logits, tokens):
    softmax_logits = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
    log_probs = mx.take_along_axis(
        softmax_logits, mx.expand_dims(tokens, -1), axis=-1
    ).squeeze(-1)
    return log_probs


def build_rollout(model, tokenizer, dataset, indices, generations):
    prompts = []
    answers = []
    for i in indices:
        question = dataset[i]["question"]
        answer = (
            dataset[i]["answer"].split("####")[1].strip()
            if "####" in dataset[i]["answer"]
            else None
        )
        prompt = tokenizer.apply_chat_template(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question},
            ],
            tokenize=True,
            padding=True,
            padding_side="left",
            return_attention_mask=True,
            add_generation_prompt=True,
        )
        prompts.append(prompt)
        answers += [answer] * generations
    max_len = max(len(p) for p in prompts)
    padded_prompts = [[0] * (max_len - len(p)) + p for p in prompts]
    mask = mx.full([generations * len(indices), len(padded_prompts[0]) + 1], True)
    ended = mx.repeat(mx.array(False), generations * len(indices))
    cache = kv_cache.make_prompt_cache(model)
    logits = []
    for i in padded_prompts:
        temp_cache = kv_cache.make_prompt_cache(model)
        logit = model([prompt], cache=temp_cache)
        for t, c in zip(temp_cache, cache):
            c.keys = mx.concat(
                [c.keys, mx.repeat(t.keys, generations, 0)]
                if c.keys is not None
                else [mx.repeat(t.keys, generations, 0)]
            )
            c.values = mx.concat(
                [c.values, mx.repeat(t.values, generations, 0)]
                if c.values is not None
                else [mx.repeat(t.values, generations, 0)]
            )
            c.offset = t.offset
        logit = logit[:, -1, :]
        logit = mx.repeat(logit, generations, 0)
        logits.append(logit)
    tokens, _ = sample(mx.concat(logits), temperature, 5)
    responses = mx.expand_dims(tokens, 1)
    prompts = mx.concat(
        [
            mx.repeat(mx.expand_dims(mx.array(p), 0), generations, 0)
            for p in padded_prompts
        ]
    )
    return cache, tokens, responses, answers, mask, ended, prompts


def create_generations(model, ref_model, tokenizer, dataset, generations, indices):
    cache, tokens, responses, answers, mask, ended, prompts = build_rollout(
        model, tokenizer, dataset, indices, generations
    )
    for _ in range(max_tokens):
        tokens, _ = sample(
            model(mx.expand_dims(tokens, 1), cache=cache)[:, -1, :], temperature, 5
        )
        ended = mx.logical_or(ended, mx.equal(tokens, 151645))
        mask = mx.concat([mask, mx.expand_dims(ended, 1)], 1)
        responses = mx.concat([responses, mx.expand_dims(tokens, 1)], 1)
        if mx.all(ended):
            break
    decoded = tokenizer.batch_decode(responses.tolist())
    decoded_prompts = tokenizer.batch_decode(prompts.tolist())
    for a, d, p in zip(answers, decoded, decoded_prompts):
        print(p + d.split("<|im_end|>")[0])
        print("---------")
        print(a)

    rewards = mx.array(
        [
            format_reward(response.split("<|im_end|>")[0], answer)
            + answer_reward(response.split("<|im_end|>")[0], answer)
            for response, answer in zip(decoded, answers)
        ]
    )
    baseline = mx.mean(rewards)
    std = mx.sqrt(mx.mean(mx.square(rewards - baseline)) + epsilon)
    advantages = (rewards - baseline) / (std + epsilon)
    full_prompts = mx.concat([prompts, responses], 1)

    logits = ref_model(full_prompts)
    ref_log_probs = selective_softmax(logits[:, :-1, :], full_prompts[:, 1:])
    return {
        "tokens": full_prompts,
        "response_mask": mx.logical_not(mask)[:, 1:],
        "advantages": advantages,
        "ref_log_probs": ref_log_probs,
    }, baseline


def grpo_loss(model, tokens, response_mask, advantages, ref_log_probs):
    beta = 0.4
    logits = model(tokens)[:, :-1, :]
    current_log_probs = selective_softmax(logits, tokens[:, 1:])

    # Expand advantages to match sequence length
    advantages_expanded = mx.expand_dims(advantages, 1)
    advantages_expanded = mx.repeat(
        advantages_expanded, current_log_probs.shape[1], axis=1
    )

    # Weight log probabilities by advantages
    weighted_log_probs = current_log_probs * advantages_expanded

    # KL Term
    kl_ratio = ref_log_probs - current_log_probs
    kl = beta * (kl_ratio.exp() - kl_ratio - 1)

    # Apply response mask to focus only on response tokens
    masked_weighted_log_probs = (
        weighted_log_probs - kl
    ) * response_mask  # .astype(mx.float32)  # Shape: [batch_size, sequence_length - 1]
    sequence_weighted_log_probs = mx.sum(
        masked_weighted_log_probs, axis=1
    ) / mx.maximum(
        mx.sum(response_mask, axis=1), 1e-8
    )  # Shape: [batch_size]

    # Compute the loss as the negative mean of the weighted log probabilities (maximize advantage)
    loss = -mx.mean(sequence_weighted_log_probs)

    return loss


def train(weights=None):
    model, tokenizer = load(
        "Qwen/Qwen2.5-1.5B-Instruct", tokenizer_config={"eos_token": "<|im_end|>"}
    )
    ref_model, tokenizer = load(
        "Qwen/Qwen2.5-1.5B-Instruct", tokenizer_config={"eos_token": "<|im_end|>"}
    )
    ref_model.freeze()
    if weights:
        model.load_weights(weights, strict=False)

    dataset = load_dataset("openai/gsm8k", "main")["train"]
    dataset.shuffle()
    grad_checkpoint(model.layers[0])

    learning_rate = 1e-5
    optimizer = optim.AdamW(
        learning_rate=learning_rate, betas=[0.9, 0.95], weight_decay=0.1
    )

    run = wandb.init()

    for step in range(0, 100, batch_size):
        print("Generating set")
        gens, averageReward = create_generations(
            model,
            ref_model,
            tokenizer,
            dataset,
            generations,
            range(step, step + batch_size),
        )

        value_and_grad_fn = nn.value_and_grad(model, grpo_loss)
        loss, grads = value_and_grad_fn(model, **gens)
        run.log({"avg_reward": averageReward.item(), "loss": loss.item()})

        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state)

        adapter_weights = dict(tree_flatten(model.trainable_parameters()))
        mx.save_safetensors(f"qwen-rl.safetensors", adapter_weights)


train()
