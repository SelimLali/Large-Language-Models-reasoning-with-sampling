import argparse
import json
import math
import os
import random
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


# -------------------------
# Answer extraction helpers
# -------------------------
_BOX_RE = re.compile(r"\\boxed\{([^}]*)\}")

def extract_boxed(text: str) -> Optional[str]:
    matches = _BOX_RE.findall(text)
    if matches:
        return matches[-1].strip()
    return None

def extract_final_answer(text: str) -> str:
    boxed = extract_boxed(text)
    if boxed is not None:
        return boxed

    m = re.search(r"(final answer|answer)\s*[:\-]\s*(.+)$", text, flags=re.IGNORECASE | re.MULTILINE)
    if m:
        return m.group(2).strip()

    m2 = re.search(r"####\s*(.+)$", text, flags=re.MULTILINE)
    if m2:
        return m2.group(1).strip()

    lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
    return lines[-1].strip() if lines else ""

def normalize_answer(ans: str) -> str:
    ans = ans.strip().replace("$", "")
    ans = ans.replace("\\left", "").replace("\\right", "")
    ans = ans.replace("\\,", "").replace("\\ ", "")
    ans = re.sub(r"\s+", "", ans)
    ans = re.sub(r"^\\boxed\{", "", ans)
    ans = ans[:-1] if ans.endswith("}") else ans
    if len(ans) >= 2 and ans[0] == "(" and ans[-1] == ")":
        ans = ans[1:-1]
    return ans


# -------------------------
# Device + seeding
# -------------------------
def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# -------------------------
# Prompt formatting
# -------------------------
def build_messages(problem: str) -> List[Dict[str, str]]:
    # Keep it minimal and consistent across instruct models.
    return [
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": (
                "Solve the following math problem. "
                "Please reason step by step, and put your final answer within \\boxed{}.\n\n"
                f"Problem:\n{problem}"
            ),
        },
    ]

def encode_prompt(tok, problem: str, use_chat_template: bool) -> Tuple[torch.Tensor, str]:
    msgs = build_messages(problem)
    if use_chat_template and hasattr(tok, "apply_chat_template"):
        prompt_text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    else:
        prompt_text = (
            "You are a helpful assistant.\n"
            "Solve the following math problem. Please reason step by step, and put your final answer within \\boxed{}.\n\n"
            f"Problem:\n{problem}\n"
        )
    prompt_ids = tok(prompt_text, return_tensors="pt").input_ids
    return prompt_ids, prompt_text


# -------------------------
# Logprob + entropy utilities
# -------------------------
@torch.no_grad()
def logprob_sum(model, input_ids: torch.Tensor, temperature: float = 1.0) -> float:
    """
    Sum_t log p(x_t | x_<t) over all tokens except the first.
    Temperature is applied as softmax(logits / temperature).
    """
    out = model(input_ids=input_ids)
    logits = out.logits[:, :-1, :] / max(temperature, 1e-8)
    target = input_ids[:, 1:]
    logp = torch.log_softmax(logits, dim=-1)
    token_logp = logp.gather(-1, target.unsqueeze(-1)).squeeze(-1)  # [1, L-1]
    return float(token_logp.sum().item())

@torch.no_grad()
def logprob_suffix_sum(model, full_ids: torch.Tensor, prefix_len: int, temperature: float = 1.0) -> float:
    """
    Sum logprobs for tokens in the suffix starting at position prefix_len (inclusive),
    i.e., tokens >= prefix_len, conditioned on previous context.
    """
    out = model(input_ids=full_ids)
    logits = out.logits[:, :-1, :] / max(temperature, 1e-8)
    target = full_ids[:, 1:]
    logp = torch.log_softmax(logits, dim=-1)
    token_logp = logp.gather(-1, target.unsqueeze(-1)).squeeze(-1)

    start = max(prefix_len - 1, 0)  # token at prefix_len uses logits at prefix_len-1
    return float(token_logp[:, start:].sum().item())

@torch.no_grad()
def avg_entropy_over_completion(model, full_ids: torch.Tensor, prompt_len: int) -> float:
    """
    Average entropy of next-token distributions used to generate completion tokens.
    For completion tokens positions t in [prompt_len, L-1], distribution is logits at t-1.
    So we average entropies for logits indices [prompt_len-1, L-2].
    """
    out = model(input_ids=full_ids)
    logits = out.logits[:, :-1, :]  # [1, L-1, V]
    Lm1 = logits.shape[1]
    start = max(prompt_len - 1, 0)
    if start >= Lm1:
        return 0.0

    sub = logits[:, start:, :]  # [1, *, V]
    logp = torch.log_softmax(sub, dim=-1)
    p = torch.softmax(sub, dim=-1)
    ent = -(p * logp).sum(dim=-1)  # [1, *]
    return float(ent.mean().item())


# -------------------------
# Generation helpers (exact token count)
# -------------------------
@torch.no_grad()
def generate_exact(
    model,
    tokenizer,
    prefix_ids: torch.Tensor,
    new_tokens: int,
    temperature: float,
    top_p: float,
) -> torch.Tensor:
    """
    Force generation of exactly `new_tokens` tokens by setting min_new_tokens=max_new_tokens.
    This is important for the block schedule and MH ratios (fixed length proposals).
    """
    if new_tokens <= 0:
        return prefix_ids

    return model.generate(
        prefix_ids,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        min_new_tokens=new_tokens,
        max_new_tokens=new_tokens,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )


# -------------------------
# Blockwise Power Sampling (B schedule)
# -------------------------
@torch.no_grad()
def power_sampling_blockwise_mh(
    model,
    tokenizer,
    prompt_ids: torch.Tensor,
    completion_len: int,
    alpha: float,
    B: int,
    proposal_temperature: float,
    top_p: float,
    n_mcmc: int,
) -> Tuple[torch.Tensor, float]:
    """
    Block schedule (B) closer to the paper:
      for k=1..K:
        - extend to length kB (completion tokens) with proposal
        - run n_mcmc MH steps within that block length
    """
    prompt_len = prompt_ids.shape[1]
    K = math.ceil(completion_len / B)
    current = prompt_ids.clone()

    total_steps = 0
    accepted = 0

    # cache current base logp (temp=1.0) for current length
    cur_logp = logprob_sum(model, current, temperature=1.0)

    for k in range(1, K + 1):
        target_comp = min(k * B, completion_len)
        target_total_len = prompt_len + target_comp

        # 1) extend with proposal to reach target_total_len
        if current.shape[1] < target_total_len:
            need = target_total_len - current.shape[1]
            current = generate_exact(model, tokenizer, current, need, proposal_temperature, top_p)
            cur_logp = logprob_sum(model, current, temperature=1.0)

        # 2) MH resampling within [1..target_comp]
        for _ in range(n_mcmc):
            total_steps += 1
            comp_now = target_comp
            if comp_now <= 1:
                break

            m = random.randint(1, comp_now)  # keep m completion tokens, resample remainder
            prefix_len = prompt_len + m
            prefix = current[:, :prefix_len]

            # propose suffix to reach target_total_len
            need = target_total_len - prefix_len
            proposed = generate_exact(model, tokenizer, prefix, need, proposal_temperature, top_p)

            prop_logp = logprob_sum(model, proposed, temperature=1.0)

            # proposal ratio uses suffix probs under proposal temp
            logq_old = logprob_suffix_sum(model, current, prefix_len=prefix_len, temperature=proposal_temperature)
            logq_new = logprob_suffix_sum(model, proposed, prefix_len=prefix_len, temperature=proposal_temperature)

            # log acceptance = alpha*(logp_prop-logp_cur) + (logq_old - logq_new)
            log_accept = alpha * (prop_logp - cur_logp) + (logq_old - logq_new)

            if math.log(random.random()) < min(0.0, log_accept):
                current = proposed
                cur_logp = prop_logp
                accepted += 1

    accept_rate = accepted / max(total_steps, 1)
    return current, accept_rate


# -------------------------
# Dataset loading
# -------------------------
def ensure_math500_jsonl(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    out = path / "test.jsonl"
    if out.exists() and out.stat().st_size > 0:
        return out
    from datasets import load_dataset
    ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    with out.open("w", encoding="utf-8") as f:
        for ex in ds:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    return out

def load_jsonl(path: Path) -> List[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows

def shard_list(items: List[dict], shard_id: int, num_shards: int) -> List[dict]:
    assert 0 <= shard_id < num_shards
    return [ex for i, ex in enumerate(items) if (i % num_shards) == shard_id]


# -------------------------
# Output schema
# -------------------------
@dataclass
class Row:
    task: str
    model: str
    method: str
    alpha: float
    B: int
    n_mcmc: int
    temperature: float
    top_p: float
    seed: int
    shard_id: int
    example_id: str
    prompt: str
    completion: str
    gold: str
    pred: str
    correct: int
    # diagnostics
    accept_rate: float
    num_tokens_generated: int
    base_loglik_sum: float
    base_loglik_avg: float
    avg_entropy: float


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="Qwen/Qwen2.5-Math-1.5B-Instruct")
    ap.add_argument("--method", type=str, choices=["base", "lowtemp", "power_mh"], default="base")

    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--shard_id", type=int, default=0)
    ap.add_argument("--num_shards", type=int, default=1)
    ap.add_argument("--max_examples", type=int, default=0, help="0 = all examples in this shard")

    ap.add_argument("--max_new_tokens", type=int, default=512)
    ap.add_argument("--top_p", type=float, default=1.0)

    ap.add_argument("--temperature", type=float, default=1.0, help="Used for base; for lowtemp/power we use 1/alpha by default")
    ap.add_argument("--alpha", type=float, default=4.0)

    # block schedule params (paper-style B=192)
    ap.add_argument("--B", type=int, default=192)
    ap.add_argument("--n_mcmc", type=int, default=10)

    ap.add_argument("--results_dir", type=str, default="results/math500/raw")
    ap.add_argument("--use_chat_template", action="store_true")
    args = ap.parse_args()

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    set_seed(args.seed)
    device = pick_device()
    print(f"[Device] {device}")

    # data
    data_path = ensure_math500_jsonl(Path("data") / "math500")
    all_rows = load_jsonl(data_path)
    shard = shard_list(all_rows, args.shard_id, args.num_shards)
    if args.max_examples and args.max_examples > 0:
        shard = shard[: args.max_examples]
    print(f"[Data] shard {args.shard_id}/{args.num_shards} -> {len(shard)} examples")

    # model
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id

    dtype = torch.float16 if device.type in ("cuda", "mps") else torch.float32
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype, trust_remote_code=True)
    model.to(device)
    model.eval()

    out_rows: List[Row] = []
    task_name = "MATH500"

    for ex in tqdm(shard, desc="Generating"):
        problem = ex["problem"]
        gold = str(ex.get("answer", "")).strip()
        example_id = str(ex.get("unique_id", ""))

        prompt_ids_cpu, prompt_text = encode_prompt(tok, problem, args.use_chat_template)
        prompt_ids = prompt_ids_cpu.to(device)

        prompt_len = prompt_ids.shape[1]
        completion_len = int(args.max_new_tokens)

        if args.method == "base":
            temperature = float(args.temperature)
            full_ids = generate_exact(model, tok, prompt_ids, completion_len, temperature, args.top_p)
            accept_rate = 0.0

        elif args.method == "lowtemp":
            temperature = float(1.0 / max(args.alpha, 1e-8))
            full_ids = generate_exact(model, tok, prompt_ids, completion_len, temperature, args.top_p)
            accept_rate = 0.0

        else:  # power_mh
            proposal_temperature = float(1.0 / max(args.alpha, 1e-8))
            full_ids, accept_rate = power_sampling_blockwise_mh(
                model=model,
                tokenizer=tok,
                prompt_ids=prompt_ids,
                completion_len=completion_len,
                alpha=float(args.alpha),
                B=int(args.B),
                proposal_temperature=proposal_temperature,
                top_p=float(args.top_p),
                n_mcmc=int(args.n_mcmc),
            )
            temperature = proposal_temperature  # store proposal temp in CSV (matches their style)

        # decode completion
        completion_ids = full_ids[0, prompt_len:]
        completion_text = tok.decode(completion_ids, skip_special_tokens=True)

        pred = extract_final_answer(completion_text)
        correct = int(normalize_answer(pred) == normalize_answer(gold))

        # diagnostics under base model
        base_loglik_sum = logprob_suffix_sum(model, full_ids, prefix_len=prompt_len, temperature=1.0)
        base_loglik_avg = base_loglik_sum / max(int(completion_ids.shape[0]), 1)
        avg_ent = avg_entropy_over_completion(model, full_ids, prompt_len=prompt_len)

        out_rows.append(Row(
            task=task_name,
            model=args.model,
            method=args.method,
            alpha=float(args.alpha),
            B=int(args.B if args.method == "power_mh" else 0),
            n_mcmc=int(args.n_mcmc if args.method == "power_mh" else 0),
            temperature=float(temperature),
            top_p=float(args.top_p),
            seed=int(args.seed),
            shard_id=int(args.shard_id),
            example_id=example_id,
            prompt=prompt_text,
            completion=completion_text,
            gold=gold,
            pred=pred,
            correct=correct,
            accept_rate=float(accept_rate),
            num_tokens_generated=int(completion_ids.shape[0]),
            base_loglik_sum=float(base_loglik_sum),
            base_loglik_avg=float(base_loglik_avg),
            avg_entropy=float(avg_ent),
        ))

    # write csv
    Path(args.results_dir).mkdir(parents=True, exist_ok=True)
    safe_model = args.model.replace("/", "_")
    run_name = f"{task_name}__{safe_model}__{args.method}"
    out_path = Path(args.results_dir) / f"{run_name}__seed{args.seed}__shard{args.shard_id}of{args.num_shards}.csv"

    df = pd.DataFrame([asdict(r) for r in out_rows])
    df.to_csv(out_path, index=False)
    print(f"[OK] Wrote {len(df)} rows -> {out_path}")


if __name__ == "__main__":
    main()