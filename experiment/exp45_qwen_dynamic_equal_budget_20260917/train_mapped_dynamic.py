#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import json
import random
import sys
from pathlib import Path
from statistics import mean, pstdev

import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = Path(__file__).resolve().parent / "log_mapped_policy"
PROMPT_CACHE = ROOT / "38+RL_loop/outputs/llm_dynamic/llm_prompt_pools.json"
EXP43_SCRIPT = ROOT / "experiment/exp43_rl_action_policy_ablation_20260911/scripts/run_exp43_rl_action_policy_ablation_20260911.py"
EXP44_SCRIPT = ROOT / "experiment/exp44_reward_hyperparameter_validation_20260916/scripts/run_exp44_reward_search.py"
SEEDS = [42]
RANDOM_SEEDS = [42, 43, 44, 45, 46]
EPOCHS = 20
SOURCE_TO_ACTION = {
    "gdino_base": 1, "person": 2, "people": 3, "group of people": 4,
    "tricycle": 5, "covered tricycle": 6, "awning tricycle": 7,
    "motorcycle": 8, "motorbike": 9, "scooter": 10, "bicycle": 11,
}
ROLES = ["primary", "alias", "attribute", "relation"]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class DynamicScorer(nn.Module):
    def __init__(self):
        super().__init__()
        self.state_encoder = nn.Sequential(nn.Linear(38, 128), nn.LayerNorm(128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU())
        self.action_encoder = nn.Sequential(nn.Linear(20, 64), nn.LayerNorm(64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU())
        self.scorer = nn.Sequential(nn.Linear(192, 96), nn.ReLU(), nn.Linear(96, 1))

    def forward(self, states, actions):
        s = self.state_encoder(states)
        a = self.action_encoder(actions)
        return self.scorer(torch.cat([s, a, s * a], dim=1)).squeeze(1)


def normalize(value: str) -> set[str]:
    return set("".join(ch.lower() if ch.isalnum() else " " for ch in value).split())


def pool_for_stem(cache: dict, stem: str) -> tuple[str, dict[int, dict]]:
    item = cache["items"][stem]
    query = str(item.get("query", ""))
    result: dict[int, dict] = {}
    for row in item.get("actions", []):
        source = str(row.get("source_prompt", ""))
        action_id = SOURCE_TO_ACTION.get(source)
        if action_id is not None and action_id not in result:
            result[action_id] = row
    return query, result


def action_feature(action_id: int, row: dict | None, query: str, round_idx: int, used: set[int]) -> list[float]:
    if action_id == 0:
        return [1.0] + [0.0] * 19
    one_hot = [1.0 if action_id == idx else 0.0 for idx in range(1, 12)]
    text = str((row or {}).get("text", ""))
    text_tokens, query_tokens = normalize(text), normalize(query)
    overlap = len(text_tokens & query_tokens) / max(1, len(text_tokens | query_tokens))
    role = str((row or {}).get("llm_role", "alias"))
    return [
        0.0, *one_hot,
        min(float((row or {}).get("llm_rank", 0)) / 8.0, 1.0),
        round_idx / 2.0,
        1.0 if action_id in used else 0.0,
        overlap,
        *[1.0 if role == value else 0.0 for value in ROLES],
    ]


def build_groups(exp44, prompt_cache):
    cache = exp44.load_cache()
    rewards = exp44.reward_table(cache, exp44.CURRENT)
    flat_states, flat_actions, flat_rewards, groups = [], [], [], []
    for idx, stem in enumerate(cache["stems"]):
        query, rows = pool_for_stem(prompt_cache, stem)
        allowed = [0, *rows.keys()]
        start = len(flat_rewards)
        for first in allowed:
            second_allowed = [a for a in allowed if a == 0 or a != first]
            best = max(float(rewards[idx, first, second]) for second in second_allowed)
            flat_states.append(cache["states"][idx, 0].tolist())
            flat_actions.append(action_feature(first, rows.get(first), query, 0, set()))
            flat_rewards.append(best)
        groups.append(list(range(start, len(flat_rewards))))
        for first in allowed:
            start = len(flat_rewards)
            used = {first} if first else set()
            for second in allowed:
                if second != 0 and second == first:
                    continue
                flat_states.append(cache["states"][idx, 1 + first].tolist())
                flat_actions.append(action_feature(second, rows.get(second), query, 1, used))
                flat_rewards.append(float(rewards[idx, first, second]))
            groups.append(list(range(start, len(flat_rewards))))
    return (
        torch.tensor(flat_states, dtype=torch.float32),
        torch.tensor(flat_actions, dtype=torch.float32),
        torch.tensor(flat_rewards, dtype=torch.float32),
        groups,
    )


def train_one(states, actions, rewards, groups, seed: int):
    random.seed(seed); torch.manual_seed(seed)
    model = DynamicScorer()
    optimizer = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=1e-4)
    best_score, best_state, best_epoch = -1e9, None, 0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        all_logits = model(states, actions)
        losses = []
        for idx in groups:
            logits = all_logits[idx]
            group_rewards = rewards[idx]
            target = torch.softmax(group_rewards / 0.04, dim=0).detach()
            probs = torch.softmax(logits, dim=0)
            entropy = -(probs * torch.log_softmax(logits, dim=0)).sum()
            losses.append(-(target * torch.log_softmax(logits, dim=0)).sum() - 0.15 * (probs * group_rewards).sum() - 0.005 * entropy)
        loss = torch.stack(losses).mean()
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        model.eval()
        with torch.no_grad():
            chosen = []
            for idx in groups:
                logits = model(states[idx], actions[idx])
                chosen.append(float(rewards[idx[int(torch.argmax(logits))]]))
            score = mean(chosen)
        if score > best_score:
            best_score, best_epoch = score, epoch
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model, {"seed": seed, "best_epoch": best_epoch, "mean_training_reward": best_score}


def choose_test_plan(m, exp44, prompt_cache, models):
    plan = {}
    details = []
    for stem in m.read_split_stems("test"):
        query, rows = pool_for_stem(prompt_cache, stem)
        allowed = [0, *rows.keys()]
        width, height = m.image_size("test", stem)
        current, _ = m.exp38_rescore("test", stem, m.merge_action_pool("test", stem, 0))
        selected, used = [], set()
        for round_idx in range(2):
            candidates = [a for a in allowed if a == 0 or a not in used]
            state = exp44.state_from_boxes(current, width, height)
            state[-2] = round_idx / 2.0; state[-1] = (2 - round_idx) / 2.0
            state_t = torch.tensor([state] * len(candidates), dtype=torch.float32)
            action_t = torch.tensor([action_feature(a, rows.get(a), query, round_idx, used) for a in candidates], dtype=torch.float32)
            with torch.no_grad():
                scores = torch.stack([model(state_t, action_t) for model in models]).mean(dim=0)
            action = candidates[int(torch.argmax(scores))]
            selected.append(action)
            if action:
                current = m.apply_action_to_pool("test", stem, current, action)
                used.add(action)
        plan[stem] = selected
        details.append({"stem": stem, "allowed": allowed, "selected": selected})
    return plan, details


def compact(result):
    return {
        "prediction_boxes": float(result["metrics"]["prediction_box_count"]),
        "acc_05": float(result["metrics"]["acc_05"]),
        "acc_075": float(result["metrics"]["acc_075"]),
        "map_05": float(result["metrics"]["map_05"]),
        "small_final_recall_05": float(result["small_object_final_recall_05"]["recall_05"]),
        "actions_per_image": float(result["average_actions_per_image"]),
    }


def aggregate(rows):
    out = {}
    for key in rows[0]:
        vals = [row[key] for row in rows]
        out[key], out[f"{key}_std"] = mean(vals), pstdev(vals)
    return out


def main():
    m = load_module("exp43_mapped_dynamic", EXP43_SCRIPT)
    exp44 = load_module("exp44_mapped_dynamic", EXP44_SCRIPT)
    prompt_cache = json.loads(PROMPT_CACHE.read_text(encoding="utf-8"))
    states, actions, rewards, groups = build_groups(exp44, prompt_cache)
    models, training = [], []
    for seed in SEEDS:
        model, info = train_one(states, actions, rewards, groups, seed)
        models.append(model); training.append(info)
        print(info, flush=True)

    plan, selection_trace = choose_test_plan(m, exp44, prompt_cache, models)
    budgets = {stem: sum(action != 0 for action in values) for stem, values in plan.items()}
    pools = {stem: [a for a in selection_trace[idx]["allowed"] if a != 0] for idx, stem in enumerate(m.read_split_stems("test"))}
    prompt_plan = {stem: pools[stem][:budgets[stem]] for stem in pools}
    total_actions = sum(budgets.values())
    m.LOG_DIR = OUT_DIR; m.PRED_ROOT = OUT_DIR / "predictions"
    rl_result, _ = m.evaluate_method("mapped_qwen_dynamic_rl", plan, 0, 300, save_predictions=True)
    prompt_result, _ = m.evaluate_method("prompt_rank_equal_budget", prompt_plan, 0, total_actions, save_predictions=True)
    random_rows = []
    for seed in RANDOM_SEEDS:
        rng = random.Random(seed)
        random_plan = {stem: rng.sample(pools[stem], budgets[stem]) for stem in pools}
        result, _ = m.evaluate_method(f"random_equal_budget_{seed}", random_plan, 0, total_actions, save_predictions=False)
        random_rows.append(compact(result))
    payload = {
        "protocol": {
            "training_split": "validation only",
            "test_images": 150,
            "model_seeds": SEEDS,
            "random_baseline_seeds": RANDOM_SEEDS,
            "comparison": "same per-image Qwen mapped action pool and learned-policy non-STOP budget",
            "test_used_for_selection": False,
        },
        "training": training,
        "results": {
            "mapped_qwen_dynamic_rl": compact(rl_result),
            "prompt_rank_equal_budget": compact(prompt_result),
            "random_equal_budget": aggregate(random_rows),
            "random_runs": random_rows,
        },
        "selection_trace": selection_trace,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload["results"], indent=2), flush=True)


if __name__ == "__main__":
    main()
