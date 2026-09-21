#!/usr/bin/env python3
from __future__ import annotations

import argparse
import functools
import csv
import json
import math
import random
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
import torch
import torch.nn as nn

from common import (
    CKPT_PATH,
    Det,
    OUT_ROOT,
    ROOT,
    SEED,
    deduplicate,
    evaluate_split,
    image_size,
    match_stats,
    read_gt,
    read_split_stems,
    require_inputs,
    reward_from_stats,
    state_from_boxes,
    write_predictions,
)


LLM_OUT = OUT_ROOT / "llm_dynamic"
PROMPT_CACHE = LLM_OUT / "llm_prompt_pools.json"
DYNAMIC_CKPT = LLM_OUT / "dynamic_prompt_policy.pt"
SUMMARY_JSON = LLM_OUT / "summary.json"
SUMMARY_MD = LLM_OUT / "summary.md"
TRACE_JSON = LLM_OUT / "closed_loop_trace.json"
PRED_DIR = LLM_OUT / "predictions"

GROUNDINGDINO_ROOT = Path("/home/wangzhe/GroundingDINO")
GDINO_CONFIG = GROUNDINGDINO_ROOT / "groundingdino/config/GroundingDINO_SwinT_OGC.py"
GDINO_CHECKPOINT = GROUNDINGDINO_ROOT / "weights/groundingdino_swint_ogc.pth"

QWEN25_DIR = Path("/home/wangzhe/VLM/model_weights/Qwen2.5-VL-3B/models--Qwen--Qwen2.5-VL-3B-Instruct/snapshots/66285546d2b821cf421d4f5eb2576359d3770cd3")
AERIAL_ANN = Path("/home/wangzhe/DroneP_VG/dataset/AerialVG/annotation")

CLASS_NAMES = {
    1: "pedestrian",
    2: "people",
    3: "bicycle",
    4: "car",
    5: "van",
    6: "truck",
    7: "tricycle",
    8: "awning tricycle",
    9: "bus",
    10: "motor",
}
CLASS_NAME_TO_ID = {name: idx for idx, name in CLASS_NAMES.items()}
SMALL_CLASSES = {1, 2, 3, 7, 8, 10}
BASE_PROMPT_COUNT = 2
GDINO_BOX_THRESHOLD = 0.03
GDINO_TEXT_THRESHOLD = 0.15


@dataclass(frozen=True)
class PromptAction:
    text: str
    class_name: str
    class_id: int
    llm_rank: int
    llm_role: str


class DynamicPolicyNet(nn.Module):
    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
        )
        self.action_encoder = nn.Sequential(
            nn.Linear(action_dim, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
        )
        self.scorer = nn.Sequential(
            nn.Linear(64 * 3, 96),
            nn.ReLU(),
            nn.Linear(96, 1),
        )

    def forward(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        s = self.state_encoder(states)
        a = self.action_encoder(actions)
        return self.scorer(torch.cat([s, a, s * a], dim=1)).squeeze(1)


def normalize(text: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9 ]+", " ", text.lower().replace("-", " ")).split())


def read_aerial_captions() -> dict[str, str]:
    captions: dict[str, str] = {}
    for split in ["train", "val", "test"]:
        path = AERIAL_ANN / f"vg_{split}_odvg.jsonl"
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            filename = row["filename"]
            stem = "_".join(Path(filename).stem.split("_")[1:])
            captions.setdefault(stem, row["grounding"]["caption"])
    return captions


def fallback_query(stem: str, captions: dict[str, str]) -> str:
    return captions.get(stem, "detect pedestrians, people, bicycles, cars, vans, trucks, tricycles, buses, and motorcycles in the drone traffic image")


def extract_json_array(text: str) -> list[str]:
    match = re.search(r"\[[\s\S]*\]", text)
    if not match:
        return []
    try:
        values = json.loads(match.group(0))
    except json.JSONDecodeError:
        return []
    return [str(v).strip() for v in values if str(v).strip()]


def load_qwen(device: str = "cuda:0"):
    from transformers import AutoModelForImageTextToText, AutoProcessor

    processor = AutoProcessor.from_pretrained(str(QWEN25_DIR), local_files_only=True, trust_remote_code=True, use_fast=False)
    model = AutoModelForImageTextToText.from_pretrained(
        str(QWEN25_DIR),
        local_files_only=True,
        trust_remote_code=True,
        dtype=torch.bfloat16,
        attn_implementation="eager",
    ).to(device)
    model.eval()
    return processor, model


def qwen_prompt_pool(processor, model, query: str, device: str = "cuda:0") -> list[dict[str, str]]:
    instruction = (
        "You are a visual grounding prompt proposer. "
        "Given a drone-image query, return JSON array only with 8 objects. "
        "Each object must have keys 'class' and 'prompt'. "
        "The class must be one of pedestrian, people, bicycle, car, van, truck, tricycle, awning tricycle, bus, motor. "
        "The prompt should be a short English detection phrase. Avoid explanations. Query: "
        + query
    )
    messages = [{"role": "user", "content": [{"type": "text", "text": instruction}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=120, do_sample=False)
    decoded = processor.batch_decode(out[:, inputs.input_ids.shape[1] :], skip_special_tokens=True)[0]
    prompts = extract_json_objects(decoded)
    return prompts[:8]


def extract_json_objects(text: str) -> list[dict[str, str]]:
    match = re.search(r"\[[\s\S]*\]", text)
    if not match:
        return []
    try:
        values = json.loads(match.group(0))
    except json.JSONDecodeError:
        return []
    objects: list[dict[str, str]] = []
    for value in values:
        if isinstance(value, dict):
            class_name = str(value.get("class", "")).strip()
            prompt = str(value.get("prompt", "")).strip()
            if class_name and prompt:
                objects.append({"class": class_name, "prompt": prompt})
        elif isinstance(value, str) and value.strip():
            objects.append({"class": "", "prompt": value.strip()})
    return objects


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False.")
    return torch.device(name)


def ensure_groundingdino_importable() -> None:
    if str(GROUNDINGDINO_ROOT) not in sys.path:
        sys.path.insert(0, str(GROUNDINGDINO_ROOT))


def load_groundingdino_model(device: str):
    ensure_groundingdino_importable()
    from groundingdino.util.inference import Model

    return Model(
        model_config_path=str(GDINO_CONFIG),
        model_checkpoint_path=str(GDINO_CHECKPOINT),
        device=resolve_device(device),
    )


def gdino_predict(model, image_bgr: np.ndarray, prompts: Sequence[str], box_thr: float, text_thr: float):
    detections = model.predict_with_classes(
        image=image_bgr,
        classes=list(prompts),
        box_threshold=box_thr,
        text_threshold=text_thr,
    )
    return np.asarray(detections.xyxy), np.asarray(detections.confidence), np.asarray(detections.class_id, dtype=object)


@functools.lru_cache(maxsize=None)
def read_image_bgr(split: str, stem: str) -> np.ndarray:
    path = ROOT / f"dataset/VisDroneSplit1000Guarded/VisDrone2019-DET-{split}/images/{stem}.jpg"
    image_bgr = cv2.imread(str(path))
    if image_bgr is None:
        raise RuntimeError(f"Cannot read image: {path}")
    return image_bgr


def role_for_prompt(prompt: str, rank: int) -> str:
    text = normalize(prompt)
    if rank == 0:
        return "primary"
    if any(word in text for word in ["near", "left", "right", "above", "below", "front", "behind"]):
        return "relation"
    if any(word in text for word in ["white", "black", "red", "blue", "gray", "grey", "silver", "yellow"]):
        return "attribute"
    return "alias"


def infer_class_name(prompt: str) -> str:
    text = normalize(prompt)
    for class_name in CLASS_NAMES.values():
        if normalize(class_name) in text:
            return class_name
    if "motorbike" in text or "motorcycle" in text or "scooter" in text:
        return "motor"
    if "pedestrian" in text or "person" in text or "people" in text or "group of people" in text:
        return "pedestrian"
    return "bicycle"


def build_prompt_actions(raw_prompts: Sequence[dict[str, str] | str]) -> list[PromptAction]:
    actions: list[PromptAction] = []
    seen: set[tuple[str, int]] = set()
    for idx, item in enumerate(raw_prompts):
        if isinstance(item, str):
            prompt = item.strip()
            class_name = infer_class_name(prompt)
        else:
            prompt = str(item.get("prompt", "")).strip()
            class_name = str(item.get("class", "")).strip() or infer_class_name(prompt)
        class_id = CLASS_NAME_TO_ID.get(class_name, 0)
        if not prompt or class_id not in CLASS_NAMES:
            continue
        key = (normalize(prompt), class_id)
        if key in seen:
            continue
        seen.add(key)
        actions.append(PromptAction(prompt, class_name, class_id, idx, role_for_prompt(prompt, idx)))
    if not actions:
        actions.append(PromptAction("traffic object", "bicycle", 3, 0, "fallback"))
    return actions


def generate_prompt_cache(limit: int = 0, device: str = "cuda:0") -> dict[str, object]:
    require_inputs()
    if not torch.cuda.is_available():
        raise RuntimeError("Qwen prompt generation requires CUDA in this experiment.")
    captions = read_aerial_captions()
    stems = sorted(set(read_split_stems("val") + read_split_stems("test")))
    if limit:
        stems = stems[:limit]
    processor, model = load_qwen(device=device)
    cache: dict[str, object] = {"model": "Qwen2.5-VL-3B-Instruct", "items": {}}
    for idx, stem in enumerate(stems, start=1):
        query = fallback_query(stem, captions)
        try:
            prompts = qwen_prompt_pool(processor, model, query, device=device)
        except Exception as exc:
            prompts = []
            print(f"[llm generate] fallback stem={stem} err={exc}", flush=True)
        if not prompts:
            prompts = [
                {"class": "pedestrian", "prompt": "a pedestrian standing on the road"},
                {"class": "people", "prompt": "a group of people on the sidewalk"},
                {"class": "bicycle", "prompt": "a bicycle on the road"},
                {"class": "tricycle", "prompt": "a tricycle near the intersection"},
                {"class": "awning tricycle", "prompt": "an awning tricycle on the street"},
                {"class": "motor", "prompt": "a motorcycle on the road"},
                {"class": "car", "prompt": "a car on the road"},
                {"class": "truck", "prompt": "a truck on the road"},
            ]
        cache["items"][stem] = {
            "query": query,
            "prompts": prompts,
            "actions": [action.__dict__ for action in build_prompt_actions(prompts)],
        }
        if idx == 1 or idx % 25 == 0 or idx == len(stems):
            print(f"[llm generate] {idx}/{len(stems)} stem={stem} prompts={[p['prompt'] for p in prompts[:3]]}", flush=True)
    PROMPT_CACHE.parent.mkdir(parents=True, exist_ok=True)
    PROMPT_CACHE.write_text(json.dumps(cache, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return cache


def load_prompt_cache(device: str = "cuda:0") -> dict[str, object]:
    if not PROMPT_CACHE.exists():
        return generate_prompt_cache(device=device)
    return json.loads(PROMPT_CACHE.read_text(encoding="utf-8"))


def action_features(action: PromptAction | None, query: str, used_prompts: set[str], round_idx: int) -> list[float]:
    if action is None:
        return [1.0] + [0.0] * 10 + [0.0, 0.0, 0.0, round_idx / 2.0, 0.0] + [0.0, 0.0, 0.0, 0.0]
    class_one_hot = [1.0 if action.class_id == idx else 0.0 for idx in range(1, 11)]
    prompt_tokens = set(normalize(action.text).split())
    query_tokens = set(normalize(query).split())
    overlap = len(prompt_tokens & query_tokens) / max(1, len(prompt_tokens | query_tokens))
    role_one_hot = [
        1.0 if action.llm_role == "primary" else 0.0,
        1.0 if action.llm_role == "alias" else 0.0,
        1.0 if action.llm_role == "attribute" else 0.0,
        1.0 if action.llm_role == "relation" else 0.0,
    ]
    return [
        0.0,
        *class_one_hot,
        min(len(prompt_tokens) / 8.0, 1.0),
        min(action.llm_rank / 8.0, 1.0),
        overlap,
        round_idx / 2.0,
        1.0 if normalize(action.text) in used_prompts else 0.0,
        *role_one_hot,
    ]


@functools.lru_cache(maxsize=None)
def load_groundingdino_model(device: str):
    ensure_groundingdino_importable()
    from groundingdino.util.inference import Model

    return Model(
        model_config_path=str(GDINO_CONFIG),
        model_checkpoint_path=str(GDINO_CHECKPOINT),
        device=resolve_device(device),
    )


def prompt_action_boxes(split: str, stem: str, action: PromptAction, device: str = "cuda:0") -> list[Det]:
    model = load_groundingdino_model(device)
    image_bgr = read_image_bgr(split, stem)
    boxes, scores, _ = gdino_predict(model, image_bgr, [action.text], GDINO_BOX_THRESHOLD, GDINO_TEXT_THRESHOLD)
    dets: list[Det] = []
    for box, score in zip(boxes, scores):
        x1, y1, x2, y2 = [float(v) for v in box[:4]]
        if x2 > x1 and y2 > y1:
            dets.append(Det(action.class_id, x1, y1, x2, y2, float(score)))
    return dets


@functools.lru_cache(maxsize=None)
def cached_prompt_action_boxes(
    split: str,
    stem: str,
    prompt: str,
    class_id: int,
    device: str,
) -> tuple[tuple[float, float, float, float, float], ...]:
    action = PromptAction(prompt, CLASS_NAMES[class_id], class_id, 0, "alias")
    return tuple((det.x1, det.y1, det.x2, det.y2, det.score) for det in prompt_action_boxes(split, stem, action))


def action_to_dets(split: str, stem: str, action: PromptAction, device: str = "cuda:0") -> list[Det]:
    return [
        Det(action.class_id, x1, y1, x2, y2, score)
        for x1, y1, x2, y2, score in cached_prompt_action_boxes(split, stem, action.text, action.class_id, device)
    ]


def apply_prompt_action(split: str, stem: str, current: Sequence[Det], action: PromptAction | None, device: str = "cuda:0") -> list[Det]:
    if action is None:
        return deduplicate(current)
    return deduplicate(list(current) + action_to_dets(split, stem, action, device=device))


def candidate_actions(cache: dict[str, object], stem: str) -> tuple[str, list[PromptAction]]:
    item = cache["items"].get(stem)
    if item is None:
        prompts = [
            {"class": "pedestrian", "prompt": "person"},
            {"class": "people", "prompt": "people"},
            {"class": "motor", "prompt": "motorbike"},
            {"class": "bicycle", "prompt": "bicycle"},
            {"class": "tricycle", "prompt": "tricycle"},
            {"class": "awning tricycle", "prompt": "covered tricycle"},
            {"class": "car", "prompt": "traffic vehicle"},
        ]
        return "", build_prompt_actions(prompts)
    actions: list[PromptAction] = []
    for row in item.get("actions", []):
        if isinstance(row, PromptAction):
            actions.append(row)
            continue
        if isinstance(row, dict):
            prompt = str(row.get("text") or row.get("prompt") or row.get("source_prompt") or "").strip()
            class_name = str(row.get("class_name") or row.get("class") or "").strip() or infer_class_name(prompt)
            class_id = int(row.get("class_id") or CLASS_NAME_TO_ID.get(class_name, 0))
            llm_rank = int(row.get("llm_rank", 0))
            llm_role = str(row.get("llm_role") or row.get("role") or role_for_prompt(prompt, llm_rank))
            if prompt and class_id in CLASS_NAMES:
                actions.append(PromptAction(prompt, class_name, class_id, llm_rank, llm_role))
    if not actions:
        actions = build_prompt_actions(item.get("prompts", []))
    return str(item.get("query", "")), actions


def build_dynamic_dataset(
    split: str,
    cache: dict[str, object],
    device: str = "cuda:0",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[list[int]], dict[str, object]]:
    states: list[list[float]] = []
    actions: list[list[float]] = []
    rewards: list[float] = []
    groups: list[list[int]] = []
    oracle = {}
    for stem in read_split_stems(split):
        width, height = image_size(split, stem)
        query, prompt_actions = candidate_actions(cache, stem)
        base, _ = exp38_rescore(split, stem, merge_action_pool(split, stem, 0))
        gt = read_gt(split, stem)
        base_stats = match_stats(base, gt)

        start = len(rewards)
        first_scores = []
        all_first = [None] + prompt_actions
        for first_idx, first_action in enumerate(all_first):
            cur1 = apply_prompt_action(split, stem, base, first_action, device=device)
            used = {normalize(first_action.text)} if first_action is not None else set()
            best_second = -1e9
            for second_action in [None] + prompt_actions:
                if second_action is not None and normalize(second_action.text) in used:
                    continue
                final = apply_prompt_action(split, stem, cur1, second_action, device=device)
                stats = match_stats(final, gt)
                count = int(first_action is not None) + int(second_action is not None)
                score = reward_from_stats(stats, base_stats, count, used_stop=(first_action is None or second_action is None))
                best_second = max(best_second, score)
            first_scores.append(best_second)
            st = state_from_boxes(base, width, height)
            st[-2] = 0.0
            st[-1] = 1.0
            states.append(st)
            actions.append(action_features(first_action, query, set(), 0))
            rewards.append(best_second)
        groups.append(list(range(start, len(rewards))))
        best_first = all_first[max(range(len(first_scores)), key=lambda i: first_scores[i])]
        oracle[("round2", "STOP" if best_first is None else best_first.text)] = oracle.get(("round2", "STOP" if best_first is None else best_first.text), 0) + 1

        for first_action in all_first:
            cur1 = apply_prompt_action(split, stem, base, first_action, device=device)
            used = {normalize(first_action.text)} if first_action is not None else set()
            st2 = state_from_boxes(cur1, width, height)
            st2[-2] = 0.5
            st2[-1] = 0.5
            start = len(rewards)
            second_scores = []
            for second_action in [None] + prompt_actions:
                if second_action is not None and normalize(second_action.text) in used:
                    continue
                final = apply_prompt_action(split, stem, cur1, second_action, device=device)
                stats = match_stats(final, gt)
                count = int(first_action is not None) + int(second_action is not None)
                score = reward_from_stats(stats, base_stats, count, used_stop=(second_action is None))
                states.append(st2)
                actions.append(action_features(second_action, query, used, 1))
                rewards.append(score)
                second_scores.append((score, second_action))
            groups.append(list(range(start, len(rewards))))
            best_second = max(second_scores, key=lambda item: item[0])[1]
            oracle[("round3", "STOP" if best_second is None else best_second.text)] = oracle.get(("round3", "STOP" if best_second is None else best_second.text), 0) + 1
    meta = {
        "decision_count": len(groups),
        "pair_count": len(rewards),
        "state_dim": len(states[0]),
        "action_feature_dim": len(actions[0]),
        "oracle_counts": {f"{k[0]}:{k[1]}": v for k, v in sorted(oracle.items())},
    }
    return torch.tensor(states, dtype=torch.float32), torch.tensor(actions, dtype=torch.float32), torch.tensor(rewards, dtype=torch.float32), groups, meta


def train_dynamic(device: torch.device, epochs: int, seed: int) -> dict[str, object]:
    random.seed(seed)
    torch.manual_seed(seed)
    cache = load_prompt_cache(device=str(device))
    states, actions, rewards, groups, meta = build_dynamic_dataset("val", cache, device=str(device))
    states = states.to(device)
    actions = actions.to(device)
    rewards = rewards.to(device)
    model = DynamicPolicyNet(states.shape[1], actions.shape[1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=1e-4)
    best = {"epoch": 0.0, "val_mean_reward": -1e9, "loss": 0.0}
    for epoch in range(1, epochs + 1):
        total = 0.0
        random.shuffle(groups)
        model.train()
        for group in groups:
            idx = torch.tensor(group, dtype=torch.long, device=device)
            logits = model(states[idx], actions[idx])
            group_rewards = rewards[idx]
            target = torch.softmax(group_rewards / 0.04, dim=0).detach()
            logp = torch.log_softmax(logits, dim=0)
            expected_reward = (torch.softmax(logits, dim=0) * group_rewards).sum()
            entropy = -(torch.softmax(logits, dim=0) * logp).sum()
            loss = -(target * logp).sum() - 0.15 * expected_reward - 0.005 * entropy
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
            total += float(loss.item())
        model.eval()
        with torch.no_grad():
            vals = []
            for group in groups:
                idx = torch.tensor(group, dtype=torch.long, device=device)
                action_idx = int(torch.argmax(model(states[idx], actions[idx])).item())
                vals.append(float(rewards[idx[action_idx]].item()))
            objective = sum(vals) / max(1, len(vals))
        if objective > best["val_mean_reward"]:
            best = {"epoch": float(epoch), "val_mean_reward": objective, "loss": total / max(1, len(groups))}
            DYNAMIC_CKPT.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                    "state_dim": states.shape[1],
                    "action_feature_dim": actions.shape[1],
                    "best": best,
                    "llm_model": "Qwen2.5-VL-3B-Instruct",
                },
                DYNAMIC_CKPT,
            )
        if epoch == 1 or epoch % 20 == 0 or epoch == epochs:
            print(f"[llm dynamic train] epoch={epoch:03d}/{epochs} loss={total/max(1,len(groups)):.6f} greedy_reward={objective:.6f}", flush=True)
    summary = {"device": str(device), "best": best, **meta}
    (LLM_OUT / "train_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def load_dynamic_policy(device: torch.device) -> DynamicPolicyNet:
    ckpt = torch.load(DYNAMIC_CKPT, map_location="cpu")
    model = DynamicPolicyNet(int(ckpt["state_dim"]), int(ckpt["action_feature_dim"])).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model


def test_dynamic(device: torch.device) -> dict[str, object]:
    cache = load_prompt_cache(device=str(device))
    model = load_dynamic_policy(device)
    predictions = {}
    trace = []
    counts: dict[str, int] = {}
    for stem in read_split_stems("test"):
        width, height = image_size("test", stem)
        query, prompt_actions = candidate_actions(cache, stem)
        current, _ = exp38_rescore("test", stem, merge_action_pool("test", stem, 0))
        used: set[str] = set()
        chosen = []
        for round_idx in range(2):
            pool = [None] + [a for a in prompt_actions if normalize(a.text) not in used]
            st = state_from_boxes(current, width, height)
            st[-2] = round_idx / 2.0
            st[-1] = (2 - round_idx) / 2.0
            state_batch = torch.tensor([st] * len(pool), dtype=torch.float32, device=device)
            action_batch = torch.tensor([action_features(a, query, used, round_idx) for a in pool], dtype=torch.float32, device=device)
            with torch.no_grad():
                logits = model(state_batch, action_batch)
                choice = int(torch.argmax(logits).item())
            action = pool[choice]
            name = "STOP" if action is None else action.text
            counts[name] = counts.get(name, 0) + 1
            chosen.append(name)
            if action is not None:
                current = apply_prompt_action("test", stem, current, action, device=str(device))
                used.add(normalize(action.text))
        final, _ = exp38_rescore("test", stem, current)
        predictions[stem] = final
        trace.append(
            {
                "stem": stem,
                "query": query,
                "llm_prompts": [a.text for a in prompt_actions],
                "prompt_classes": [a.class_name for a in prompt_actions],
                "actions": chosen,
                "final_count": len(final),
            }
        )
    write_predictions(PRED_DIR, predictions)
    metrics = evaluate_split("test", PRED_DIR)
    base_predictions = {}
    for stem in read_split_stems("test"):
        base_predictions[stem], _ = exp38_rescore("test", stem, merge_action_pool("test", stem, 0))
    base_dir = LLM_OUT / "predictions_query_base"
    write_predictions(base_dir, base_predictions)
    base_metrics = evaluate_split("test", base_dir)
    TRACE_JSON.write_text(json.dumps(trace, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    payload = {
        "method": "Qwen2.5-VL prompt-direct GroundingDINO + three-round RL + Exp38 rescoring",
        "device": str(device),
        "llm_model": "Qwen2.5-VL-3B-Instruct",
        "prompt_cache": str(PROMPT_CACHE),
        "checkpoint": str(DYNAMIC_CKPT),
        "action_counts": counts,
        "query_base_metrics": base_metrics,
        "test_metrics": metrics,
        "trace_path": str(TRACE_JSON),
        "prediction_dir": str(PRED_DIR),
    }
    SUMMARY_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# LLM Dynamic Prompt RL Summary",
        "",
        "- LLM: `Qwen2.5-VL-3B-Instruct`",
        "- policy: `state encoder + prompt action encoder + scorer`",
        "- loop_rounds: `3`",
        f"- action_counts: `{counts}`",
        "",
        "| Method | Pred boxes | Acc@0.5 | Acc@0.75 | mAP@0.5 |",
        "| --- | ---: | ---: | ---: | ---: |",
        f"| query_base_exp38_scored | {int(base_metrics['prediction_box_count'])} | {base_metrics['acc_05']:.4f} | {base_metrics['acc_075']:.4f} | {base_metrics['map_05']:.4f} |",
        f"| llm_dynamic_38+RL_loop | {int(metrics['prediction_box_count'])} | {metrics['acc_05']:.4f} | {metrics['acc_075']:.4f} | {metrics['map_05']:.4f} |",
    ]
    SUMMARY_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LLM dynamic prompt action space experiment.")
    parser.add_argument("--mode", choices=["generate", "train", "test", "all"], default="all")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--limit-generate", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    require_inputs()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable.")
    device = torch.device(args.device)
    print(f"[llm dynamic] torch={torch.__version__} cuda={torch.cuda.is_available()} device={device}", flush=True)
    if args.mode in {"generate", "all"}:
        generate_prompt_cache(limit=args.limit_generate, device=args.device)
    if args.mode in {"train", "all"}:
        info = train_dynamic(device, args.epochs, args.seed)
        print(json.dumps(info, ensure_ascii=False, indent=2), flush=True)
    if args.mode in {"test", "all"}:
        payload = test_dynamic(device)
        print(json.dumps(payload["test_metrics"], ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
