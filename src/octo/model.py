"""Transformer hidden states plus one shared fp32 pointer head."""
from dataclasses import dataclass
import math
import torch
from torch import nn
from .encoding import Batch

BASE_MODEL = "Qwen/Qwen3-1.7B-Base"
BASE_REVISION = "ea980cb0a6c2ae4b936e82123acc929f1cec04c1"
LORA_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


@dataclass
class ModelOutput:
    logits: torch.Tensor
    probabilities: torch.Tensor
    candidate_states: torch.Tensor
    decision_states: torch.Tensor


class OctoModel(nn.Module):
    def __init__(self, backbone, pointer_dimension=256, *, freeze_backbone=True):
        super().__init__()
        if type(pointer_dimension) is not int or pointer_dimension <= 0:
            raise ValueError("pointer dimension must be a positive integer")
        if backbone.config.model_type != "qwen3":
            raise ValueError("only the verified Qwen3 attention backbone is supported")
        if backbone.config._attn_implementation not in ("eager", "sdpa"):
            raise ValueError("Octo requires eager or SDPA attention with arbitrary masks")
        self.backbone = backbone
        if freeze_backbone:
            self.backbone.requires_grad_(False)
        self.pointer_dimension = pointer_dimension
        parameter = next(backbone.parameters())
        self.query = nn.Linear(backbone.config.hidden_size, pointer_dimension, bias=False,
                               device=parameter.device, dtype=torch.float32)
        self.key = nn.Linear(backbone.config.hidden_size, pointer_dimension, bias=False,
                             device=parameter.device, dtype=torch.float32)

    def forward(self, batch: Batch):
        if self.query.weight.dtype != torch.float32 or self.key.weight.dtype != torch.float32:
            raise ValueError("pointer projections must remain fp32; cast only the backbone")
        capacity = self.backbone.config.max_position_embeddings
        if batch.input_ids.shape[1] > capacity or batch.position_ids.max().item() >= capacity:
            raise ValueError("request exceeds backbone physical or logical context capacity")
        dtype = next(self.backbone.parameters()).dtype
        h = self.backbone(input_ids=batch.input_ids, position_ids=batch.position_ids,
                          attention_mask=batch.attention_mask.to(dtype=dtype), use_cache=False).last_hidden_state
        candidates = h[batch.rows[:, None], batch.candidate_indices].float()
        decision = h[batch.rows, batch.decision_indices].float()
        # Explicitly disable autocast for the baseline pointer math and loss.
        with torch.autocast(device_type=h.device.type, enabled=False):
            q, k = self.query(decision), self.key(candidates)
            logits = torch.einsum("qd,qkd->qk", q, k) / math.sqrt(self.pointer_dimension)
            logits = logits.masked_fill(~batch.valid_candidates, float("-inf"))
            probabilities = logits.softmax(-1)
        if not torch.isfinite(logits[batch.valid_candidates]).all():
            raise FloatingPointError("nonfinite candidate logits")
        return ModelOutput(logits, probabilities, candidates, decision)

    @classmethod
    def from_pretrained(cls, model_id=BASE_MODEL, revision=BASE_REVISION, *, device="cpu",
                        dtype=torch.float32, backend="sdpa", lora=True, local_files_only=False):
        from transformers import AutoModel
        from peft import LoraConfig, get_peft_model
        backbone = AutoModel.from_pretrained(model_id, revision=revision, dtype=dtype,
                    attn_implementation=backend, local_files_only=local_files_only).to(device)
        if lora:
            names = {name.rsplit(".", 1)[-1] for name, module in backbone.named_modules()
                     if isinstance(module, nn.Linear)}
            if not set(LORA_TARGETS) <= names:
                raise ValueError("backbone is missing expected LoRA projection targets")
            backbone = get_peft_model(backbone, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05,
                                      target_modules=list(LORA_TARGETS)))
        return cls(backbone, freeze_backbone=not lora)

    @torch.inference_mode()
    def predict(self, batch):
        was_training = self.training
        self.eval()
        try:
            return interpret(self(batch).probabilities, batch)
        finally:
            self.train(was_training)


def categorical_loss(output: ModelOutput, batch: Batch):
    """Mean labeled-question loss per record, then equal mean over labeled records."""
    if not batch.labeled.any():
        raise ValueError("loss requires at least one labeled question")
    with torch.autocast(device_type=output.logits.device.type, enabled=False):
        logp = output.logits.float().log_softmax(-1).masked_fill(~batch.valid_candidates, 0.)
        losses = -(batch.targets * logp).sum(-1)
        record_losses = [losses[(batch.record_indices == ri) & batch.labeled].mean()
                         for ri in range(batch.record_count)
                         if ((batch.record_indices == ri) & batch.labeled).any()]
        loss = torch.stack(record_losses).mean()
    if not torch.isfinite(loss):
        raise FloatingPointError("nonfinite training loss")
    return loss


def interpret(probabilities, batch):
    answers = [[] for _ in range(batch.record_count)]
    for row, q in enumerate(batch.questions):
        p = probabilities[row, :len(q.candidates)].detach().float().cpu()
        if not torch.isfinite(p).all() or (p < 0).any() or not torch.isclose(p.sum(), torch.tensor(1.), atol=1e-6):
            raise ValueError("invalid probability distribution")
        distribution = {c.id: float(pi) for c, pi in zip(q.candidates, p)}
        result = {"id": q.id, "type": q.type, "probabilities": distribution}
        if q.type == "noul":
            result["probability_true"] = distribution["true"]
        else:
            entropy = -torch.special.xlogy(p, p).sum().item()
            result["confidence"] = min(1., max(0., 1-entropy/math.log(len(p))))
            if q.type == "choice":
                maximum = max(distribution.values())
                result["selected_id"] = min(i for i, pi in distribution.items() if pi == maximum)
            else:
                result["score"] = sum(c.value * distribution[c.id] for c in q.candidates)
        answers[int(batch.record_indices[row])].append(result)
    return answers


def train_step(model, batch, optimizer, max_grad_norm=1.):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    loss = categorical_loss(model(batch), batch)
    loss.backward()
    parameters = [p for p in model.parameters() if p.requires_grad]
    if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in parameters):
        optimizer.zero_grad(set_to_none=True)
        raise FloatingPointError("nonfinite training gradients; optimizer was not updated")
    torch.nn.utils.clip_grad_norm_(parameters, max_grad_norm, error_if_nonfinite=True)
    optimizer.step()
    return float(loss.detach())


def make_optimizer(model, adapter_lr=1e-4, pointer_lr=1e-3):
    return torch.optim.AdamW([
        {"params": [p for p in model.backbone.parameters() if p.requires_grad], "lr": adapter_lr},
        {"params": list(model.query.parameters()) + list(model.key.parameters()), "lr": pointer_lr},
    ])
