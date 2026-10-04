"""Prediction artifacts: adapters/backbone, pointer, tokenizer, and request contract."""
from dataclasses import asdict
import json
from pathlib import Path
import torch
from .data import Limits, NORMALIZATION_VERSION, NOUL_TEMPLATE
from .encoding import Encoder, StructuralTokens
from .model import OctoModel


def save_checkpoint(path, model, encoder, *, metadata=None):
    from peft import PeftModel
    if not hasattr(encoder.tokenizer, "save_pretrained"):
        raise ValueError("checkpoint requires a serializable Hugging Face tokenizer")
    encoder.tokens.verify(encoder.tokenizer)
    if isinstance(model.backbone, PeftModel) and not getattr(model.backbone.config, "_commit_hash", None):
        raise ValueError("adapter checkpoint must record an exact base revision")
    path = Path(path)
    path.mkdir(parents=True, exist_ok=False)
    model.backbone.save_pretrained(path / "backbone")
    if not hasattr(encoder.tokenizer, "save_pretrained"):
        raise ValueError("checkpoint requires a serializable Hugging Face tokenizer")
    encoder.tokenizer.save_pretrained(path / "tokenizer")
    torch.save({"query": model.query.state_dict(), "key": model.key.state_dict()}, path / "pointer.pt")
    config = {"format_version": 1, "normalization_version": NORMALIZATION_VERSION,
              "noul_template": NOUL_TEMPLATE, "tokens": asdict(encoder.tokens),
              "limits": asdict(encoder.limits), "pad_token_id": encoder.pad_token_id,
              "pointer_dimension": model.pointer_dimension,
              "adapter": isinstance(model.backbone, PeftModel),
              "backbone_dtype": str(next(model.backbone.parameters()).dtype).removeprefix("torch."),
              "backend": model.backbone.config._attn_implementation,
              "freeze_backbone": not any(p.requires_grad for p in model.backbone.parameters()),
              "base_model": model.backbone.config._name_or_path,
              "base_revision": getattr(model.backbone.config, "_commit_hash", None),
              "metadata": metadata or {}}
    (path / "octo.json").write_text(json.dumps(config, indent=2)+"\n")


def load_checkpoint(path, *, device="cpu", local_files_only=True):
    from transformers import AutoModel, AutoTokenizer
    from peft import PeftModel
    path = Path(path)
    config = json.loads((path / "octo.json").read_text())
    if config["format_version"] != 1 or config["normalization_version"] != NORMALIZATION_VERSION or config["noul_template"] != NOUL_TEMPLATE:
        raise ValueError("unsupported checkpoint contract version")
    dtype = getattr(torch, config["backbone_dtype"])
    if config["adapter"]:
        if not config["base_revision"]:
            raise ValueError("adapter checkpoint must record an exact base revision")
        base = AutoModel.from_pretrained(config["base_model"], revision=config["base_revision"],
                    dtype=dtype, attn_implementation=config["backend"], local_files_only=local_files_only)
        # Offline Transformers loads may omit the commit field even when an
        # exact revision was requested. Preserve the saved, pinned identity so
        # resumed training can save another valid adapter checkpoint.
        base.config._commit_hash = config["base_revision"]
        base.config._name_or_path = config["base_model"]
        backbone = PeftModel.from_pretrained(base, path / "backbone", is_trainable=True)
    else:
        backbone = AutoModel.from_pretrained(path / "backbone", dtype=dtype,
                    attn_implementation=config["backend"], local_files_only=True)
    model = OctoModel(backbone.to(device), config["pointer_dimension"],
                      freeze_backbone=config["freeze_backbone"])
    pointer = torch.load(path / "pointer.pt", map_location=device, weights_only=True)
    model.query.load_state_dict(pointer["query"])
    model.key.load_state_dict(pointer["key"])
    tokenizer = AutoTokenizer.from_pretrained(path / "tokenizer", local_files_only=True)
    tokens = StructuralTokens(**config["tokens"])
    tokens.verify(tokenizer)
    encoder = Encoder(tokenizer, tokens, Limits(**config["limits"]), config["pad_token_id"])
    model.eval()
    return model, encoder, config["metadata"]
