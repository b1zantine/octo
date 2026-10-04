"""Reproducible architecture and MPS training probe; synthetic inputs, not quality evaluation."""
import argparse
import gc
import itertools
import json
import math
import platform
import time
from pathlib import Path
import torch
import transformers
import peft
from transformers import AutoModel, Qwen3Config, Qwen3Model
from peft import LoraConfig, get_peft_model

REVISION = 'ea980cb0a6c2ae4b936e82123acc929f1cec04c1'
TARGETS = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']


def fixture(device, dtype, order=(0, 1, 2), length=None, changed=False):
    state = list(range(10, 18))
    question = list(range(20, 25))
    branches = [[30, 31, 32], [40, 41, 42, 43], [50, 51]]
    if changed:
        branches[1][1] = 60
    if length:
        state = [10 + i % 8 for i in range(length - len(question) - sum(map(len, branches)) - 1)]
    ids = state + question
    roles = [0] * len(state) + [1] * len(question)
    candidates = [-1] * len(ids)
    positions = list(range(len(ids)))
    p = len(ids) - 1
    ends = []
    for identity in order:
        branch = branches[identity]
        ids += branch
        roles += [2] * len(branch)
        candidates += [identity] * len(branch)
        positions += list(range(p + 1, p + 1 + len(branch)))
        ends.append(len(ids) - 1)
    ids += [70]
    roles += [3]
    candidates += [-1]
    positions += [p + max(map(len, branches)) + 1]
    n = len(ids)
    role = torch.tensor(roles)
    cand = torch.tensor(candidates)
    i = torch.arange(n)[:, None]
    j = torch.arange(n)[None, :]
    allowed = (j <= i) & ((role[None, :] == 0) |
        ((role[:, None] >= 1) & (role[None, :] == 1)) |
        ((role[:, None] == 2) & (role[None, :] == 2) & (cand[:, None] == cand[None, :])) |
        (role[:, None] == 3))
    mask = torch.full((n, n), float('-inf'), dtype=dtype).masked_fill(allowed, 0)
    return dict(input_ids=torch.tensor([ids], device=device),
                position_ids=torch.tensor([positions], device=device),
                attention_mask=mask[None, None].to(device), use_cache=False), ends, allowed


class Probe(torch.nn.Module):
    def __init__(self, backbone, device):
        super().__init__()
        self.backbone = get_peft_model(backbone, LoraConfig(r=16, lora_alpha=32,
                                    lora_dropout=0.05, target_modules=TARGETS))
        d = backbone.config.hidden_size
        self.query = torch.nn.Linear(d, 256, bias=False, device=device)
        self.key = torch.nn.Linear(d, 256, bias=False, device=device)

    def forward(self, batch, ends):
        h = self.backbone(**batch).last_hidden_state
        states = h[:, ends].float()
        q = self.query(h[:, -1].float())
        k = self.key(states)
        logits = torch.einsum('bd,bkd->bk', q, k) / 16
        return logits, states


def sync(device):
    if device == 'mps':
        torch.mps.synchronize()


def memory(device):
    if device != 'mps':
        return {}
    return {'allocated_gib': torch.mps.current_allocated_memory() / 2**30,
            'driver_gib': torch.mps.driver_allocated_memory() / 2**30}


def invariance(probe, device, dtype):
    probe.eval()
    errors = []
    with torch.no_grad():
        batch, ends, _ = fixture(device, dtype)
        logits, baseline = probe(batch, ends)
        probabilities = logits.softmax(-1)
        for order in itertools.permutations(range(3)):
            b, e, _ = fixture(device, dtype, order)
            z, h = probe(b, e)
            inverse = [order.index(i) for i in range(3)]
            errors.append({'order': order,
                'hidden_max_error': (h[:, inverse] - baseline).abs().max().item(),
                'probability_max_error': (z.softmax(-1)[:, inverse] - probabilities).abs().max().item(),
                'hidden_relative_l2_error': ((h[:, inverse] - baseline).norm()/baseline.norm()).item()})
        b, e, _ = fixture(device, dtype, changed=True)
        _, changed_h = probe(b, e)
        perturb_error = (changed_h[:, [0, 2]] - baseline[:, [0, 2]]).abs().max().item()
    tolerance = 0.03 if dtype == torch.bfloat16 else 1e-4
    relative_tolerance = 1e-4
    passed = (max(x['hidden_relative_l2_error'] for x in errors) < relative_tolerance and
              max(x['probability_max_error'] for x in errors) < 0.005 and perturb_error < tolerance)
    return {'permutations': errors, 'other_candidate_perturbation_error': perturb_error,
            'perturbation_absolute_tolerance': tolerance, 'hidden_relative_l2_tolerance': relative_tolerance, 'probability_tolerance': 0.005, 'passes_initial_tolerances': passed}


def train_steps(probe, device, dtype, length):
    probe.train()
    optimizer = torch.optim.AdamW([p for p in probe.parameters() if p.requires_grad], lr=1e-5)
    batch, ends, _ = fixture(device, dtype, length=length)
    timings, losses, memories = [], [], []
    for step in range(4):
        optimizer.zero_grad(set_to_none=True)
        sync(device)
        start = time.perf_counter()
        logits, _ = probe(batch, ends)
        target = torch.tensor([[0.2, 0.5, 0.3]], device=device)
        loss = -(target * logits.log_softmax(-1)).sum()
        assert torch.isfinite(loss)
        loss.backward()
        adapter_grads = [p.grad for n,p in probe.named_parameters() if 'lora_' in n and p.grad is not None]
        assert adapter_grads and any(g.abs().max().item() > 0 for g in adapter_grads)
        assert all(torch.isfinite(g).all().item() for g in adapter_grads)
        assert probe.query.weight.grad.abs().max().item() > 0
        assert all(p.grad is None for p in probe.parameters() if not p.requires_grad)
        memories.append(memory(device))
        torch.nn.utils.clip_grad_norm_([p for p in probe.parameters() if p.requires_grad], 1.0)
        optimizer.step()
        sync(device)
        timings.append(time.perf_counter() - start)
        losses.append(loss.item())
        print(json.dumps({'length': length, 'step': step, 'loss': losses[-1], 'seconds': timings[-1], **memories[-1]}), flush=True)
    result = {'length': length, 'losses': losses, 'step_seconds': timings,
              'median_seconds_after_warmup': sorted(timings[1:])[1],
              'sampled_training_memory': memories, 'finite_adapter_and_pointer_gradients': True,
              'frozen_parameters_have_no_gradients': True}
    del optimizer, batch, logits, loss
    gc.collect()
    if device == 'mps': torch.mps.empty_cache()
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrained', action='store_true')
    parser.add_argument('--output', default='docs/octo_v1_training_probe.json')
    args = parser.parse_args()
    assert torch.backends.mps.is_available(), 'Run outside sandbox for MPS GPU access'
    torch.manual_seed(17)
    torch.set_num_threads(8)
    report = {'hardware': 'Apple M4 Max / 64 GB unified memory', 'macos': platform.mac_ver()[0],
        'torch': torch.__version__, 'transformers': transformers.__version__, 'peft': peft.__version__,
        'mps_recommended_max_gib': torch.mps.recommended_max_memory()/2**30,
        'scope': 'synthetic architecture compatibility and short training steps; not task-quality evaluation'}
    config = Qwen3Config(vocab_size=128, hidden_size=128, intermediate_size=256,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2, head_dim=32)
    config._attn_implementation = 'eager'
    tiny = Qwen3Model(config).to('mps')
    batch, ends, allowed = fixture('mps', torch.float32)
    with torch.no_grad():
        attention = tiny(**batch, output_attentions=True).attentions
        forbidden = (~allowed).to('mps')
        max_forbidden = max(a[0, :, forbidden].abs().max().item() for a in attention)
        assert max_forbidden == 0
    report['eager_forbidden_attention_max'] = max_forbidden
    probe = Probe(tiny, 'mps')
    report['tiny_fp32_invariance'] = invariance(probe, 'mps', torch.float32)
    report['tiny_training'] = train_steps(probe, 'mps', torch.float32, 512)
    del probe, tiny, attention, batch
    gc.collect()
    torch.mps.empty_cache()
    Path(args.output).write_text(json.dumps(report, indent=2)+'\n')
    if args.pretrained:
        print('Loading verified 1.7B base checkpoint', flush=True)
        backbone = AutoModel.from_pretrained('Qwen/Qwen3-1.7B-Base', revision=REVISION,
            dtype=torch.bfloat16, attn_implementation='sdpa').to('mps')
        probe = Probe(backbone, 'mps')
        result = {'model': 'Qwen/Qwen3-1.7B-Base', 'revision': REVISION,
            'pointer_dtype': 'float32', 'parameters': sum(p.numel() for p in probe.parameters()),
            'trainable_parameters': sum(p.numel() for p in probe.parameters() if p.requires_grad),
            'supervision': 'synthetic soft target [0.2, 0.5, 0.3]', 'learning_rate': 1e-5}
        for dtype in (torch.bfloat16, torch.float32):
            probe.backbone.to(dtype=dtype)
            for backend in ('sdpa', 'eager'):
                probe.backbone.set_attn_implementation(backend)
                key = ('bf16' if dtype == torch.bfloat16 else 'fp32') + '_' + backend + '_invariance'
                result[key] = invariance(probe, 'mps', dtype)
                print(key + ': ' + str(result[key]['passes_initial_tolerances']), flush=True)
        batch, _, allowed = fixture('mps', torch.float32)
        with torch.no_grad():
            weights = probe.backbone(**batch, output_attentions=True).attentions
            forbidden = (~allowed).to('mps')
            result['pretrained_eager_forbidden_attention_max'] = max(a[0, :, forbidden].abs().max().item() for a in weights)
            assert result['pretrained_eager_forbidden_attention_max'] == 0
        del weights, batch
        for dtype in (torch.bfloat16, torch.float32):
            probe.backbone.to(dtype=dtype)
            probe.backbone.set_attn_implementation('sdpa')
            key = 'bf16_training' if dtype == torch.bfloat16 else 'fp32_training'
            result[key] = [train_steps(probe, 'mps', dtype, n) for n in (512, 1024)]
        result['fp32_sdpa_post_training_invariance'] = invariance(probe, 'mps', torch.float32)
        assert result['fp32_sdpa_invariance']['passes_initial_tolerances']
        assert result['fp32_sdpa_post_training_invariance']['passes_initial_tolerances']
        result['config'] = {'hidden_size': backbone.config.hidden_size, 'layers': backbone.config.num_hidden_layers, 'attention_heads': backbone.config.num_attention_heads}
        report['pretrained'] = result
    Path(args.output).write_text(json.dumps(report, indent=2)+'\n')
    print('Probe complete; inspect tolerance results in: '+args.output, flush=True)

if __name__ == '__main__':
    main()
