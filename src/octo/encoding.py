"""Branch layout, repeated positions, and physically causal attention topology."""
from dataclasses import dataclass
from typing import Protocol
import torch
from .data import Limits, Question, Record, normalize_record

STATE, INSTRUCTION, CANDIDATE, DECISION, PADDING = range(5)


class Tokenizer(Protocol):
    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]: ...


@dataclass(frozen=True)
class StructuralTokens:
    state: int
    question: int
    option: int
    option_end: int
    decide: int

    def __post_init__(self):
        values = list(vars(self).values())
        if any(type(v) is not int or v < 0 for v in values) or len(set(values)) != 5:
            raise ValueError("five distinct nonnegative structural token IDs are required")

    def verify(self, tokenizer):
        reserved = set(tokenizer.all_special_ids)
        if any(v not in reserved or v >= len(tokenizer) for v in vars(self).values()):
            raise ValueError("structural IDs must be existing reserved tokenizer tokens")

    @classmethod
    def from_tokenizer(cls, tokenizer):
        # Avoid end/start/padding tokens; never silently add frozen random embeddings.
        excluded = {tokenizer.bos_token_id, tokenizer.eos_token_id, tokenizer.pad_token_id}
        ids = list(dict.fromkeys(i for i in tokenizer.all_special_ids if i not in excluded))
        if len(ids) < 5:
            raise ValueError("tokenizer has fewer than five usable reserved tokens; provide a verified map")
        result = cls(*ids[:5])
        result.verify(tokenizer)
        return result


@dataclass
class EncodedRecord:
    input_ids: list[int]
    position_ids: list[int]
    roles: list[int]
    segments: list[int]
    candidate_owners: list[int]
    questions: tuple[Question, ...]
    candidate_ends: list[list[int]]
    decisions: list[int]


class Encoder:
    def __init__(self, tokenizer: Tokenizer, tokens: StructuralTokens, limits: Limits = Limits(),
                 pad_token_id: int = 0):
        self.tokenizer, self.tokens, self.limits, self.pad_token_id = tokenizer, tokens, limits, pad_token_id
        if type(pad_token_id) is not int or pad_token_id < 0:
            raise ValueError("pad_token_id must be nonnegative")
        if hasattr(tokenizer, "all_special_ids"):
            tokens.verify(tokenizer)
            if pad_token_id >= len(tokenizer):
                raise ValueError("padding token is outside the tokenizer vocabulary")

    def _text(self, text):
        ids = list(self.tokenizer.encode(text, add_special_tokens=False))
        if any(type(i) is not int or i < 0 for i in ids):
            raise ValueError("tokenizer returned invalid IDs")
        if set(ids) & set(vars(self.tokens).values()):
            raise ValueError("content contains a reserved structural token")
        return ids

    def encode(self, record: Record | dict, *, candidate_orders=None, question_order=None):
        if isinstance(record, dict):
            record = normalize_record(record, self.limits)
        if not isinstance(record, Record):
            raise ValueError("expected a normalized Record or request object")
        # Record objects are produced by normalization; still enforce configured release limits.
        if not 1 <= len(record.questions) <= self.limits.max_questions:
            raise ValueError("question count exceeds supported limits")
        question_order = list(range(len(record.questions))) if question_order is None else list(question_order)
        if sorted(question_order) != list(range(len(record.questions))):
            raise ValueError("question_order must be a permutation")
        candidate_orders = candidate_orders or {}
        if set(candidate_orders) - {q.id for q in record.questions}:
            raise ValueError("candidate order refers to an unknown question")
        ids = [self.tokens.state] + self._text(record.state)
        roles, segments, owners = [STATE]*len(ids), [0]*len(ids), [-1]*len(ids)
        positions = list(range(len(ids)))
        state_length = len(ids)
        questions, ends, decisions = [], [], []
        for segment, qi in enumerate(question_order, 1):
            q = record.questions[qi]
            maximum = self.limits.max_choice_candidates if q.type == "choice" else 10 if q.type == "score" else 2
            if not 2 <= len(q.candidates) <= maximum:
                raise ValueError("candidate count exceeds supported limits")
            order = list(candidate_orders.get(q.id, range(len(q.candidates))))
            if sorted(order) != list(range(len(q.candidates))):
                raise ValueError("candidate order must be a permutation")
            instruction = [self.tokens.question] + self._text(q.instruction)
            ids.extend(instruction)
            roles.extend([INSTRUCTION]*len(instruction))
            segments.extend([segment]*len(instruction))
            owners.extend([-1]*len(instruction))
            positions.extend(range(state_length, state_length+len(instruction)))
            p = state_length+len(instruction)-1
            branches = [[self.tokens.option] + self._text(q.candidates[i].description) + [self.tokens.option_end]
                        for i in order]
            qends = []
            for original_index, branch in zip(order, branches):
                ids.extend(branch)
                roles.extend([CANDIDATE]*len(branch))
                segments.extend([segment]*len(branch))
                owners.extend([original_index]*len(branch))
                positions.extend(range(p+1, p+1+len(branch)))
                qends.append(len(ids)-1)
            ids.append(self.tokens.decide)
            roles.append(DECISION)
            segments.append(segment)
            owners.append(-1)
            positions.append(p+max(map(len, branches))+1)
            questions.append(Question(q.id, q.type, q.instruction, tuple(q.candidates[i] for i in order),
                                      None if q.target is None else tuple(q.target[i] for i in order)))
            ends.append(qends)
            decisions.append(len(ids)-1)
        if len(ids) > self.limits.max_physical_tokens:
            raise ValueError("physical token budget exceeded; input was not truncated")
        if max(positions) >= self.limits.max_logical_positions:
            raise ValueError("logical position budget exceeded")
        if len(ids)**2*4 > self.limits.max_mask_bytes:
            raise ValueError("dense attention mask memory budget exceeded")
        return EncodedRecord(ids, positions, roles, segments, owners, tuple(questions), ends, decisions)

    def batch(self, records, **encode_options):
        encoded = [self.encode(r, **encode_options) for r in records]
        if not encoded:
            raise ValueError("batch cannot be empty")
        length = max(len(r.input_ids) for r in encoded)
        # Guard before allocating any quadratic tensor (including the Boolean topology).
        if len(encoded)*length**2*4 > self.limits.max_mask_bytes:
            raise ValueError("batched dense attention mask memory budget exceeded")
        padded = lambda field, pad: torch.tensor([getattr(r, field)+[pad]*(length-len(r.input_ids)) for r in encoded])
        roles, segments, owners = padded("roles", PADDING), padded("segments", -1), padded("candidate_owners", -1)
        allowed = attention_topology(roles, segments, owners)
        mask = torch.full(allowed.shape, float("-inf"), dtype=torch.float32).masked_fill_(allowed, 0)
        rows, decision_indices, candidate_indices, valid, targets, labeled, record_indices, questions = [], [], [], [], [], [], [], []
        width = max(len(q.candidates) for r in encoded for q in r.questions)
        for ri, r in enumerate(encoded):
            for qi, q in enumerate(r.questions):
                k = len(q.candidates)
                rows.append(ri)
                decision_indices.append(r.decisions[qi])
                candidate_indices.append(r.candidate_ends[qi]+[0]*(width-k))
                valid.append([True]*k+[False]*(width-k))
                targets.append(list(q.target or [0.]*k)+[0.]*(width-k))
                labeled.append(q.target is not None)
                record_indices.append(ri)
                questions.append(q)
        return Batch(padded("input_ids", self.pad_token_id), padded("position_ids", 0), mask[:, None],
                     torch.tensor(rows), torch.tensor(decision_indices), torch.tensor(candidate_indices),
                     torch.tensor(valid), torch.tensor(targets, dtype=torch.float32), torch.tensor(labeled),
                     torch.tensor(record_indices), tuple(questions), len(encoded))


def attention_topology(roles, segments, owners):
    length = roles.shape[-1]
    qr, kr = roles[:, :, None], roles[:, None, :]
    same_question = segments[:, :, None] == segments[:, None, :]
    same_candidate = owners[:, :, None] == owners[:, None, :]
    causal = torch.arange(length)[None, :] <= torch.arange(length)[:, None]
    real = (qr != PADDING) & (kr != PADDING)
    permitted = (kr == STATE) | (same_question & (
        ((qr >= INSTRUCTION) & (kr == INSTRUCTION)) |
        ((qr == CANDIDATE) & (kr == CANDIDATE) & same_candidate) |
        (qr == DECISION)))
    diagonal = torch.eye(length, dtype=torch.bool)[None]
    return (real & causal & permitted) | ((qr == PADDING) & diagonal)


@dataclass
class Batch:
    input_ids: torch.Tensor
    position_ids: torch.Tensor
    attention_mask: torch.Tensor
    rows: torch.Tensor
    decision_indices: torch.Tensor
    candidate_indices: torch.Tensor
    valid_candidates: torch.Tensor
    targets: torch.Tensor
    labeled: torch.Tensor
    record_indices: torch.Tensor
    questions: tuple[Question, ...]
    record_count: int

    def to(self, device):
        from dataclasses import fields, replace
        return replace(self, **{f.name: getattr(self, f.name).to(device) for f in fields(self)
                                if isinstance(getattr(self, f.name), torch.Tensor)})
