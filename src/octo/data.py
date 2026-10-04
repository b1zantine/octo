"""Typed request normalization. Identities and supervision never enter model text."""
from dataclasses import dataclass, field
import math
from typing import Any

NOUL_TEMPLATE = "Is the following statement true of the state?"
NORMALIZATION_VERSION = 1


@dataclass(frozen=True)
class Limits:
    max_physical_tokens: int = 512
    max_logical_positions: int = 512
    max_mask_bytes: int = 64 * 1024 * 1024
    max_choice_candidates: int = 8
    max_questions: int = 1

    def __post_init__(self):
        for name, value in vars(self).items():
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if not 2 <= self.max_choice_candidates <= 255:
            raise ValueError("Choice limit must be between 2 and 255")


@dataclass(frozen=True)
class Candidate:
    id: str
    description: str
    value: int | None = None

    def __post_init__(self):
        _text(self.id, "candidate ID")
        _text(self.description, "candidate description")
        if self.value is not None and (type(self.value) is not int or self.value < 0):
            raise ValueError("candidate value must be a nonnegative integer")


@dataclass(frozen=True)
class Question:
    id: str
    type: str
    instruction: str
    candidates: tuple[Candidate, ...]
    target: tuple[float, ...] | None = None

    def __post_init__(self):
        _text(self.id, "question ID")
        _text(self.instruction, "instruction")
        if self.type not in ("choice", "score", "noul"):
            raise ValueError("invalid question type")
        if not isinstance(self.candidates, tuple) or any(not isinstance(c, Candidate) for c in self.candidates):
            raise ValueError("candidates must be a tuple of Candidate objects")
        ids = [c.id for c in self.candidates]
        maximum = {"choice": 255, "score": 10, "noul": 2}[self.type]
        if not 2 <= len(ids) <= maximum or len(set(ids)) != len(ids):
            raise ValueError("invalid candidate count or duplicate identity")
        if self.type == "score" and {c.value for c in self.candidates} != set(range(len(ids))):
            raise ValueError("Score values must be immutable original rubric indices")
        if self.type != "score" and any(c.value is not None for c in self.candidates):
            raise ValueError("only Score candidates have values")
        if self.type == "noul":
            if {c.id: c.description for c in self.candidates} != {"true":"True", "false":"False"}:
                raise ValueError("Noul requires normalized True/False candidates")
            if not self.instruction.startswith(NOUL_TEMPLATE + "\n"):
                raise ValueError("Noul instruction must use the versioned template")
        if self.target is not None:
            if len(self.target) != len(ids):
                raise ValueError("target length must match candidate count")
            values = [_number(t, "target") for t in self.target]
            if any(t < 0 for t in values) or not math.isclose(sum(values), 1., abs_tol=1e-6):
                raise ValueError("targets must be nonnegative and normalized")


@dataclass(frozen=True)
class Record:
    state: str
    questions: tuple[Question, ...]
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not isinstance(self.state, str):
            raise ValueError("state must be text")
        if not isinstance(self.questions, tuple) or not self.questions or any(not isinstance(q, Question) for q in self.questions):
            raise ValueError("questions must be a nonempty tuple of Question objects")
        if len({q.id for q in self.questions}) != len(self.questions):
            raise ValueError("duplicate question ID")


def _text(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return value


def _number(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be finite numeric data")
    return float(value)


def normalize_record(raw: dict, limits: Limits = Limits()) -> Record:
    if not isinstance(raw, dict):
        raise ValueError("record must be an object")
    state = raw.get("state")
    if not isinstance(state, str):
        raise ValueError("state must be a string")
    questions = raw.get("questions")
    if not isinstance(questions, list) or not 1 <= len(questions) <= limits.max_questions:
        raise ValueError("question count exceeds supported limits or is empty")
    normalized, question_ids = [], set()
    for q in questions:
        if not isinstance(q, dict):
            raise ValueError("question must be an object")
        identity = _text(q.get("id"), "question ID")
        if identity in question_ids:
            raise ValueError("duplicate question ID")
        question_ids.add(identity)
        kind = q.get("type")
        if kind == "choice":
            instruction = _text(q.get("instruction"), "instruction")
            options = q.get("options")
            if not isinstance(options, list) or not 2 <= len(options) <= limits.max_choice_candidates:
                raise ValueError("Choice requires 2 to the supported maximum candidates")
            if any(not isinstance(o, dict) for o in options):
                raise ValueError("Choice options must be objects")
            candidates = tuple(Candidate(_text(o.get("id"), "candidate ID"),
                                         _text(o.get("description"), "description")) for o in options)
        elif kind == "score":
            instruction = _text(q.get("instruction"), "instruction")
            criteria = q.get("criteria")
            if not isinstance(criteria, list) or not 2 <= len(criteria) <= 10:
                raise ValueError("Score requires 2–10 descriptive criteria")
            candidates = tuple(Candidate(str(i), _text(text, "criterion"), i)
                               for i, text in enumerate(criteria))
        elif kind == "noul":
            instruction = NOUL_TEMPLATE + "\n" + _text(q.get("statement"), "statement")
            candidates = (Candidate("true", "True"), Candidate("false", "False"))
        else:
            raise ValueError("question type must be choice, score, or noul")
        ids = [c.id for c in candidates]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate candidate ID")
        label_fields = [name for name in ("label_id", "target", "truth_target", "score_target") if name in q]
        if len(label_fields) > 1:
            raise ValueError("supply exactly one supervision field")
        target = None
        if label_fields:
            name = label_fields[0]
            if name == "label_id":
                label = q[name]
                if not isinstance(label, str) or label not in ids:
                    raise ValueError("label_id must identify a candidate")
                target = tuple(float(c.id == label) for c in candidates)
            elif name == "target":
                distribution = q[name]
                if not isinstance(distribution, dict) or set(distribution) != set(ids):
                    raise ValueError("target must map every candidate ID to a probability")
                target = tuple(_number(distribution[i], "target probability") for i in ids)
                if any(p < 0 for p in target) or not math.isclose(sum(target), 1, abs_tol=1e-6):
                    raise ValueError("target must be nonnegative and sum to one")
            elif name == "truth_target":
                if kind != "noul":
                    raise ValueError("truth_target requires Noul")
                r = _number(q[name], name)
                if not 0 <= r <= 1:
                    raise ValueError("truth_target must be in [0, 1]")
                target = (r, 1-r)
            else:
                if kind != "score" or q.get("score_interpolation") != "adjacent":
                    raise ValueError("score_target requires Score and explicit adjacent interpolation")
                value = _number(q[name], name)
                if not 0 <= value <= len(ids)-1:
                    raise ValueError("score_target is outside the rubric")
                lower, upper = math.floor(value), math.ceil(value)
                target = tuple(float(i == lower) if lower == upper else
                               (upper-value if i == lower else value-lower if i == upper else 0.)
                               for i in range(len(ids)))
        normalized.append(Question(identity, kind, instruction, candidates, target))
    return Record(state, tuple(normalized), {k: v for k, v in raw.items() if k not in ("state", "questions")})
