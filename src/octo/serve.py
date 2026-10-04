"""Local checkpoint API for the playground and chess demo."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import random
import threading
import time

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field, JsonValue

from .data import normalize_record


class Request(BaseModel):
    model_config = ConfigDict(extra="forbid")
    state: JsonValue
    model: str = "octo-latest"
    questions: dict[str, dict[str, JsonValue]] = Field(min_length=1, max_length=32)


class PermutationRequest(BaseModel):
    request: Request
    question: str
    n_perm: int = Field(default=6, ge=1, le=64)
    seed: int = 0


def text(value):
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, allow_nan=False)


def to_record(request, limits):
    """Translate the UI's question map into Octo's normalized record contract."""
    questions = []
    for qid, question in request.questions.items():
        kind = question.get("type")
        instruction = question.get("instructions")
        if instruction is None or instruction == "":
            raise ValueError("instructions must be nonempty")
        q = {"id": qid, "type": kind}
        criteria = question.get("criteria")
        if kind == "choice":
            if not isinstance(criteria, dict):
                raise ValueError("Choice criteria must be an object")
            q.update(instruction=text(instruction), options=[
                {"id": key, "description": key if value is None else text(value)}
                for key, value in criteria.items()
            ])
        elif kind == "score":
            if not isinstance(criteria, list):
                raise ValueError("Score criteria must be an array")
            q.update(instruction=text(instruction), criteria=[text(c) for c in criteria])
        elif kind == "noul":
            statement = text(instruction)
            if criteria is not None:
                if not isinstance(criteria, dict) or set(criteria) - {"true", "false"}:
                    raise ValueError("Noul criteria may only contain true and false")
                statement += "\nCriteria: " + text(criteria)
            q["statement"] = statement
        questions.append(q)
    return normalize_record({"state": text(request.state), "questions": questions}, limits)


class CheckpointService:
    def __init__(self, model, encoder, checkpoint, device):
        self.model, self.encoder = model, encoder
        self.checkpoint, self.device = str(checkpoint), device
        # Predict changes eval/train state; serialize requests against one model.
        self.lock = threading.Lock()

    def info(self):
        return {"models": [{"name": "octo-latest", "run": self.checkpoint,
                            "base": self.model.backbone.config._name_or_path}]}

    def answer(self, request):
        if request.model != "octo-latest":
            raise ValueError("unknown model; use octo-latest")
        started = time.perf_counter()
        record = to_record(request, self.encoder.limits)
        with self.lock:
            batch = self.encoder.batch([record]).to(self.device)
            predictions = self.model.predict(batch)[0]
        answers = {}
        for question, prediction in zip(record.questions, predictions):
            kind = question.type
            if kind == "noul":
                answer = {"type": kind, "noul": prediction["probability_true"]}
            else:
                answer = {"type": kind, "confidence": prediction["confidence"],
                          "probabilities": prediction["probabilities"]}
                if kind == "choice":
                    answer["choice"] = prediction["selected_id"]
                else:
                    answer.update(score=prediction["score"], legend={
                        c.id: c.description for c in question.candidates})
            answers[question.id] = answer
        return {"model": request.model, "answers": answers,
                "usage": {"input_tokens": int(batch.input_ids.numel()), "output_tokens": 0},
                "latency_ms": round((time.perf_counter() - started) * 1000, 2)}


def create_app(service):
    app = FastAPI(title="Octo playground")

    def answer(request):
        try:
            return service.answer(request)
        except (ValueError, TypeError) as error:
            raise HTTPException(422, str(error)) from error

    @app.get("/v1/models")
    def models():
        return service.info()

    @app.post("/v1/systemone")
    def systemone(request: Request):
        return answer(request)

    @app.post("/v1/systemone/separate")
    def separate(request: Request):
        started = time.perf_counter()
        parts = [answer(request.model_copy(update={"questions": {qid: q}}))
                 for qid, q in request.questions.items()]
        return {"model": request.model,
                "answers": {qid: a for part in parts for qid, a in part["answers"].items()},
                "usage": {key: sum(p["usage"][key] for p in parts)
                          for key in ("input_tokens", "output_tokens")},
                "latency_ms": round((time.perf_counter() - started) * 1000, 2)}

    @app.post("/v1/systemone/permute")
    def permute(body: PermutationRequest):
        q = body.request.questions.get(body.question)
        if q is None or q.get("type") != "choice" or not isinstance(q.get("criteria"), dict):
            raise HTTPException(422, "question must identify an existing Choice question")
        rng = random.Random(body.seed)
        keys = list(q["criteria"])
        runs = []
        for index in range(body.n_perm):
            order = list(keys)
            if index:
                rng.shuffle(order)
            question = {**q, "criteria": {key: q["criteria"][key] for key in order}}
            result = answer(body.request.model_copy(update={"questions": {body.question: question}}))
            a = result["answers"][body.question]
            runs.append({"order": order, "probabilities": a["probabilities"],
                         "choice": a["choice"], "latency_ms": result["latency_ms"]})
        return {"runs": runs, "argmax_stable": len({r["choice"] for r in runs}) == 1,
                "spread": {key: max(r["probabilities"][key] for r in runs)
                           - min(r["probabilities"][key] for r in runs) for key in keys}}

    return app


def default_checkpoint(root=None):
    """Resolve the downloaded Colab best checkpoint, including its original /content path."""
    root = Path(root) if root is not None else Path(__file__).resolve().parents[2]
    run = root / "artifacts/runs/octo_v1_colab_1"
    manifest = run / "best_checkpoint.json"
    if not manifest.is_file():
        raise ValueError("Colab best_checkpoint.json is missing; supply --checkpoint")
    saved = Path(json.loads(manifest.read_text())["path"])
    checkpoint = run / "checkpoints" / saved.name
    for required in ("octo.json", "pointer.pt", "backbone", "tokenizer"):
        if not (checkpoint / required).exists():
            raise ValueError(f"Colab checkpoint is incomplete: {checkpoint}; supply --checkpoint")
    return checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", help="directory containing octo.json; default: best Colab checkpoint")
    parser.add_argument("--device", default="cpu", help="cpu, cuda, or mps")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8009)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--allow-download", action="store_true", help="allow loading uncached base weights")
    args = parser.parse_args()
    try:
        checkpoint = Path(args.checkpoint) if args.checkpoint else default_checkpoint()
    except ValueError as error:
        parser.error(str(error))
    from .checkpoint import load_checkpoint
    import uvicorn
    model, encoder, _ = load_checkpoint(checkpoint, device=args.device,
                                        local_files_only=not args.allow_download)
    capacity = model.backbone.config.max_position_embeddings
    # Demo-only inference limits: training/checkpoint limits remain unchanged.
    encoder.limits = replace(encoder.limits, max_questions=32, max_choice_candidates=255,
                            max_physical_tokens=min(args.max_tokens, capacity),
                            max_logical_positions=min(args.max_tokens, capacity),
                            max_mask_bytes=256 * 1024 * 1024)
    print(f"Loading playground checkpoint: {checkpoint}", flush=True)
    uvicorn.run(create_app(CheckpointService(model, encoder, checkpoint, args.device)),
                host=args.host, port=args.port)


if __name__ == "__main__":
    main()
