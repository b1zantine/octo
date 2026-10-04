"""Held-out metrics per primitive; confidence and correctness remain distinct."""
import math
import torch
from .model import categorical_loss, interpret


@torch.inference_mode()
def evaluate(model, batches, *, noul_threshold=.5):
    if not 0 <= noul_threshold <= 1:
        raise ValueError('Noul threshold must be in [0, 1]')
    sums, counts = {}, {}
    total_loss, records = 0., 0
    was_training = model.training
    model.eval()
    try:
        for batch in batches:
            output = model(batch)
            if batch.labeled.any():
                labeled_records = len(set(batch.record_indices[batch.labeled].tolist()))
                total_loss += categorical_loss(output, batch).item()*labeled_records
                records += labeled_records
            answers = interpret(output.probabilities,batch)
            flat_answers = [answer for group in answers for answer in group]
            for i,q in enumerate(batch.questions):
                if q.target is None:
                    continue
                p = output.probabilities[i,:len(q.candidates)].cpu()
                target = torch.tensor(q.target)
                logp = output.logits[i,:len(q.candidates)].float().log_softmax(-1).cpu()
                values = {f'{q.type}/nll': float(-(target*logp).sum())}
                if q.type == 'choice':
                    selected = flat_answers[i]['selected_id']
                    correct = {c.id for c,t in zip(q.candidates,q.target) if t == max(q.target)}
                    values['choice/accuracy'] = float(selected in correct)
                elif q.type == 'score':
                    levels = torch.tensor([c.value for c in q.candidates])
                    values['score/mae'] = abs(float((p*levels).sum()-(target*levels).sum()))
                else:
                    index = next(j for j,c in enumerate(q.candidates) if c.id == 'true')
                    values['noul/brier'] = float((p[index]-target[index])**2)
                    values['noul/accuracy'] = float((p[index]>=noul_threshold)==(target[index]>=noul_threshold))
                if q.type != 'noul':
                    values[f'{q.type}/mean_confidence'] = flat_answers[i]['confidence']
                for name,value in values.items():
                    sums[name] = sums.get(name,0.)+value
                    counts[name] = counts.get(name,0)+1
        if not records:
            raise ValueError('evaluation requires labeled records')
        return {'loss':total_loss/records, **{name:value/counts[name] for name,value in sums.items()}}
    finally:
        model.train(was_training)
