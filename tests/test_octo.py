import itertools
import math
from pathlib import Path
import tempfile
import unittest
import torch
from transformers import Qwen3Config, Qwen3Model, PreTrainedTokenizerFast
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from octo import (Encoder, Limits, OctoModel, StructuralTokens, categorical_loss, interpret,
                  load_checkpoint, make_optimizer, normalize_record, save_checkpoint, train_step)


def tokenizer():
    vocab = {word: i for i, word in enumerate([
        '[UNK]', '[PAD]', '<state>', '<question>', '<option>', '</option>', '<decide>',
        'report', 'broken', 'available', 'Which', 'team', 'payment', 'product', 'functionality',
        'extra', 'True', 'False', 'minor', 'major', 'Is', 'the', 'following', 'statement',
        'true', 'of', 'state', '?', 'workaround', 'billing', 'engineering', 'cosmetic'])}
    backend = Tokenizer(WordLevel(vocab, unk_token='[UNK]'))
    backend.pre_tokenizer = Whitespace()
    return PreTrainedTokenizerFast(tokenizer_object=backend, unk_token='[UNK]', pad_token='[PAD]',
        additional_special_tokens=['<state>', '<question>', '<option>', '</option>', '<decide>'])


def request():
    return {'state': 'report broken', 'questions': [{'id': 'route', 'type': 'choice',
        'instruction': 'Which team', 'options': [
            {'id': 'b', 'description': 'payment'},
            {'id': 'a', 'description': 'product functionality'},
            {'id': 'c', 'description': 'extra major report'}], 'label_id': 'a'}]}


def model(backend='eager', lora=False):
    torch.manual_seed(17)
    config = Qwen3Config(vocab_size=32, hidden_size=32, intermediate_size=64,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2, head_dim=8,
        max_position_embeddings=256, attention_dropout=0.)
    config._attn_implementation = backend
    backbone = Qwen3Model(config)
    if lora:
        from peft import LoraConfig, get_peft_model
        from octo.model import LORA_TARGETS
        backbone = get_peft_model(backbone, LoraConfig(r=4, lora_alpha=8, lora_dropout=0.,
                                                     target_modules=list(LORA_TARGETS)))
    return OctoModel(backbone, freeze_backbone=not lora).eval()


class OctoTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.encoder = Encoder(tokenizer(), StructuralTokens(2, 3, 4, 5, 6),
                               Limits(max_questions=3), pad_token_id=1)

    def test_layout_and_every_mask_edge(self):
        r = request()
        r['questions'][0]['options'] = r['questions'][0]['options'][:2]
        encoded = self.encoder.encode(r)
        self.assertEqual(encoded.position_ids, [0,1,2,3,4,5,6,7,8,6,7,8,9,10])
        self.assertEqual(encoded.candidate_ends, [[8,12]])
        batch = self.encoder.batch([r])
        for i, role in enumerate(encoded.roles):
            for j, key in enumerate(encoded.roles):
                expected = j <= i and (key == 0 or
                    (role in (1,2,3) and key == 1) or
                    (role == key == 2 and encoded.candidate_owners[i] == encoded.candidate_owners[j]) or role == 3)
                self.assertEqual(batch.attention_mask[0,0,i,j].item() == 0, expected)
        output = model().backbone(input_ids=batch.input_ids, position_ids=batch.position_ids,
            attention_mask=batch.attention_mask, use_cache=False, output_attentions=True)
        for weights in output.attentions:
            forbidden = torch.isneginf(batch.attention_mask).expand_as(weights)
            self.assertEqual(weights[forbidden].abs().max().item(), 0.)

    def test_permutations_isolation_and_backends(self):
        for backend in ('eager', 'sdpa'):
            m = model(backend)
            with torch.no_grad():
                baseline = m(self.encoder.batch([request()]))
                for order in itertools.permutations(range(3)):
                    result = m(self.encoder.batch([request()], candidate_orders={'route': order}))
                    inverse = [order.index(i) for i in range(3)]
                    torch.testing.assert_close(result.candidate_states[:,inverse], baseline.candidate_states, atol=1e-6, rtol=1e-5)
                    torch.testing.assert_close(result.probabilities[:,inverse], baseline.probabilities, atol=1e-6, rtol=1e-5)
                changed = request()
                changed['questions'][0]['options'][1]['description'] = 'major available'
                result = m(self.encoder.batch([changed]))
                torch.testing.assert_close(result.candidate_states[:,[0,2]], baseline.candidate_states[:,[0,2]], atol=0, rtol=0)

    def test_packing_and_batch_padding(self):
        r = request()
        other = {'id': 'truth', 'type': 'noul', 'statement': 'workaround available', 'truth_target': .8}
        packed = {'state': r['state'], 'questions': r['questions']+[other]}
        shorter = {'state': 'report', 'questions': [other]}
        for backend in ('eager', 'sdpa'):
            m = model(backend)
            with torch.no_grad():
                alone = m(self.encoder.batch([r])).probabilities
                for order in ([0,1], [1,0]):
                    b = self.encoder.batch([packed], question_order=order)
                    result = m(b)
                    torch.testing.assert_close(result.probabilities[order.index(0)], alone[0], atol=1e-6, rtol=1e-5)
                b = self.encoder.batch([packed, shorter])
                result = m(b)
                self.assertEqual(result.probabilities[1,-1].item(), 0.)
                torch.testing.assert_close(result.probabilities[2,:2], m(self.encoder.batch([shorter])).probabilities[0], atol=1e-6, rtol=1e-5)
                self.assertTrue(torch.isfinite(categorical_loss(result, b)))

    def test_typed_outputs_and_targets(self):
        r = {'state':'report', 'questions':[
            {'id':'s', 'type':'score', 'instruction':'major', 'criteria':['cosmetic','minor','major'],
             'score_target':1.43, 'score_interpolation':'adjacent'},
            {'id':'n', 'type':'noul', 'statement':'report', 'truth_target':.7},
            request()['questions'][0]]}
        b = self.encoder.batch([r], candidate_orders={'s':[2,0,1], 'n':[1,0], 'route':[2,0,1]})
        p = torch.tensor([[.43,0.,.57],[.3,.7,0.],[1/3,1/3,1/3]])
        answers = interpret(p,b)[0]
        self.assertAlmostEqual(answers[0]['score'],1.43,places=6)
        self.assertAlmostEqual(answers[1]['probability_true'],.7,places=6)
        self.assertEqual(answers[2]['selected_id'],'a')
        self.assertAlmostEqual(answers[2]['confidence'],0.,places=6)
        torch.testing.assert_close(b.targets[0], p[0])

    def test_record_weighting_and_padding_gradient(self):
        from octo.model import ModelOutput
        r = request()
        packed = {'state':r['state'], 'questions': r['questions']+[
            {'id':'n','type':'noul','statement':'report','label_id':'true'}]}
        b = self.encoder.batch([packed,r])
        z = torch.tensor([[0.,1.,2.],[2.,0.,float('-inf')],[2.,0.,1.]], requires_grad=True)
        out = ModelOutput(z,z.softmax(-1),None,None)
        loss = categorical_loss(out,b)
        expected = ((-z[0].log_softmax(0)[1]-z[1,:2].log_softmax(0)[0])/2-z[2].log_softmax(0)[1])/2
        torch.testing.assert_close(loss,expected)
        loss.backward()
        self.assertTrue(torch.isfinite(z.grad).all())
        self.assertEqual(z.grad[1,2].item(),0.)

    def test_validation(self):
        cases = []
        r=request(); r['questions'][0]['options'][1]['id']='b'; cases.append(r)
        r=request(); r['questions'][0]['label_id']='unknown'; cases.append(r)
        r=request(); r['questions'][0]['target']={'a':1.,'b':0.,'c':0.}; cases.append(r)
        r=request(); r['state']='<state>'; cases.append(r)
        r=request(); del r['questions'][0]['label_id']; r['questions'][0]['target']={'a':float('nan'),'b':0.,'c':0.}; cases.append(r)
        for r in cases:
            with self.assertRaises(ValueError): self.encoder.batch([r])
        for limits in (Limits(max_physical_tokens=5), Limits(max_logical_positions=5), Limits(max_mask_bytes=4)):
            with self.assertRaises(ValueError): Encoder(tokenizer(),self.encoder.tokens,limits).batch([request()])
        with self.assertRaises(ValueError): normalize_record({'state':'x','questions':[]})
        with self.assertRaises(ValueError): StructuralTokens(1,1,2,3,4)

    def test_training_frozen_weights_and_memorization(self):
        b = self.encoder.batch([request()])
        for lora in (False, True):
            m=model(lora=lora)
            frozen = {n:p.detach().clone() for n,p in m.named_parameters() if not p.requires_grad}
            initial = {n:p.detach().clone() for n,p in m.named_parameters() if p.requires_grad}
            optimizer=make_optimizer(m)
            initial_loss=categorical_loss(m(b),b).item()
            for _ in range(60): train_step(m,b,optimizer)
            m.eval()
            self.assertLess(categorical_loss(m(b),b).item(), initial_loss*.1)
            self.assertGreater(m(b).probabilities[0,1].item(), .95)
            for n,p in m.named_parameters():
                if n in frozen:
                    self.assertTrue(torch.equal(p,frozen[n])); self.assertIsNone(p.grad)
            self.assertFalse(torch.equal(m.query.weight,initial['query.weight']))
            if lora:
                self.assertTrue(any(not torch.equal(p,initial[n]) for n,p in m.named_parameters() if 'lora_' in n))

    def test_adapter_save_reload(self):
        from transformers import AutoModel
        from peft import LoraConfig, get_peft_model
        from octo.model import LORA_TARGETS
        with tempfile.TemporaryDirectory() as tmp:
            base_path = Path(tmp)/'base'
            model().backbone.save_pretrained(base_path)
            backbone = AutoModel.from_pretrained(base_path, attn_implementation='sdpa')
            backbone.config._commit_hash = 'local-fixture-revision'
            adapted = get_peft_model(backbone, LoraConfig(r=4, lora_alpha=8, lora_dropout=0.,
                target_modules=list(LORA_TARGETS)))
            m = OctoModel(adapted, freeze_backbone=False)
            b = self.encoder.batch([request()])
            train_step(m, b, make_optimizer(m))
            before = m.predict(b)
            save_checkpoint(Path(tmp)/'adapter', m, self.encoder)
            loaded, encoder, _ = load_checkpoint(Path(tmp)/'adapter')
            self.assertEqual(before, loaded.predict(encoder.batch([request()])))
            self.assertTrue(any(p.requires_grad for n,p in loaded.named_parameters() if 'lora_' in n))

    def test_evaluation_metrics(self):
        from octo.cli import demo_records, tiny_setup
        from octo.evaluation import evaluate
        m,t,tokens = tiny_setup('sdpa')
        encoder = Encoder(t,tokens)
        batches = [encoder.batch([r]) for r in demo_records()]
        metrics = evaluate(m, batches)
        self.assertEqual(set(metrics), {'loss','choice/nll','choice/accuracy','choice/mean_confidence',
            'score/nll','score/mae','score/mean_confidence','noul/nll','noul/brier','noul/accuracy'})
        self.assertTrue(all(math.isfinite(v) for v in metrics.values()))
        self.assertAlmostEqual(metrics['loss'], sum(metrics[k+'/nll'] for k in ('choice','score','noul'))/3)

    def test_save_reload(self):
        m=model('sdpa')
        b=self.encoder.batch([request()])
        before=m.predict(b)
        with tempfile.TemporaryDirectory() as tmp:
            save_checkpoint(Path(tmp)/'checkpoint',m,self.encoder,metadata={'dataset_version':'fixture-v1'})
            loaded, encoder, metadata=load_checkpoint(Path(tmp)/'checkpoint')
            self.assertEqual(metadata['dataset_version'],'fixture-v1')
            self.assertEqual(before,loaded.predict(encoder.batch([request()])))


if __name__ == '__main__': unittest.main()
