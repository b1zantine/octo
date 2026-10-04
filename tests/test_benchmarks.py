from argparse import Namespace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from octo.benchmark_data import neutralize,records,sample
from octo.benchmark import (JevAdapter,FatalAPIError,jev_payload,prompt_for,score,validate_result,source_group_ci,compare,prediction_manifest,run)
from octo.dataset import digest,write_json


def example():
    return {'record_id':'original-001','domain':'banking77','source_group_id':'family',
        'state':'Please help change my PIN.','provenance':{'source_split':'test'},
        'questions':[{'id':'secret-question','type':'choice','instruction':'Choose the request category.',
            'options':[{'id':'banking:change_pin','description':'Changing a card PIN.'},
                       {'id':'banking:lost_card','description':'Reporting a lost card.'}], 'label_id':'banking:change_pin'}]}


class BenchmarkTests(unittest.TestCase):
    def test_colab_checkpoint_lineage_accepts_training_hash_only(self):
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint=Path(temporary)
            args=Namespace(benchmark='artifacts/benchmarks/bfsi-v1',suite='core',
                output='artifacts/evaluations/lineage-test',adapter='octo',device='mps',
                checkpoint=str(checkpoint),resume=False)
            manifest={'training_file_sha256':'train-hash'}
            for key,accepted in [('bfsi/train',True),('bfsi/dev',False)]:
                write_json(checkpoint/'octo.json',{'metadata':{'dataset_sha256':{key:'train-hash'}}})
                with patch('octo.benchmark.verify_bundle',return_value=(manifest,{})),\
                     patch('octo.benchmark.digest',return_value='hash'),\
                     patch('octo.benchmark.write_json',side_effect=RuntimeError('lineage accepted')),\
                     patch('pathlib.Path.mkdir'):
                    with self.assertRaisesRegex(RuntimeError if accepted else ValueError,
                        'lineage accepted' if accepted else 'lineage does not match'):
                        run(args)

    def test_gold_and_semantic_ids_never_enter_any_adapter_input(self):
        request,gold=neutralize(example())
        self.assertEqual(gold['label_id'],'A')
        self.assertEqual({o['id'] for o in request['questions'][0]['options']},{'A','B'})
        for payload in (json.dumps(request),json.dumps(jev_payload(request,'jev-1.13.0')),prompt_for(request)):
            for forbidden in ['label_id','provenance','source_group_id','banking:change_pin','secret-question']:
                self.assertNotIn(forbidden,payload)
        self.assertEqual(request['state'],example()['state'])

    def test_invalid_distribution_or_choice_rejected(self):
        request,_=neutralize(example())
        for result in [{'selected_id':'C'},{'selected_id':'A','probabilities':{'A':.3}},
                       {'selected_id':'A','probabilities':{'A':.3,'B':.3}},
                       {'selected_id':'A','probabilities':{'A':float('nan'),'B':1}}]:
            with self.assertRaises(ValueError):validate_result(result,request)

    def test_jev_official_schema_and_version_without_live_request(self):
        request,_=neutralize(example())
        config=json.loads(Path('configs/bfsi_benchmark_v1.json').read_text())
        with patch('octo.benchmark.key_from_env',return_value='test-key'),patch('requests.Session') as session:
            response=session.return_value.post.return_value
            response.status_code=200
            response.json.return_value={'model':'jev-1.13.0','answers':{'decision':{'choice':'A','probabilities':{'A':.9,'B':.1}}},'usage':{'input_tokens':12}}
            result=JevAdapter(config)(request)
            self.assertEqual(result['selected_id'],'A')
            kwargs=session.return_value.post.call_args.kwargs
            self.assertFalse(kwargs['allow_redirects'])
            self.assertEqual(kwargs['json']['questions']['decision']['instructions'],request['questions'][0]['instruction'])
            response.json.return_value['model']='unexpected-version'
            with self.assertRaises(FatalAPIError):JevAdapter(config)(request)

    def test_endpoint_cannot_redirect_credentials_to_other_host(self):
        config=json.loads(Path('configs/bfsi_benchmark_v1.json').read_text());config['jev_endpoint']='https://example.com'
        with self.assertRaises(ValueError):JevAdapter(config)

    def test_errors_count_as_wrong_and_probability_metrics_have_coverage(self):
        _,gold=neutralize(example());other={**gold,'record_id':'other','source_group_id':'other'}
        predictions=[{'record_id':gold['record_id'],'selected_id':'A','probabilities':{'A':.8,'B':.2},'error':None,'latency_seconds':.2},
            {'record_id':'other','selected_id':None,'probabilities':None,'error':'Timeout','latency_seconds':1}]
        stats=score(predictions,[gold,other],{'bootstrap_samples':50,'seed':17})
        self.assertEqual(stats['accuracy'],.5);self.assertEqual(stats['error_rate'],.5)
        self.assertEqual(stats['probability_coverage'],.5)
        with self.assertRaises(ValueError):score(predictions[:1],[gold,other],{'bootstrap_samples':50,'seed':17})
        self.assertEqual(source_group_ci([1.,1.],['same','same'],10,17),[1.,1.])

    def test_frozen_benchmarks_are_disjoint_and_label_free(self):
        self.assertFalse(Path('configs/bfsi_chess_v2.json').exists())
        for domain in ['bfsi','chess']:
            config=json.loads(Path(f'configs/{domain}_v2.json').read_text())
            self.assertEqual('lichess' in config['sources'],domain=='chess')
            root=Path(f'artifacts/benchmarks/{domain}-v1')
            if not root.exists():continue
            manifest=json.loads((root/'manifest.json').read_text())
            for name,expected in manifest['files'].items():self.assertEqual(digest(root/name),expected)
            full={r['record_id'] for r in records(root/'full.inputs.jsonl')}
            core={r['record_id'] for r in records(root/'core.inputs.jsonl')}
            dev={r['record_id'] for r in records(root/'dev.inputs.jsonl')}
            self.assertLessEqual(core,full);self.assertFalse(dev & full)
            if domain=='bfsi':
                core_gold=list(records(root/'core.gold.jsonl'))
                full_gold=list(records(root/'full.gold.jsonl'))
                for source in config['sources']:
                    a={g['provenance']['original_intent'] for g in core_gold if g['domain']==source}
                    b={g['provenance']['original_intent'] for g in full_gold if g['domain']==source}
                    self.assertEqual(a,b)
            for suite in ['dev','core','full']:
                gold={r['record_id']:r for r in records(root/f'{suite}.gold.jsonl')}
                for request in records(root/f'{suite}.inputs.jsonl'):
                    self.assertEqual(set(request),{'record_id','state','questions'})
                    q=request['questions'][0]
                    self.assertEqual(set(q),{'id','type','instruction','options'})
                    self.assertIn(gold[request['record_id']]['label_id'],{o['id'] for o in q['options']})

    def test_training_accepts_separate_files_and_preserves_both_hashes(self):
        from octo.cli import main,demo_records
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            first=root/'bfsi';second=root/'chess';first.mkdir();second.mkdir()
            examples=demo_records()
            for directory,row in [(first,examples[0]),(second,examples[2])]:
                (directory/'train.jsonl').write_text(json.dumps(row)+'\n')
            checkpoint=root/'checkpoint'
            args=['octo-train','--tiny','--epochs','1','--train',str(first/'train.jsonl'),
                  '--train',str(second/'train.jsonl'),'--checkpoint',str(checkpoint)]
            with patch('sys.argv',args):main()
            config=json.loads((checkpoint/'octo.json').read_text())['metadata']
            self.assertEqual(config['training_records'],2)
            self.assertEqual(len(config['train_file_sha256']),2)
            self.assertEqual(config['train_file_sha256'][str(first/'train.jsonl')],digest(first/'train.jsonl'))

    def test_paired_report_compares_complete_identical_records(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)/'benchmark';root.mkdir()
            request,gold=neutralize(example())
            for name,value in [('core.inputs.jsonl',request),('core.gold.jsonl',gold)]:
                (root/name).write_text(json.dumps(value)+'\n')
            write_json(root/'config.json',{'bootstrap_samples':20,'seed':17})
            write_json(root/'manifest.json',{'files':{p.name:digest(p) for p in root.iterdir()}})
            paths=[]
            for adapter,selected in [('qwen-prompt','A'),('jev','B'),('octo','A')]:
                path=Path(temporary)/adapter;path.mkdir();paths.append(str(path))
                write_json(path/'run_config.json',{'adapter':adapter,'suite':'core','benchmark_manifest_sha256':digest(root/'manifest.json'),'inputs_sha256':digest(root/'core.inputs.jsonl')})
                (path/'predictions.jsonl').write_text(json.dumps({'record_id':gold['record_id'],'selected_id':selected,'probabilities':None,'error':None,'latency_seconds':.1})+'\n')
                prediction_manifest(path,json.loads((path/'run_config.json').read_text()))
            output=Path(temporary)/'report'
            compare(Namespace(benchmark=str(root),suite='core',output=str(output),predictions=paths))
            report=json.loads((output/'comparison.json').read_text())
            self.assertEqual(report['paired']['jev minus octo']['accuracy_difference'],-1)
            self.assertEqual(report['paired']['jev minus octo']['b_wins'],1)

if __name__=='__main__':unittest.main()
