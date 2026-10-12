"""Offline-only request/data/package tests; no keys, network or model dependencies."""
import copy
import json
import tempfile
from pathlib import Path
import unittest
from infra import openrouter as api
from infra.source_inventory import load_primary,select_rows,inventory,FIELDS
from tools.verify_release import verify


class RequestTests(unittest.TestCase):
    def setUp(self):
        self.config=api.load_json(api.ROOT/'configs/luna_standard.json')
        self.messages=api.load_json(api.ROOT/'prompts/smoke_messages.json')
        self.metadata=api.load_json(api.ROOT/self.config['metadata_snapshot'])
        self.response={'model':'openai/gpt-5.6-luna','provider':'OpenAI','service_tier':'default',
                       'choices':[{'finish_reason':'stop','message':{'content':'{"utterance":"We propose reciprocal de-escalation."}'}}]}

    def test_request_snapshot_and_seed(self):
        a=api.build_request(self.config,self.messages,0)
        self.assertEqual(a,api.build_request(self.config,self.messages,0))
        self.assertNotEqual(api.digest(a),api.digest(api.build_request(self.config,self.messages,1)))
        api.check_metadata(self.metadata,a)
        self.assertNotIn('temperature',a)
        self.assertFalse(self.config['execution_enabled'])

    def test_mismatched_model_metadata(self):
        self.config['request']['model']='openai/gpt-5.6-sol'
        with self.assertRaises(ValueError):api.build_request(self.config,self.messages,0)

    def test_unapproved_model(self):
        self.config['request']['model']='openai/gpt-5.6-terra'
        with self.assertRaises(ValueError):api.build_request(self.config,self.messages,0)

    def test_sol_authorized_separately(self):
        config=api.load_json(api.ROOT/'configs/sol_standard.json')
        config['request']['reasoning']['effort']='medium';config['request']['max_tokens']=4096
        request=api.build_request(config,self.messages,0)
        api.check_metadata(api.load_json(api.ROOT/'sol_results/api/metadata_live.json'),request)
        result=dict(self.response,model='openai/gpt-5.6-sol-20260709')
        self.assertIn('utterance',api.validate_response(result,expected_model='openai/gpt-5.6-sol'))
        with self.assertRaises(ValueError):api.validate_response(result)
        self.config['request']['max_tokens']=4096
        with self.assertRaises(ValueError):api.build_request(self.config,self.messages,0)

    def test_standard_only(self):
        self.config['request']['service_tier']='flex'
        with self.assertRaises(ValueError):api.build_request(self.config,self.messages,0)

    def test_no_fallback(self):
        self.config['request']['provider']['allow_fallbacks']=True
        with self.assertRaises(ValueError):api.build_request(self.config,self.messages,0)

    def test_price_limit(self):
        request=api.build_request(self.config,self.messages,0)
        request['provider']['max_price']['prompt']=0
        with self.assertRaises(ValueError):api.check_metadata(self.metadata,request)

    def test_schema_and_response_routing(self):
        self.assertIn('utterance',api.validate_response(self.response))
        for key,value in [('model','other'),('provider','Azure'),('service_tier','flex')]:
            with self.assertRaises(ValueError):api.validate_response(dict(self.response,**{key:value}))
        bad=copy.deepcopy(self.response);bad['choices'][0]['finish_reason']='length'
        with self.assertRaises(ValueError):api.validate_response(bad)
        bad=copy.deepcopy(self.response);bad['choices'][0]['message']['content']='{"utterance":"x","extra":1}'
        with self.assertRaises(ValueError):api.validate_response(bad)

    def test_no_live_transport_or_key_reader(self):
        for name in ['urlopen','execute_one','read_key']:
            self.assertFalse(hasattr(api,name))

    def test_exclusive_write(self):
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)/'value.json';api.save_new(p,{'a':1})
            with self.assertRaises(FileExistsError):api.save_new(p,{'a':2})
            self.assertEqual(json.loads(p.read_text()),{'a':1})


class DataTests(unittest.TestCase):
    def test_primary_rows_and_no_labels_or_splits(self):
        rows=load_primary();self.assertEqual(len(rows),281)
        self.assertTrue(all(set(r)==set(FIELDS) for r in rows))
        self.assertEqual(len({r['id'] for r in rows}),281)

    def test_ignore_old_annotations(self):
        row=dict(load_primary()[0],split='test',function='invented',model_answer='gold')
        selected=select_rows([row])[0]
        self.assertNotIn('split',selected);self.assertNotIn('function',selected)
        self.assertNotIn('model_answer',selected)

    def test_invalid_provenance_and_duplicates(self):
        row=load_primary()[0]
        with self.assertRaises(ValueError):select_rows([row,row])
        with self.assertRaises(ValueError):select_rows([dict(row,url_or_doi='')])

    def test_inventory(self):
        info=inventory()
        self.assertEqual(info['candidate_rows'],7429)
        self.assertEqual(info['context_text_files'],73)
        self.assertFalse(info['final_split_created'])

    def test_package_imports(self):
        result=verify()
        self.assertTrue(result['all_hashes_match'])
        self.assertGreater(result['release_files_verified'],100)

if __name__=='__main__':unittest.main()
