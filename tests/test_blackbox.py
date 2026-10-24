import copy
import unittest
import blackbox as b

class Tests(unittest.TestCase):
    def setUp(self):
        self.req = {'id':'r','text':'Applies to 12 V devices'}
        self.ctx = [{'id':'s','text':'Applies to 12 V devices'}]
        self.conditions = [{'id':'c','text':'Supply voltage is 12 V','field':'voltage','op':'eq','value':12,'unit':'V'}]
        self.product = {'passages':[{'id':'p','text':'Rated 12 V'}, {'id':'other','text':'Housing red'}],
                        'facts':[{'id':'f','field':'voltage','value':12,'unit':'V','status':'explicit','evidence_ids':['p']}]}
        self.result = {'decision':'applicable','reason':'Voltage matches scope','regulatory_evidence_ids':['s'],
                       'product_evidence_ids':['p'],'missing_information':[],
                       'condition_results':[{'condition_id':'c','status':'satisfied','reason':'12 V','product_evidence_ids':['p']}]}
    def run_mode(self, mode, **kw):
        self.payload = None
        def fake(system,payload,schema):
            self.payload=payload
            return copy.deepcopy(self.result)
        return b.BlackBox(mode,llm=fake,**kw).evaluate(self.req,self.ctx,self.product,self.conditions)
    def test_all_variants(self):
        for mode in ['direct_gpt','retrieval_rules','bert']:
            kwargs = {'reranker':lambda q,p:[1.0]*len(p)} if mode=='bert' else {}
            self.assertEqual(self.run_mode(mode,**kwargs)['decision'],'applicable')
            self.assertEqual(bool(self.payload['rule_checks']),mode!='direct_gpt')
    def test_disagreement(self):
        self.result['condition_results'][0]['status']='contradicted'
        self.assertEqual(self.run_mode('retrieval_rules')['decision'],'unresolved')
    def test_bad_citation(self):
        self.result['product_evidence_ids']=['invented']
        with self.assertRaises(ValueError): self.run_mode('direct_gpt')
    def test_missing_citation(self):
        self.result['product_evidence_ids']=[]
        self.assertEqual(self.run_mode('direct_gpt')['decision'],'unresolved')
    def test_missing_condition(self):
        self.result['condition_results']=[]
        with self.assertRaises(ValueError): self.run_mode('direct_gpt')
    def test_unknown_conflict_units(self):
        for change in [{'status':'inferred'}, {'unit':'mV'}, {'evidence_ids':[]}]:
            f={**self.product['facts'][0],**change}
            self.assertEqual(b.rule_checks(self.conditions,[f],{'p'})[0]['status'],'unknown')
        f={**self.product['facts'][0],'value':24}
        self.assertEqual(b.rule_checks(self.conditions,self.product['facts']+[f],{'p'})[0]['status'],'unknown')
    def test_pinned_evidence(self):
        self.run_mode('bert',top_k=1,candidate_k=2,reranker=lambda q,p:[0,10])
        self.assertIn('p',[p['id'] for p in self.payload['product']['passages']])
    def test_budget(self):
        with self.assertRaises(ValueError): self.run_mode('direct_gpt',max_input_chars=10)
    def test_bert_required(self):
        with self.assertRaises(ValueError): b.BlackBox('bert')
    def test_semantic_and_invalid_scores(self):
        result=self.run_mode('retrieval_rules',semantic=lambda q,p:[0,1])
        self.assertEqual(result['decision'],'applicable')
        with self.assertRaises(ValueError): self.run_mode('bert',reranker=lambda q,p:[float('nan')]*len(p))
    def test_gpt_adapter(self):
        from types import SimpleNamespace
        response=SimpleNamespace(status='completed',output_text=__import__('json').dumps(self.result))
        client=SimpleNamespace(responses=SimpleNamespace(create=lambda **kwargs:response))
        self.assertEqual(b.gpt_llm(model='explicit-model',client=client)('s',{},b.SCHEMA), self.result)
        response.status='incomplete'
        with self.assertRaises(RuntimeError): b.gpt_llm(model='m',client=client)('s',{},b.SCHEMA)

if __name__=='__main__': unittest.main()
