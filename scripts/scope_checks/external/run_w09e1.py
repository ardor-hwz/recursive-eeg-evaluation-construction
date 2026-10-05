"""W09E-1 only: nested OOF -> clean development calibration -> outer proxies.

No headroom gate, scientific perturbation, outer updater replay or M1 endpoint.
Targets remain verbatim strings until an authorized development request decodes
them; the current outer participant has no numerical target API in this stage.
"""
from __future__ import annotations
import argparse, csv, hashlib, json, os, platform, sys, time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

for variable in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS'):
    os.environ.setdefault(variable,'1')
sys.dont_write_bytecode=True
import numpy as np
import scipy
from w09e1_core import weighted_ridge_fit, predict, clean_replay, choose_c0, cmean, validate_contract

CONTRACT_SHA='3af09b6d71dccf6403222967530099ec419e9569b883e0582983c808ce3c1008'
RT_SHA='71db65ce9d0218e53d45d689dffe86d8b840297a88971a4f4c8c45ae08338ee0'

class StageError(ValueError):pass
def require(condition,reason):
    if not condition:raise StageError(reason)
def sha(path):
    d=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):d.update(block)
    return d.hexdigest()
def object_sha(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def now():return datetime.now(timezone.utc).isoformat()
def csv_read(path):
    with Path(path).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def bool_value(value):
    require(value in ('True','False'), 'Invalid boolean ledger flag');return value=='True'
def csv_write(path,rows):
    require(bool(rows),'Empty output ledger');path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def seed_alias_manifest(array_digest,seeds):
    require(len(seeds)>0 and len(set(seeds))==len(seeds),'Seed identity mismatch')
    return {'deterministic':True,'independent_replications':False,
            'cells':[{'seed':s,'payload_sha256':array_digest,'alias_of_seed':seeds[0]} for s in seeds]}

class TargetFirewall:
    def __init__(self,records,development_ids,outer_id):
        self._records=records;self.development_ids=tuple(development_ids);self.outer_id=outer_id;self.access_log=[]
        require(len(set(self.development_ids))==len(self.development_ids) and outer_id not in self.development_ids,'Target firewall fold mismatch')
    def values(self,row_indices,allowed_ids,purpose):
        require(purpose in ('inner_fit','outer_fit','development_calibration'),'Target purpose forbidden in W09E-1')
        allowed=tuple(allowed_ids);require(len(set(allowed))==len(allowed) and bool(allowed),'Duplicate/empty target permission')
        require(set(allowed)<=set(self.development_ids) and self.outer_id not in allowed,'Outer or foreign targets forbidden')
        indices=[int(i) for i in row_indices];require(len(set(indices))==len(indices),'Duplicate target rows')
        selected=[self._records[i] for i in indices]
        actual={r['participant_id'] for r in selected}
        require(actual<=set(allowed) and all(r['target_observed'] for r in selected),'Target request crosses current fit/observed scope')
        values=[]
        for r in selected:
            source=r.get('opaque_target_row',r)
            rt=float(source['native_RT_seconds']);require(np.isfinite(rt) and rt>0,'Observed RT must be finite and positive')
            if 'response_sample' in source and 'deviation_sample' in source:
                measured=(int(source['response_sample'])-int(source['deviation_sample']))/500.0
                require(abs(measured-rt)<=1e-12,'Target equation/source alignment mismatch')
            values.append(rt)
        rt=np.asarray(values,dtype=np.float64)
        self.access_log.append({'purpose':purpose,'allowed_ids':list(allowed),'actual_decoded_ids':sorted(actual),
            'rows_count':len(indices),'row_indices_sha256':object_sha(indices),'outer_decoded':False})
        return {'log_rt':np.log(rt),'y':rt/(rt+1.0)}

class FoldStateMachine:
    def __init__(self,development_ids,outer_id):
        self.development_ids=tuple(development_ids);self.outer_id=outer_id;self.oof=set();self.state='OOF_BUILD';self.events=[]
        require(len(self.development_ids)==18 and len(set(self.development_ids))==18 and outer_id not in self.development_ids,'Exact18 development people required')
    def mark_oof(self,validation_id):
        require(self.state=='OOF_BUILD' and validation_id in self.development_ids and validation_id not in self.oof,'OOF membership/order/duplicate violation')
        self.oof.add(validation_id);self.events.append({'event':'OOF_PROXY_SEALED','validation_id':validation_id})
    def begin_calibration(self):
        require(self.state=='OOF_BUILD' and self.oof==set(self.development_ids),'Calibration requires full18 OOF proxies')
        self.state='CALIBRATION';self.events.append({'event':'DEVELOPMENT_CLEAN_CALIBRATION_STARTED'})
    def seal_comparators(self,digest):
        require(self.state=='CALIBRATION' and isinstance(digest,str) and len(digest)==64,'Comparator seal/order violation')
        self.state='COMPARATORS_SEALED';self.events.append({'event':'C0_CMEAN_SEALED','sha256':digest})
    def allow_outer_fit(self):
        require(self.state=='COMPARATORS_SEALED','Outer fitting must follow calibration/comparator seal');return True
    def seal_outer_models(self,digest):
        require(self.state=='COMPARATORS_SEALED' and isinstance(digest,str) and len(digest)==64,'Outer model seal/order violation')
        self.state='OUTER_MODELS_SEALED';self.events.append({'event':'OUTER_MODELS_SEALED','sha256':digest})
    def allow_outer_prediction(self):
        require(self.state=='OUTER_MODELS_SEALED','Outer prediction requires model/comparator seal');return True

def validate_fold(fold,c):
    ids=c['cohort']['group_ids'];outer=fold['outer_held_out'];dev=[p for p in ids if p!=outer]
    require(len(ids)==19 and outer in ids and fold['development_group_ids']==dev,'Outer18+1 membership mismatch')
    records={p:[s['recording_id'] for s in c['cohort']['sessions'] if s['group_id']==p] for p in ids}
    require(fold['outer_recording_ids']==records[outer],'Outer recordings mismatch')
    require([f['validation_group'] for f in fold['inner_folds']]==dev,'Inner18 OOF coverage mismatch')
    for inner in fold['inner_folds']:
        val=inner['validation_group'];require(inner['fit_group_ids']==[p for p in dev if p!=val],'Inner17+1 membership mismatch')
        require(inner['validation_recording_ids']==records[val],'Same-person recordings not jointly held out')
    return True

def source_guard(plan,c):
    require(sha(plan['contract_path'])==CONTRACT_SHA==plan['contract_sha256'],'Contract drift')
    for item in plan['source_files']+plan['code_files']:
        require(sha(item['path'])==item['sha256'],'Frozen source/code drift: '+item['path'])
    qa=read(plan['synthetic_QA_path']);require(qa['status']=='PASS' and qa['contract_sha256']==CONTRACT_SHA,'Production integration synthetic QA not accepted')
    require(qa['source_code_sha256']=={Path(i['path']).name:i['sha256'] for i in plan['code_files'] if i['role']=='production'},'Production QA/code binding mismatch')
    require(qa['qa_script_sha256']==next(i['sha256'] for i in plan['code_files'] if i['role']=='synthetic_QA'),'QA script binding mismatch')
    review=read(plan['prospective_review_path']);require(review['status']=='PASS' and review['contract_sha256']==CONTRACT_SHA,'Prospective independent review missing')
    require(review['source_code_sha256']=={Path(i['path']).name:i['sha256'] for i in plan['code_files'] if i['role']=='production'},'Prospective review/code binding mismatch')
    require(review['synthetic_QA_sha256']==sha(plan['synthetic_QA_path']),'Prospective review/QA binding mismatch')
    require(plan['authorization']['W09E_1_real_execution'] is True and plan['authorization']['W09E_2_plus'] is False,'Execution scope mismatch')
    return True

def load_inputs(plan,c):
    parent=Path(plan['feature_directory']);ids=c['cohort']['group_ids']
    index=csv_read(parent/'FEATURE_TRIAL_INDEX.csv');segments=csv_read(parent/'FEATURE_SEGMENT_LEDGER.csv')
    require(len(index)==len(segments)==27192,'Source metadata row count mismatch')
    n=len(index);fast=np.load(parent/'FAST_FEATURES.npy',mmap_mode='r',allow_pickle=False);slow=np.load(parent/'SLOW_FEATURES.npy',mmap_mode='r',allow_pickle=False)
    fm=np.load(parent/'FAST_VALID.npy',allow_pickle=False);sm=np.load(parent/'SLOW_VALID.npy',allow_pickle=False);jt=np.load(parent/'JOINT_INPUT_ELIGIBLE.npy',allow_pickle=False)
    require(fast.shape==slow.shape==(n,c['features']['dimensions_per_role']) and fast.dtype==slow.dtype==np.float64,'Feature shape/dtype mismatch')
    require(fm.shape==sm.shape==jt.shape==(n,) and fm.dtype==sm.dtype==jt.dtype==np.bool_,'Mask shape/dtype mismatch')
    valid=fm&sm;observed=np.array([bool_value(r['target_observed']) for r in index])
    require(np.array_equal(jt,valid&observed),'Legacy JOINT file must equal joint-target support')
    owner=np.array([r['participant_id'] for r in index]);selected=np.isin(owner,ids)
    records={s['recording_id']:s['group_id'] for s in c['cohort']['sessions']}
    for i,(a,b) in enumerate(zip(index,segments)):
        require(int(a['row_index'])==int(b['row_index'])==i,'Source row ordering mismatch')
        for name in ('participant_id','session_id','recording_id','original_trial_index','deviation_sample','target_observed'):
            require(a[name]==b[name],'Source identity alignment mismatch: '+name)
        require(bool_value(b['feature_input_valid'])==bool(valid[i]) and bool_value(b['joint_target_eligible'])==bool(jt[i]),'Source support flags mismatch')
        require(bool_value(a['fast_valid'])==bool(fm[i]) and bool_value(a['slow_valid'])==bool(sm[i]),'Feature index/mask mismatch')
        if selected[i]:require(records[a['recording_id']]==a['participant_id'],'Selected recording owner mismatch')
    take=np.flatnonzero(selected&valid);require(np.isfinite(fast[take]).all() and np.isfinite(slow[take]).all(),'Selected eligible features nonfinite')
    raw={}
    # Keep outcome fields opaque; do not numerically decode outside current fold.
    require(sha(plan['RT_ledger'])==RT_SHA,'Accepted RT ledger drift')
    with Path(plan['RT_ledger']).open(encoding='utf-8-sig',newline='') as f:
        for i,r in enumerate(csv.DictReader(f)):
            a=index[i];require(r['trial_id']==a['trial_id'] and r['group_id']==a['participant_id'] and r['recording_id']==a['recording_id'] and r['original_trial_index']==a['original_trial_index'] and r['deviation_sample']==a['deviation_sample'] and r['target_observed']==a['target_observed'],'RT/source trial alignment mismatch')
            if selected[i]:raw[i]={'participant_id':r['group_id'],'target_observed':observed[i].item(),
                                  'opaque_target_row':r}
    require(i+1==n,'RT/source trial length mismatch')
    grouped=defaultdict(list)
    for i in take:grouped[segments[i]['segment_id']].append(int(i))
    for sid,rows in grouped.items():
        require(bool(sid),'Valid input lacks segment id')
        first=segments[rows[0]];last=segments[rows[-1]]
        require(first['segment_position']=='0','Segment start lost')
        for pos,i in enumerate(rows):
            b=segments[i];require(int(b['segment_position'])==pos and b['recording_id']==first['recording_id'] and b['participant_id']==first['participant_id'] and int(b['original_trial_index'])==int(first['original_trial_index'])+pos,'Original segment continuity violation')
            require(bool_value(b['post_warmup'])==(pos>=c['operational_contract']['calibration']['C0']['postwarmup_indices_start']),'Warmup position mismatch')
    support={'selected_persons':len(ids),'selected_recordings':len(records),'selected_trial_rows':int(selected.sum()),
        'selected_feature_valid_rows':len(take),'selected_fit_target_rows':int((selected&jt).sum()),
        'selected_missing_target_input_rows':int((selected&valid&~observed).sum()),'selected_segments':len(grouped),
        'mask_semantics':'FAST_VALID & SLOW_VALID for full proxy/clean trajectory; legacy JOINT_INPUT_ELIGIBLE is additionally target_observed for fitting',
        'outside_cohort_numerical_RT_decoded':False,'raw_EEG_read':False,'features_reextracted':False}
    ledger=[];min_accuracy=validate_contract(c)['min_accuracy_rows']
    for recording in dict.fromkeys(r['recording_id'] for r in index):
        rows=np.asarray([i for i,r in enumerate(index) if r['recording_id']==recording],dtype=np.int64)
        person=index[int(rows[0])]['participant_id'];post=valid[rows]&np.asarray([bool_value(segments[int(i)]['post_warmup']) for i in rows])
        n_pair=int((post&observed[rows]).sum())
        ledger.append({'participant_id':person,'recording_id':recording,'in_fixed19':person in ids,'trial_rows':len(rows),
             'feature_valid_rows':int(valid[rows].sum()),'fit_target_rows':int(jt[rows].sum()),
             'postwarm_input_gain_rows':int(post.sum()),'postwarm_paired_target_rows':n_pair,
             'C0_accuracy_support':n_pair>=min_accuracy,'Cmean_gain_support':bool(post.sum()),
             'analysis_exclusion':'OUTSIDE_FIXED19' if person not in ids else 'NO_FEATURE_VALID_ROWS' if not valid[rows].any() else '',
             'future_headroom_status':'NOT_RUN','future_common_root_status':'NOT_RUN'})
    return {'fast':fast,'slow':slow,'index':index,'segments':segments,'valid':valid,'observed':observed,'owners':owner,
            'raw_targets':raw,'grouped_segments':dict(grouped),'support':support,'recording_ledger':ledger}

def save_model(path,model):
    keys=('mean','sd','zero_mask','beta','intercept')
    np.savez(path,**{k:model[k] for k in keys})
    return {'path':str(path),'sha256':sha(path),'metadata':model['metadata']}

def fit_pair(data,firewall,fit_ids,c,destination,purpose):
    rows=np.flatnonzero(data['valid']&data['observed']&np.isin(data['owners'],fit_ids))
    target=firewall.values(rows,fit_ids,purpose)
    ps=[data['index'][i]['participant_id'] for i in rows];rs=[data['index'][i]['recording_id'] for i in rows]
    require(set(ps)==set(fit_ids),'Every fit person needs eligible target rows')
    models={};receipts={};destination.mkdir(parents=True,exist_ok=True)
    for role in ('fast','slow'):
        model=weighted_ridge_fit(np.asarray(data[role][rows]),target['log_rt'],ps,rs,fit_ids,c,joint_feature_valid=np.ones(len(rows),dtype=bool))
        models[role]=model;receipts[role]=save_model(destination/(role+'_model.npz'),model)
    np.save(destination/'fit_row_indices.npy',rows)
    receipt={'fit_ids':list(fit_ids),'fit_ids_sha256':object_sha(list(fit_ids)),'fit_rows':len(rows),
       'fit_row_indices':{'path':str(destination/'fit_row_indices.npy'),'sha256':sha(destination/'fit_row_indices.npy')},
       'fit_person_counts':dict(Counter(ps)),'fit_recording_counts':dict(Counter(rs)),
       'normalizer_and_ridge_same_fit_rows':True,'role_shared_fit_support':True,'models':receipts,'target_purpose':purpose,
       'only_current_fit_target_ids_decoded':True}
    write(destination/'FIT_RECEIPT.json',receipt)
    return models,receipt

def proxy_rows(data,person,models,c,destination,seeds):
    rows=np.flatnonzero(data['valid']&(data['owners']==person));require(len(rows)>0,'Held-out input support absent')
    fast=predict(models['fast'],np.asarray(data['fast'][rows]));slow=predict(models['slow'],np.asarray(data['slow'][rows]))
    require(fast.shape==slow.shape==(len(rows),) and np.isfinite(fast).all() and np.isfinite(slow).all(),'Nonfinite held-out proxies')
    np.savez(destination/'PROXIES.npz',row_indices=rows,fast=fast,slow=slow)
    identity=[]
    for i in rows:
        a=data['index'][i];b=data['segments'][i]
        identity.append({k:a[k] for k in ('row_index','participant_id','session_id','recording_id','trial_id','original_trial_index','deviation_sample','target_observed')}|
             {k:b[k] for k in ('segment_id','segment_position','warmup','post_warmup','post_warmup_target_eligible')})
    csv_write(destination/'PROXY_INDEX.csv',identity)
    digest=sha(destination/'PROXIES.npz');aliases=seed_alias_manifest(digest,seeds)
    receipt={'participant_id':person,'rows':len(rows),'full_feature_valid_input_support':True,'no_target_condition_for_predictions':True,
       'targets_numeric_read_to_predict':False,'proxy_file':str(destination/'PROXIES.npz'),'proxy_sha256':digest,
       'index_sha256':sha(destination/'PROXY_INDEX.csv'),'seed_aliases':aliases}
    write(destination/'PROXY_RECEIPT.json',receipt)
    return {'rows':rows,'fast':fast,'slow':slow,'receipt':receipt}

def calibration_entries(data,firewall,dev,oof,c,folder):
    entries=[];trace_index=[];seeds=c['split']['development_oof_seeds'];position=c['operational_contract']['calibration']['C0']['postwarmup_indices_start']
    for person in dev:
        q=oof[person];lookup={int(row):j for j,row in enumerate(q['rows'])}
        for sid,rows0 in data['grouped_segments'].items():
            b=data['segments'][rows0[0]]
            if b['participant_id']!=person:continue
            rows=np.asarray(rows0,dtype=np.int64);local=np.asarray([lookup[int(i)] for i in rows],dtype=np.int64)
            fast=q['fast'][local];slow=q['slow'][local];post=np.arange(len(rows))>=position
            observed=data['observed'][rows];target_y=np.full(len(rows),np.nan,dtype=np.float64)
            wanted=rows[post&observed]
            if len(wanted):target_y[post&observed]=firewall.values(wanted,dev,'development_calibration')['y']
            replay=clean_replay(fast,slow,c)
            payload={'row_indices':rows,'fast':fast,'slow':slow,'postwarm_mask':post,'target_observed':observed,'target_y':target_y,
                     'segment_id':np.asarray(sid),'participant_id':np.asarray(person),'recording_id':np.asarray(b['recording_id'])}
            for updater in c['updaters']:
                for field,value in replay[updater].items():
                    if isinstance(value,np.ndarray):payload[updater+'_'+field]=value
            destination=folder/'clean_traces'/(hashlib.sha256(sid.encode()).hexdigest()+'.npz');destination.parent.mkdir(parents=True,exist_ok=True)
            np.savez(destination,**payload)
            trace_index.append({'participant_id':person,'recording_id':b['recording_id'],'segment_id':sid,
                 'trace_file':destination.relative_to(folder).as_posix(),'rows':len(rows),'postwarm_rows':int(post.sum()),
                 'postwarm_paired_rows':int((post&observed).sum()),'trace_sha256':sha(destination),
                 'development_seeds':json.dumps(seeds),'seeds_are_aliases':True,
                 'input_sha256':replay['metadata']['input_sha256'],
                 'output_sha256':json.dumps(replay['metadata']['output_sha256'],sort_keys=True)})
            for seed in seeds:
                entries.append({'participant_id':person,'recording_id':b['recording_id'],'segment_id':sid,'seed':seed,
                   'fast':fast,'slow':slow,'postwarm_mask':post,'target_y':target_y,'replay':replay,'development_ids':list(dev)})
    csv_write(folder/'CALIBRATION_TRACE_INDEX.csv',trace_index)
    return entries

def run(root):
    root=Path(root).resolve();plan=read(root/'W09E_1_EXECUTION_PLAN.json');c=read(plan['contract_path']);source_guard(plan,c)
    require(not (root/'W09E_1_EXECUTION_RESULT.json').exists(),'Scientific execution already has a result; do not overwrite')
    require(not (root/'folds').exists(),'Do not silently resume/overwrite any scientific fold outputs')
    write(root/'W09E_1_RUN_STARTED.json',{'started_at_utc':now(),'contract_sha256':CONTRACT_SHA,'plan_sha256':sha(root/'W09E_1_EXECUTION_PLAN.json'),'scope':'W09E-1_ONLY'})
    data=load_inputs(plan,c);write(root/'INPUT_ALIGNMENT_RECEIPT.json',{'status':'PASS',**data['support']})
    csv_write(root/'RECORDING_ROLE_SUPPORT.csv',data['recording_ledger'])
    splits=read(c['split']['assignments_path']);require(sha(c['split']['assignments_path'])==c['split']['assignments_sha256'],'Split drift')
    fit_receipts=[];fold_summaries=[];auditlog=[];start=time.monotonic()
    for number,fold in enumerate(splits['folds'],1):
        validate_fold(fold,c);outer=fold['outer_held_out'];dev=fold['development_group_ids'];directory=root/'folds'/outer;directory.mkdir(parents=True)
        firewall=TargetFirewall(data['raw_targets'],dev,outer);chain=FoldStateMachine(dev,outer);oof={}
        for inner in fold['inner_folds']:
            val=inner['validation_group'];destination=directory/'inner'/val
            models,fit=fit_pair(data,firewall,inner['fit_group_ids'],c,destination,'inner_fit')
            require(val not in fit['fit_ids'] and outer not in fit['fit_ids'],'Inner target/model leakage')
            proxy=proxy_rows(data,val,models,c,destination,c['split']['development_oof_seeds']);oof[val]=proxy
            write(destination/'INNER_FOLD_RECEIPT.json',{'status':'PASS','outer_held_out':outer,'validation_group':val,'fit_ids':inner['fit_group_ids'],
                'fit_receipt_sha256':sha(destination/'FIT_RECEIPT.json'),'proxy_receipt_sha256':sha(destination/'PROXY_RECEIPT.json'),
                'same_person_all_recordings_held_out':True,'validation_targets_used_in_fit':False,'outer_targets_used':False})
            fit_receipts.append({'outer':outer,'kind':'inner','validation':val,'fit_receipt':str(destination/'FIT_RECEIPT.json'),'sha256':sha(destination/'FIT_RECEIPT.json')})
            chain.mark_oof(val)
        chain.begin_calibration();entries=calibration_entries(data,firewall,dev,oof,c,directory)
        chosen=choose_c0(entries,c,expected_participants=dev);mean=cmean(entries,c,expected_participants=dev)
        selection={'status':'PASS','outer_held_out':outer,'development_ids':dev,'C0':chosen,'Cmean':mean,'outer_targets_accessed':False,
            'development_only_clean_replay':True,'perturbations_run':False,'scientific_M1_generated':False,
            'OOF_seals':[{'participant_id':p,'sha256':oof[p]['receipt']['proxy_sha256']} for p in dev],
            'updater_source_sha256':c['implementation_contract']['source_code_sha256']}
        write(directory/'COMPARATOR_SELECTION.json',selection);chain.seal_comparators(sha(directory/'COMPARATOR_SELECTION.json'))
        chain.allow_outer_fit();models,fit=fit_pair(data,firewall,dev,c,directory/'outer','outer_fit')
        require(outer not in fit['fit_ids'],'Outer fit leakage')
        seal={'outer_held_out':outer,'development_ids':dev,'contract_sha256':CONTRACT_SHA,'fit_receipt_sha256':sha(directory/'outer/FIT_RECEIPT.json'),
            'comparator_selection_sha256':sha(directory/'COMPARATOR_SELECTION.json'),'outer_target_numbers_accessed':False,
            'models':{role:fit['models'][role]['sha256'] for role in models}}
        write(directory/'OUTER_FIT_COMPARATOR_SEAL.json',seal);chain.seal_outer_models(sha(directory/'OUTER_FIT_COMPARATOR_SEAL.json'))
        chain.allow_outer_prediction();proxy=proxy_rows(data,outer,models,c,directory/'outer',c['split']['final_seeds'])
        require(all(outer not in item['actual_decoded_ids'] for item in firewall.access_log),'Firewall audit found outer targets')
        write(directory/'TARGET_ACCESS_AUDIT.json',{'outer_held_out':outer,'status':'PASS','no_current_outer_RT_decoded':True,'requests':firewall.access_log,
            'cross_LOSO_note':'A person is development data in other outer folds; restriction is fold-local, not a permanent global quarantine'})
        foldresult={'status':'PASS','outer_held_out':outer,'inner_folds':len(oof),'development_ids':dev,'C0':chosen['alpha'],
          'Cmean':{u:mean[u]['alpha'] for u in c['updaters']},'outer_proxy_rows':proxy['receipt']['rows'],
          'outer_proxy_sha256':proxy['receipt']['proxy_sha256'],'models_and_comparators_sealed_before_outer_prediction':True,
          'outer_labels_numerical_accessed':False,'outer_clean_updater_run':False,'stage2_gates_run':False,'state_chain':chain.events}
        write(directory/'OUTER_FOLD_RECEIPT.json',foldresult);fold_summaries.append(foldresult)
        fit_receipts.append({'outer':outer,'kind':'outer','fit_receipt':str(directory/'outer/FIT_RECEIPT.json'),'sha256':sha(directory/'outer/FIT_RECEIPT.json')})
        auditlog.append({'outer':outer,'target_access_receipt_sha256':sha(directory/'TARGET_ACCESS_AUDIT.json')})
        print(json.dumps({'progress':'outer_fold_completed','outer':outer,'completed':number,'total':19,'elapsed_s':round(time.monotonic()-start,1)}),flush=True)
    result={'schema':'paper2.w09e1.execution-result.v1','status':'COMPLETED_PENDING_INDEPENDENT_ACCEPTANCE','completed_at_utc':now(),
      'contract_sha256':CONTRACT_SHA,'plan_sha256':sha(root/'W09E_1_EXECUTION_PLAN.json'),
      'chain':['19_person_nested_LOSO','development_OOF_proxies','development_only_clean_calibration_replay','C0_Cmean_sealed','outer_held_out_fast_slow_proxies'],
      'outer_folds':len(fold_summaries),'inner_folds':19*18,'dual_proxy_fit_pairs':19*19,'individual_ridge_fits':2*19*19,
      'fold_summaries':fold_summaries,'fit_receipts':fit_receipts,'target_audit':auditlog,'input_support':data['support'],
      'stage2_headroom_run':False,'stage2_common_root_run':False,'outer_updater_or_perturbation_replay_run':False,'scientific_M1_or_CI_generated':False,
      'actual_headroom_count':None,'actual_common_count':None,'runtime':{'python_executable':sys.executable,'python_version':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__}}
    write(root/'W09E_1_EXECUTION_RESULT.json',result);print(json.dumps({'status':result['status'],'outer_folds':19,'inner_folds':342,'individual_ridge_fits':722}),flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);args=p.parse_args()
    try:run(args.root)
    except Exception as exc:
        write(args.root/'W09E_1_ERROR_RECEIPT.json',{'status':'HALT','created_at_utc':now(),'error_type':type(exc).__name__,'reason':str(exc),
             'no_participant_deletion_or_scientific_rule_change':True,'automatic_replay_or_resume':False})
        raise
if __name__=='__main__':main()
