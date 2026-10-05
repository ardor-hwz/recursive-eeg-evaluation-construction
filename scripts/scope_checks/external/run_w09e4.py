"""W09E-4 complete sealed-trajectory reduction and conditional participant BCa."""
from __future__ import annotations
import argparse,csv,hashlib,importlib.util,json,os,sys,time
from collections import defaultdict
from datetime import datetime,timezone
from pathlib import Path
for name in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS'):os.environ[name]='1'
sys.dont_write_bytecode=True
import numpy as np
import w09e4_core as core
LOCK='3af09b6d71dccf6403222967530099ec419e9569b883e0582983c808ce3c1008'
RT_SHA='71db65ce9d0218e53d45d689dffe86d8b840297a88971a4f4c8c45ae08338ee0'
def require(ok,reason):
    if not ok:raise ValueError(reason)
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def write(path,r):Path(path).write_text(json.dumps(r,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def csv_read(path):
    with Path(path).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def csv_write(path,rows):
    require(bool(rows),'Empty product ledger '+str(path))
    with Path(path).open('w',encoding='utf-8',newline='') as f:
        fields=list(dict.fromkeys(k for r in rows for k in r))
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
def array_hash(*arrays):
    h=hashlib.sha256()
    for v in arrays:
        x=np.ascontiguousarray(v);h.update(str(x.dtype).encode());h.update(json.dumps(list(x.shape)).encode());h.update(x.tobytes())
    return h.hexdigest()
def load_module(name,path,digest):
    raw=Path(path).read_bytes();require(hashlib.sha256(raw).hexdigest()==digest,'Source helper drift '+str(path))
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m
    exec(compile(raw,str(path),'exec'),m.__dict__);return m
def source_guard(root):
    p=read(root/'W09E_4_EXECUTION_PLAN.json');require(sha(p['contract_path'])==p['contract_sha256']==LOCK,'Contract drift')
    for x in p['source_files']+p['code_files']:require(sha(x['path'])==x['sha256'],'Protected source/code drift '+x['path'])
    q=read(root/'W09E_4_SYNTHETIC_QA.json');r=read(root/'W09E_4_PROSPECTIVE_ACCEPTANCE.json')
    require(q['status']==r['status']=='PASS','Prospective QA/review required')
    codes={n:sha(root/'src'/n) for n in ('w09e4_core.py','run_w09e4.py')}
    require(q['source_code_sha256']==r['source_code_sha256']==codes,'Prospective code changed')
    require(q['contract_sha256']==r['contract_sha256']==LOCK,'Prospective scientific binding changed')
    require(r['synthetic_QA_sha256']==sha(root/'W09E_4_SYNTHETIC_QA.json'),'QA receipt drift')
    require(r['execution_plan_sha256']==sha(root/'W09E_4_EXECUTION_PLAN.json'),'Plan receipt drift')
    require(p['authorization']['W09E_4_complete'] and not p['authorization']['W09E_5_plus'],'Stage4 authorization mismatch')
    a=read(Path(p['stage3_directory'])/'W09E_3_ACCEPTANCE.json');require(a['status']=='PASS' and a['fixed_N']==19 and a['common_roots']==4018,'Accepted complete W09E3 required')
    return p,read(p['contract_path'])
def support_masks(rows,contract):
    """Source metadata only: postwarm target rows and strictly original-adjacent pairs."""
    warm=contract['operational_contract']['calibration']['C0']['postwarmup_indices_start']
    post=np.array([int(r['segment_position'])>=warm for r in rows],dtype=bool)
    require(all((r['post_warmup']=='True')==bool(v) for r,v in zip(rows,post)),'Warmup ledger drift')
    observed=np.array([r['target_observed']=='True' for r in rows],dtype=bool)
    pairs=np.zeros(len(rows),dtype=bool)
    for k in range(1,len(rows)):
        a,b=rows[k-1],rows[k]
        pairs[k]=post[k-1] and post[k] and a['recording_id']==b['recording_id'] and a['segment_id']==b['segment_id'] and int(b['segment_position'])==int(a['segment_position'])+1 and int(b['original_trial_index'])==int(a['original_trial_index'])+1
    require(all((r['post_warmup_target_eligible']=='True')==bool(v) for r,v in zip(rows,post&observed)),'Paired target support drift')
    require(all((r['adjacent_post_warmup_pair']=='True')==bool(v) for r,v in zip(rows,pairs)),'Adjacent pair support drift')
    return post&observed,pairs
def decode_targets(opaque,index,selected_rows,owner,seal_verified):
    require(seal_verified,'Outer numeric RT requires pre-existing comparator/model seal')
    ids=np.asarray(selected_rows,dtype=np.int64);require(len(set(ids.tolist()))==len(ids),'Duplicate evaluation target row')
    y=[]
    for i in ids:
        r=opaque[int(i)];a=index[int(i)]
        require(r['group_id']==a['participant_id']==owner and r['target_observed']==a['target_observed']=='True','Numeric target scope violation')
        for field in ('recording_id','original_trial_index','deviation_sample','trial_id'):require(r[field]==a[field],'Target original identity mismatch')
        rt=float(r['native_RT_seconds']);measured=(int(r['response_sample'])-int(r['deviation_sample']))/500.0
        require(np.isfinite(rt) and rt>0 and abs(rt-measured)<=1e-12,'Invalid measured positive RT')
        y.append(rt/(rt+1.0))
    return np.asarray(y,dtype=np.float64)
def auxiliary(root,plan,c):
    one=Path(plan['stage1_directory']);feature=Path(plan['feature_directory'])
    m=load_module('_w09e4_bound_clean',one/'src/w09e1_core.py',plan['stage1_helper_sha256'])
    source_index=csv_read(feature/'FEATURE_TRIAL_INDEX.csv');source_segments=csv_read(feature/'FEATURE_SEGMENT_LEDGER.csv')
    require(sha(plan['RT_ledger'])==RT_SHA,'RT ledger drift before evaluation')
    # Outcome strings are kept opaque until each outer fold's seal and clean products are checked.
    opaque=csv_read(plan['RT_ledger']);require(len(opaque)==len(source_index)==len(source_segments)==27192,'Full source row count mismatch')
    for a,b in zip(source_index,source_segments):
        for k in ('row_index','participant_id','recording_id','original_trial_index','deviation_sample','target_observed'):require(a[k]==b[k],'Feature metadata lineage mismatch')
    ids=c['cohort']['group_ids'];seeds=c['split']['final_seeds'];states_names=['heuristic','adaptive_kalman','C0','Cmean_heuristic','Cmean_adaptive_kalman']
    recording_support={};entries=[];clean_index=[];targets=[];allrecordingmetrics=[];segments_count=0;valid_rows=0
    (root/'auxiliary_clean').mkdir(exist_ok=True)
    for person in ids:
        d=one/'folds'/person;sel=read(d/'COMPARATOR_SELECTION.json');seal=read(d/'OUTER_FIT_COMPARATOR_SEAL.json');pr=read(d/'outer/PROXY_RECEIPT.json')
        require(seal['outer_held_out']==sel['outer_held_out']==person and not seal['outer_target_numbers_accessed'],'Outer fold/seal mismatch')
        require(seal['comparator_selection_sha256']==sha(d/'COMPARATOR_SELECTION.json') and seal['fit_receipt_sha256']==sha(d/'outer/FIT_RECEIPT.json'),'Comparator/model receipt drift')
        for role,h in seal['models'].items():require(sha(d/'outer'/(role+'_model.npz'))==h,'Final model drift')
        require(pr['proxy_sha256']==sha(d/'outer/PROXIES.npz') and pr['index_sha256']==sha(d/'outer/PROXY_INDEX.csv'),'Outer proxy drift')
        require([x['seed'] for x in pr['seed_aliases']['cells']]==seeds and all(x['payload_sha256']==pr['proxy_sha256'] for x in pr['seed_aliases']['cells']),'Final deterministic seed aliases changed')
        rows=csv_read(d/'outer/PROXY_INDEX.csv')
        with np.load(d/'outer/PROXIES.npz',allow_pickle=False) as z:rowids=z['row_indices'].copy();f=z['fast'].copy();s=z['slow'].copy()
        require(np.array_equal(rowids,np.array([int(r['row_index']) for r in rows])) and len(rows)==len(f)==len(s),'Proxy row alignment drift')
        expected=np.array([int(r['row_index']) for r in source_segments if r['participant_id']==person and r['feature_input_valid']=='True'])
        require(np.array_equal(rowids,expected),'Full feature-valid outer trace required')
        for r in rows:
            b=source_segments[int(r['row_index'])]
            for k in ('participant_id','recording_id','segment_id','segment_position','original_trial_index'):require(r[k]==b[k],'Proxy/source segment lineage drift')
        source_rows=[source_segments[int(i)] for i in rowids];accuracy_mask,pair_mask=support_masks(source_rows,c)
        arrays={name:np.empty(len(rows),dtype=np.float64) for name in states_names};extra={};seg_receipts=[]
        grouped=defaultdict(list)
        for k,r in enumerate(rows):grouped[r['segment_id']].append(k)
        for sid,pos in grouped.items():
            ix=np.asarray(pos,dtype=np.int64);sr=[source_rows[k] for k in pos]
            require(np.array_equal(ix,np.arange(ix[0],ix[0]+len(ix))) and [int(r['segment_position']) for r in sr]==list(range(len(ix))),'Segment positions/reset incomplete')
            require(len({r['recording_id'] for r in sr})==1 and np.all(np.diff([int(r['original_trial_index']) for r in sr])==1),'Original discontinuity compressed')
            replay=m.clean_replay(f[ix],s[ix],c)
            for name in ('heuristic','adaptive_kalman'):
                arrays[name][ix]=replay[name]['state']
                for field,value in replay[name].items():
                    if isinstance(value,np.ndarray):
                        key=name+'__'+field
                        if key not in extra:extra[key]=np.empty((len(rows),)+value.shape[1:],dtype=value.dtype)
                        extra[key][ix]=value
            arrays['C0'][ix]=m.constant_replay(f[ix],s[ix],sel['C0']['alpha'])
            for u in ('heuristic','adaptive_kalman'):arrays['Cmean_'+u][ix]=m.constant_replay(f[ix],s[ix],sel['Cmean'][u]['alpha'])
            seg_receipts.append({'segment_id':sid,'recording_id':sr[0]['recording_id'],'first_position':int(ix[0]),'updates':len(ix),'source_metadata':replay['metadata']})
        require(all(np.isfinite(v).all() for v in arrays.values()),'Nonfinite auxiliary states; no row deletion')
        recording_order=list(dict.fromkeys(r['recording_id'] for r in rows));eligible_accuracy=np.zeros(len(rows),dtype=bool)
        for rec in recording_order:
            take=np.array([r['recording_id']==rec for r in rows]);n=int((accuracy_mask&take).sum());pairs=int((pair_mask&take).sum())
            recording_support[rec]={'participant_id':person,'feature_valid_rows':int(take.sum()),'postwarm_paired_target_rows':n,'adjacent_postwarm_pairs':pairs,'accuracy_eligible':n>=2,'MAFD_eligible':pairs>=1}
            if n>=2:eligible_accuracy|=accuracy_mask&take
        # Clean trajectories and fold-local gains are completely fixed before this evaluation-only decode.
        y_full=np.full(len(rows),np.nan);target_rows=rowids[accuracy_mask]
        y=decode_targets(opaque,source_index,target_rows,person,True);y_full[accuracy_mask]=y
        for seed in seeds:
            for sid,pos in grouped.items():
                ix=np.asarray(pos,dtype=np.int64);sr=[source_rows[k] for k in pos]
                entries.append({'participant_id':person,'recording_id':sr[0]['recording_id'],'segment_id':sid,'seed':seed,'original_indices':np.array([int(r['original_trial_index']) for r in sr],dtype=np.int64),'target_y':y_full[ix],'postwarm_mask':np.arange(len(ix))>=c['operational_contract']['calibration']['C0']['postwarmup_indices_start'],'states':{k:v[ix] for k,v in arrays.items()}})
            for rec in recording_order:
                take=np.array([r['recording_id']==rec for r in rows]);mask=eligible_accuracy&take;pair=pair_mask&take
                for name,v in arrays.items():allrecordingmetrics.append({'participant_id':person,'recording_id':rec,'seed':seed,'state':name,'paired_rows':int(mask.sum()),'supported_pairs':int(pair.sum()),'squared_error_sum':float(np.sum((v[mask]-y_full[mask])**2)) if mask.any() else 0.0,'absolute_difference_sum':float(np.sum(np.abs((v[1:]-v[:-1])[pair[1:]]))) if pair.any() else 0.0})
        relative='auxiliary_clean/'+person+'.npz';metadata={'participant_id':person,'contract_sha256':LOCK,'proxy_sha256':pr['proxy_sha256'],'comparator_selection_sha256':sha(d/'COMPARATOR_SELECTION.json'),'outer_fit_comparator_seal_sha256':sha(d/'OUTER_FIT_COMPARATOR_SEAL.json'),'C0':sel['C0']['alpha'],'Cmean':{u:sel['Cmean'][u]['alpha'] for u in ('heuristic','adaptive_kalman')},'segments':seg_receipts,'numeric_RT_use':'EVALUATION_ONLY_AFTER_FINAL_MODELS_COMPARATORS_SEALED','seed_aliases':pr['seed_aliases']}
        payload={'row_indices':rowids,'fast':f,'slow':s,'accuracy_mask':eligible_accuracy,'pair_mask':pair_mask,'target_row_indices':target_rows,'represented_y':y,**{'state__'+k:v for k,v in arrays.items()},**extra,'metadata_json':np.asarray(json.dumps(metadata,sort_keys=True,separators=(',',':'),allow_nan=False))}
        with (root/relative).open('xb') as stream:np.savez_compressed(stream,**payload)
        clean_index.append({'participant_id':person,'payload_file':relative,'payload_sha256':sha(root/relative),'feature_valid_rows':len(rows),'segments':len(grouped),'paired_target_rows':len(y),'supported_pairs':int(pair_mask.sum()),'outer_fit_comparator_seal_sha256':sha(d/'OUTER_FIT_COMPARATOR_SEAL.json'),'comparator_selection_sha256':sha(d/'COMPARATOR_SELECTION.json')})
        targets.append({'participant_id':person,'purpose':'outer_held_out_auxiliary_evaluation_only','decoded_rows':len(y),'row_indices_sha256':array_hash(target_rows),'represented_y_sha256':array_hash(y),'model_comparator_seal_verified_before_decode':True,'refit_or_selection':False})
        segments_count+=len(grouped);valid_rows+=len(rows)
        print(json.dumps({'progress':'auxiliary_clean_participant','participant':person,'rows':len(rows),'segments':len(grouped),'target_rows':len(y)}),flush=True)
    ledger=[];root_counts=defaultdict(int)
    for r in csv_read(Path(plan['stage2_directory'])/'COMMON_ROOTS.csv'):root_counts[r['recording_id']]+=1
    for rec in dict.fromkeys(r['recording_id'] for r in source_index):
        source=[r for r in source_segments if r['recording_id']==rec];person=source[0]['participant_id']
        v=recording_support.get(rec,{'participant_id':person,'feature_valid_rows':sum(r['feature_input_valid']=='True' for r in source),'postwarm_paired_target_rows':sum(r['post_warmup_target_eligible']=='True' for r in source),'adjacent_postwarm_pairs':sum(r['adjacent_post_warmup_pair']=='True' for r in source),'accuracy_eligible':False,'MAFD_eligible':False})
        ledger.append({'recording_id':rec,**v,'original_trial_rows':len(source),'feature_valid_segments':len({r['segment_id'] for r in source if r['feature_input_valid']=='True'}),'in_fixed19':person in ids,'M1_roots':root_counts[rec],'M1_eligible':root_counts[rec]>0,'accuracy_reason':'OUTSIDE_FIXED19' if person not in ids else '' if v['accuracy_eligible'] else 'FEWER_THAN_2_POSTWARM_PAIRED_TARGET_ROWS','MAFD_reason':'OUTSIDE_FIXED19' if person not in ids else '' if v['MAFD_eligible'] else 'NO_CONSECUTIVE_POSTWARM_PAIR','M1_reason':'OUTSIDE_FIXED19' if person not in ids else '' if root_counts[rec] else 'NO_COMMON_ROOT'})
    reduced=core.auxiliary_metrics(entries,c,[{'participant_id':x['group_id'],'recording_id':x['recording_id']} for x in c['cohort']['sessions']])
    write(root/'AUXILIARY_ENDPOINT_STATUS.json',{'status':'PASS' if all(x['status']=='EVALUABLE' for x in reduced['endpoint_status']) else 'NON_EVALUABLE_COMPLETE19','endpoints':reduced['endpoint_status']})
    require(all(x['status']=='EVALUABLE' for x in reduced['endpoint_status']),'Auxiliary endpoint NON_EVALUABLE; exact19 retained in AUXILIARY_ENDPOINT_STATUS.json')
    csv_write(root/'AUXILIARY_CLEAN_INDEX.csv',clean_index);csv_write(root/'RECORDING_ENDPOINT_SUPPORT.csv',ledger);csv_write(root/'AUXILIARY_RECORDING_COMPONENTS.csv',allrecordingmetrics)
    csv_write(root/'AUXILIARY_PARTICIPANT_METRICS.csv',reduced['participant_rows']);csv_write(root/'AUXILIARY_PARTICIPANT_SEED_METRICS.csv',reduced['person_seed_rows']);csv_write(root/'AUXILIARY_RECORDING_METRICS.csv',reduced['recording_rows'])
    write(root/'OUTER_TARGET_EVALUATION_RECEIPT.json',{'status':'PASS','contract_sha256':LOCK,'RT_ledger_sha256':RT_SHA,'accesses':targets,'numeric_outside_fixed19_decoded':False,'native_RT_values_redistributed':False,'model_refit_comparator_reselection':False})
    return reduced['endpoint_rows'],{'segments':segments_count,'feature_valid_rows':valid_rows,'paired_target_rows':sum(x['decoded_rows'] for x in targets),'supported_pairs':sum(x['supported_pairs'] for x in clean_index),'accuracy_recordings':sum(v['accuracy_eligible'] for v in recording_support.values()),'MAFD_recordings':sum(v['MAFD_eligible'] for v in recording_support.values())}
def run(root):
    require(not (root/'W09E_4_RUN_STARTED.json').exists(),'Do not overwrite/resume stage4 scientific execution')
    plan,c=source_guard(root);core.validate_contract(c);start=time.monotonic()
    write(root/'W09E_4_RUN_STARTED.json',{'at_utc':datetime.now(timezone.utc).isoformat(),'contract_sha256':LOCK,'plan_sha256':sha(root/'W09E_4_EXECUTION_PLAN.json'),'scope':'COMPLETE_W09E4_ONLY'})
    three=Path(plan['stage3_directory']);frozen=csv_read(Path(plan['stage2_directory'])/'COMMON_ROOTS.csv');index=csv_read(three/'ROOT_REPLAY_INDEX.csv')
    require(len(index)==len(frozen)==4018 and len({x['root_id'] for x in index})==4018,'Root count/duplicates drift')
    byroot={x['root_id']:x for x in index};require(set(byroot)=={x['root_id'] for x in frozen},'Frozen root membership drift')
    primitive=[];required=c['operational_contract']['root_gates']['common']['required_keys'];expected={(u,v,sign,seed,shape) for u in required['updaters'] for v in required['views'] for sign in required['signs'] for seed in required['seeds'] for shape in required['shapes']}
    for count,r in enumerate(frozen,1):
        src=byroot[r['root_id']];path=three/src['payload_file'];require(sha(path)==src['payload_sha256'],'Root payload drift')
        with np.load(path,allow_pickle=False) as z:
            meta=json.loads(str(z['metadata_json'].item()));onset=int(z['onset_position'].item())
            require(meta['identity']==r and onset==int(r['onset_position']) and int(z['original_indices'][onset])==int(r['onset_original_trial_index']),'Root onset/identity mismatch')
            cells=meta['logical_cells'];keys=[(x['updater'],x['view'],int(x['sign']),int(x['seed']),x['shape']) for x in cells]
            require(len(keys)==108 and set(keys)==expected,'Logical branch coverage drift')
            states={k:z[k].copy() for k in z.files if k.endswith('__state')}
            for cell in cells:
                u=cell['updater'];b=cell['physical_bundle'];values=core.primitive_areas(states['clean__'+u+'__state'],states[b+'__'+u+'__state'],onset,cell['shape'],cell['view'],c)
                primitive.append({**{k:r[k] for k in ('participant_id','recording_id','segment_id','root_id')},'updater':u,'view':cell['view'],'shape':cell['shape'],'sign':int(cell['sign']),'seed':int(cell['seed']),'physical_bundle':b,**values,'replay_payload_sha256':src['payload_sha256']})
        if count%500==0:print(json.dumps({'progress':'primitive_roots','completed':count,'total':4018,'elapsed_s':round(time.monotonic()-start,1)}),flush=True)
    require(len(primitive)==433944,'Incomplete primitive ledger')
    typed_roots=[{**r,'onset_original_trial_index':int(r['onset_original_trial_index'])} for r in frozen]
    reduced=core.reduce_m1_rows(primitive,typed_roots,c)
    csv_write(root/'PRIMITIVE_AREAS.csv',primitive)
    for name,key in [('ROOT_COMPONENT_AREAS.csv','root_rows'),('RECORDING_COMPONENT_AREAS.csv','recording_rows'),('PARTICIPANT_COMPONENT_AREAS.csv','participant_rows'),('POPULATION_COMPONENT_AREAS.csv','component_population_rows')]:csv_write(root/name,reduced[key])
    aux_rows,aux_counts=auxiliary(root,plan,c);endpoint_rows=reduced['endpoint_rows']+aux_rows
    csv_write(root/'PARTICIPANT_ENDPOINTS.csv',endpoint_rows)
    groups=defaultdict(list)
    for r in endpoint_rows:groups[r['endpoint_id']].append(r)
    require(len(groups)==16,'Exactly6Theta+2directD+8auxiliary contrasts required')
    draws=core.bootstrap_indices(c);np.save(root/'BOOTSTRAP_PARTICIPANT_INDICES.npy',draws,allow_pickle=False)
    intervals=[];audit={};matrix=[];names=[]
    for eid,rows in groups.items():
        x=core.validate_participant_vector(rows,c);bca=core.bca(x,c,draws)
        require(bca['N']==19,'No participant shrink')
        audit[eid+'__bootstrap_means']=bca.pop('bootstrap_means');audit[eid+'__jackknife']=bca.pop('jackknife')
        intervals.append({'endpoint_id':eid,'endpoint_family':rows[0]['endpoint_family'],'updater':rows[0]['updater'],'view':rows[0].get('view',''),**bca})
        matrix.append(x);names.append(eid)
    np.savez_compressed(root/'BCA_AUDIT_ARRAYS.npz',**audit)
    np.savez_compressed(root/'PAIRED_ENDPOINT_MATRIX.npz',participant_ids=np.asarray(c['cohort']['group_ids']),endpoint_ids=np.asarray(names),values=np.column_stack(matrix))
    write(root/'EXPLORATORY_ESTIMATES.json',{'status':'COMPLETED_PENDING_INDEPENDENT_ACCEPTANCE','contract_sha256':LOCK,'N':19,'endpoints':intervals,'shared_bootstrap_indices_sha256':sha(root/'BOOTSTRAP_PARTICIPANT_INDICES.npy'),'bootstrap_resamples':10000,'paired_participant_resampling':True,'p_values':False,'interval_coverage':'pointwise_conditional'})
    # Nested scalar metadata is preserved in JSON; CSV is the compact result table.
    scalar_keys=['endpoint_id','endpoint_family','updater','view','N','estimate','interval_status','lower','upper','reason']
    csv_write(root/'EXPLORATORY_ESTIMATES.csv',[{k:r.get(k,'') for k in scalar_keys} for r in intervals])
    write(root/'ENDPOINT_SUPPORT_LEDGER.json',{'status':'PASS','contract_sha256':LOCK,'expected_participant_ids':c['cohort']['group_ids'],'endpoint_participant_ids':{k:[r['participant_id'] for r in v] for k,v in groups.items()},'missing_nonfinite_duplicate_extra':[],'cohort_shrink':False,'primitive_cells':len(primitive),'common_roots':len(frozen)})
    result={'status':'COMPLETED_PENDING_INDEPENDENT_ACCEPTANCE','at_utc':datetime.now(timezone.utc).isoformat(),'contract_sha256':LOCK,'plan_sha256':sha(root/'W09E_4_EXECUTION_PLAN.json'),'fixed_N':19,'common_roots':4018,'primitive_cells':len(primitive),'root_components':len(reduced['root_rows']),'recording_components':len(reduced['recording_rows']),'participant_components':len(reduced['participant_rows']),'endpoints':len(intervals),'participant_endpoint_values':len(endpoint_rows),'available_BCa_intervals':sum(r['interval_status']=='AVAILABLE' for r in intervals),'unavailable_BCa_intervals':sum(r['interval_status']!='AVAILABLE' for r in intervals),'auxiliary':aux_counts,'shared_draw_shape':list(draws.shape),'no_refit_reselection_or_root_change':True,'W09E_5_started':False,'manuscript_changed':False,'runtime_s':time.monotonic()-start}
    write(root/'W09E_4_EXECUTION_RESULT.json',result);print(json.dumps(result),flush=True)
def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);q=p.parse_args()
    try:run(q.root)
    except Exception as e:
        if q.root.exists():write(q.root/'W09E_4_ERROR_RECEIPT.json',{'status':'HALT','error_type':type(e).__name__,'reason':str(e),'no_cohort_shrink_root_deletion_or_rule_change':True})
        raise
if __name__=='__main__':main()
