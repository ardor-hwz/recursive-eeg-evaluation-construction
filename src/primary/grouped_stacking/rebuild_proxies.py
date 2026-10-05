"""Repair only five e01 ensembles; preserve historical slow and role decisions."""
from pathlib import Path
import sys,json,pickle,hashlib,time,difflib
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parents[1];EXP=HERE.parent;ROOT=EXP.parent
sys.path[:0]=[str(HERE/'src'),str(ROOT),str(EXP)]
import numpy as np
import pandas as pd
import yaml
from repair_bootstrap import subject_bootstrap,stacking_splits
from phase35_statistical_corrected import fit_point_candidate
from src.data.seed_vig import load_all_sessions
from src.data.splits import subjects_to_sessions
from src.search.development_search import FeatureCache
from src_major_revision import nested_runner as nr

F=[1,3,5,12,13]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,v):
    p.parent.mkdir(parents=True,exist_ok=True)
    if p.exists():raise FileExistsError(p)
    p.write_text(json.dumps(v,indent=2,default=lambda x:x.item() if isinstance(x,np.generic) else str(x)),encoding='utf-8')
def stack(cache,records,ids,feature):return cache.stack([records[s] for s in ids],feature)

def run():
    assert json.loads((HERE/'reports/INVARIANTS_PASS.json').read_text())['status']=='PASS'
    paths=list((ROOT/'src').rglob('*.py'))+list((ROOT/'configs').glob('*.yaml'))+list((EXP/'src_major_revision').glob('*.py'))+list((HERE/'src').glob('*.py'))
    paths += [EXP/'configs/nested_subject_evaluation_protocol.json',EXP/'configs/comparator_grid_lock.json']
    for s in range(1,22):
        b=EXP/'results'/f'outer_subject_{s:02d}'
        paths += list(b.rglob('*.json'))+list(b.rglob('*.csv'))+list((b/'checkpoints').rglob('*.pkl'))
    provenance={str(p):sha(p) for p in paths}
    pp=HERE/'reports/SOURCE_INPUT_HASHES.json'
    if pp.exists():
        old=json.loads(pp.read_text());assert old==provenance,'STOP source/input changed'
    else:save(pp,provenance)
    phase=yaml.safe_load((ROOT/'configs/phase35.yaml').read_text())
    candidates=yaml.safe_load((ROOT/'configs/phase35_search_space.yaml').read_text())['candidates']
    candidate=next(c for c in candidates if c['id']=='p35_e01')
    for outer in F:
        oldroot=EXP/'results'/f'outer_subject_{outer:02d}';out=HERE/'results'/f'outer_subject_{outer:02d}'
        if (out/'PROXIES_COMPLETE.json').exists():continue
        lock=json.loads((oldroot/'selection_lock.json').read_text());assert lock['proxy_roles']['fast_candidate_id']=='p35_e01' and lock['proxy_roles']['slow_candidate_id']=='p35_e02'
        # All development selection data excludes the outer group.
        records=load_all_sessions(ROOT/'results/data_audit/manifest.csv',data_root=Path(phase['dataset']['root']),session_ids=subjects_to_sessions(lock['inner_subject_ids']),strict=True)
        assert all(r.subject_id!=outer for r in records)
        lookup={r.session_id:r for r in records};cache=FeatureCache(records,phase['features']['windows'])
        dev=pd.read_csv(oldroot/'inner_selection/tracker/tracker_selection_oof.csv')
        result=dev.copy()
        for (seed,fold),g in dev.groupby(['seed','fold_id'],sort=True):
            row=g.iloc[0];train=json.loads(row.train_subjects);val=json.loads(row.validation_subjects)
            tx,ty,ts,_,names=stack(cache,lookup,subjects_to_sessions(train),candidate['feature_set'])
            vx,vy,vs,vsession,_=stack(cache,lookup,subjects_to_sessions(val),candidate['feature_set'])
            predictions=[]
            for ms in json.loads(row.fast_member_seeds):
                p=out/'development_members'/str(seed)/str(fold)/f'member_{ms}.npz'
                b=subject_bootstrap(tx,ty,ts,ms)
                assert outer not in b.original_subject_ids and not set(b.original_subject_ids)&set(val)
                assert all(not set(b.original_subject_ids[a])&set(b.original_subject_ids[z]) for a,z in stacking_splits(b.original_subject_ids,4))
                if p.exists():pred=np.load(p)['prediction']
                else:
                    t=time.time();model=fit_point_candidate(b.features,b.target,b.bootstrap_draw_ids,vx,vy,vs,vsession,candidate=candidate,model_config=nr._model_config(phase),seed=ms,feature_names=names,stacking_original_subject_ids=b.original_subject_ids)
                    pred=model.predict(vx,vsession);assert np.isfinite(pred).all()
                    p.parent.mkdir(parents=True,exist_ok=True);np.savez(p,prediction=pred,session_ids=vsession)
                    save(p.with_suffix('.json'),dict(outer=outer,seed=int(seed),inner_fold=int(fold),member_seed=ms,train=train,validation=val,sampled_original_subjects=b.sampled_original_subjects.tolist(),row_count=len(b.target),stack_grouping='original_subject_id',weighting='bootstrap_draw_id',model_kind=model.model_kind,source_input_hashes_sha256=sha(pp),prediction_sha256=sha(p)))
                    print(f'DEV outer={outer} seed={seed} fold={fold} member={ms} complete seconds={time.time()-t:.1f}',flush=True)
                predictions.append(pred)
            mean=np.mean(predictions,axis=0)
            for sid in np.unique(vsession):
                mask=(result.seed==seed)&(result.session_id==sid)
                assert mask.sum()==(vsession==sid).sum()
                result.loc[mask,'fast']=mean[vsession==sid]
        assert np.array_equal(result.drop(columns='fast').to_numpy(),dev.drop(columns='fast').to_numpy())
        dp=out/'inner_selection/tracker/tracker_selection_oof.csv';dp.parent.mkdir(parents=True,exist_ok=True)
        if not dp.exists():result.to_csv(dp,index=False)
        # Final proxy training uses historical checkpoint partition metadata exactly.
        outer_records=load_all_sessions(ROOT/'results/data_audit/manifest.csv',data_root=Path(phase['dataset']['root']),session_ids=subjects_to_sessions([outer]),strict=True)
        allrecords=records+outer_records;lookup={r.session_id:r for r in allrecords};cache=FeatureCache(allrecords,phase['features']['windows'])
        final=[]
        for seed in [42,123,2026]:
            predictions=[]
            for oldp in sorted((oldroot/'checkpoints/fast'/str(seed)).glob('*.pkl')):
                with oldp.open('rb') as f:meta=pickle.load(f)
                train=meta['train_subjects'];val=meta['validation_subjects'];ms=meta['member_seed']
                assert outer not in train+val and not set(train)&set(val)
                tx,ty,ts,_,names=stack(cache,lookup,subjects_to_sessions(train),candidate['feature_set'])
                vx,vy,vs,vsession,_=stack(cache,lookup,subjects_to_sessions(val),candidate['feature_set'])
                ox,_,_,osession,_=stack(cache,lookup,subjects_to_sessions([outer]),candidate['feature_set'])
                b=subject_bootstrap(tx,ty,ts,ms);p=out/'checkpoints/fast'/str(seed)/oldp.name
                if p.exists():
                    with p.open('rb') as f:model=pickle.load(f)['model']
                else:
                    t=time.time();model=fit_point_candidate(b.features,b.target,b.bootstrap_draw_ids,vx,vy,vs,vsession,candidate=candidate,model_config=nr._model_config(phase),seed=ms,feature_names=names,stacking_original_subject_ids=b.original_subject_ids)
                    p.parent.mkdir(parents=True,exist_ok=True)
                    with p.open('xb') as f:pickle.dump(dict(model=model,role='fast',candidate=candidate,outer_subject=outer,train_subjects=train,validation_subjects=val,member_seed=ms,sampled_original_subjects=b.sampled_original_subjects.tolist(),stack_grouping='original_subject_id',weighting='bootstrap_draw_id',source_input_hashes_sha256=sha(pp)),f,protocol=pickle.HIGHEST_PROTOCOL)
                    print(f'FINAL outer={outer} seed={seed} member={ms} complete seconds={time.time()-t:.1f}',flush=True)
                predictions.append(model.predict(ox,osession))
            mean=np.mean(predictions,axis=0)
            for sid in np.unique(osession):
                final.append(pd.DataFrame(dict(outer_subject_id=outer,seed=seed,session_id=sid,time_index=np.arange(sum(osession==sid)),fast=mean[osession==sid])))
        fp=out/'corrected_outer_fast.csv';pd.concat(final,ignore_index=True).to_csv(fp,index=False)
        save(out/'PROXIES_COMPLETE.json',dict(outer=outer,development_members=30,final_checkpoints=9,development_sha256=sha(dp),outer_fast_sha256=sha(fp)))
    print('ALL FIVE PROXY ENSEMBLES COMPLETE',flush=True)
if __name__=='__main__':run()
