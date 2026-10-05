"""Existing tracker/P0/C0 procedures on repaired OOF, no proxy-role reselection."""
from pathlib import Path
import sys,json,ast,copy,shutil,hashlib
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parents[1];EXP=HERE.parent;ROOT=EXP.parent
C0SRC=EXP/'c0_release_candidate_20260902_development_selection_lock_serialization_v1/src'
P0SRC=EXP/'postprocessing_release_candidate_20260730_v2_1/src'
sys.path[:0]=[str(HERE/'src'),str(ROOT),str(EXP),str(C0SRC),str(P0SRC)]
import numpy as np
import pandas as pd
import yaml
from src_major_revision import nested_runner as nr
from c0.selection import select_from_subject_rmse
from c0.candidates import AlphaCandidate
from c0.firewall import validate_development_selection_frame,DEVELOPMENT_COLUMNS
from c0.recursion import run_session,ConstantAlphaProvider
from c0.aggregation import aggregate_subject_metrics
from c0.inference import bca_mean
from bspc_postprocess.aggregation import subject_metrics_from_trajectory,build_all_paired_tables
from bspc_postprocess.inference import bootstrap_mean_difference,exact_sign_flip_test,holm_adjust
from bspc_postprocess.contracts import EXPECTED_METHODS
F=[1,3,5,12,13]
C0OLD=EXP/'c0_real_outer_execution_20260904_v1/events/C0_REAL_OUTER_EXEC_AUTH_20260904T115548Z/scientific_output'
C0DEV=EXP/'c0_development_selection_execution_retry_20260902_v4/events/C0_DEVSEL_AUTH_20260902T131152Z'
P0OLD=EXP/'postprocessing_release_candidate_20260730_v2_1/real_outputs/P0_V2_1_REAL_20260809T110708Z'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,v):
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(v,indent=2,default=lambda x:x.item() if isinstance(x,np.generic) else str(x)),encoding='utf-8')
def tracker_selection(frame,out):
    # Reuse the unmodified AST tail of the historical function, starting at sequence construction.
    tree=ast.parse((EXP/'src_major_revision/nested_runner.py').read_text())
    fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_tracker_oof_and_selection')
    start=next(i for i,n in enumerate(fn.body) if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id=='sequences')
    body=copy.deepcopy(fn.body[start:])
    new=ast.FunctionDef(name='select_existing_tracker',args=ast.arguments(posonlyargs=[],args=[ast.arg(arg=n) for n in ['frame','tracker_source_config','tracker_dir']],kwonlyargs=[],kw_defaults=[],defaults=[]),body=body,decorator_list=[])
    mod=ast.fix_missing_locations(ast.Module(body=[new],type_ignores=[]));namespace=dict(vars(nr));exec(compile(mod,str(EXP/'src_major_revision/nested_runner.py'),'exec'),namespace)
    out.mkdir(parents=True,exist_ok=True)
    return namespace['select_existing_tracker'](frame,yaml.safe_load((ROOT/'configs/change_aware_tracker.yaml').read_text()),out)

def c0_objectives(frame,outer):
    source=validate_development_selection_frame(frame[DEVELOPMENT_COLUMNS],outer_subject_id=outer,development_subject_ids=sorted(set(range(1,22))-{outer}),execution_class='SYNTHETIC_ONLY')
    # Batch only the independent alpha axis; each element uses exactly historical state_update arithmetic.
    alphas=np.arange(1001,dtype=float)/1000
    subjects=[]
    for sub,g in source.groupby('subject_id',sort=True):
        seed_values=[]
        for seed,sg in g.groupby('seed',sort=True):
            parts=[];truth=[]
            for sid,se in sg.groupby('session_id',sort=True):
                fast=se.fast.to_numpy(float);slow=se.slow.to_numpy(float);prev=np.full(1001,slow[0]);states=np.empty((1001,len(fast)))
                for t,value in enumerate(fast):
                    prev=np.clip(alphas*float(value)+(1.0-alphas)*prev,0.0,1.0);states[:,t]=prev
                parts.append(states);truth.append(se.target_perclos.to_numpy(float))
            predictions=np.concatenate(parts,axis=1);target=np.concatenate(truth)
            # Explicit per-alpha reducer keeps np.mean reduction order identical to frozen scalar code.
            seed_values.append(np.array([float(np.sqrt(np.mean((target-pred)**2,dtype=np.float64))) for pred in predictions]))
        subjects.append(np.mean(seed_values,axis=0,dtype=np.float64))
    matrix=np.asarray(subjects).T
    candidates=[AlphaCandidate(f'c0_alpha_k{k:04d}',k,k/1000) for k in range(1001)]
    metrics={c.candidate_id:matrix[i] for i,c in enumerate(candidates)}
    selected=select_from_subject_rmse(candidates,metrics,execution_class='SYNTHETIC_ONLY')
    return selected,metrics

def validate_c0_batch():
    # Every alpha, representative nonconstant signals: exact equality with original scalar recurrence.
    rng=np.random.default_rng(19);fast=rng.random(17);slow=rng.random(17);alphas=np.arange(1001)/1000;prev=np.full(1001,slow[0]);states=[]
    for f in fast:prev=np.clip(alphas*float(f)+(1-alphas)*prev,0,1);states.append(prev.copy())
    states=np.array(states).T
    for k in range(1001):assert np.array_equal(states[k],run_session(fast,slow,ConstantAlphaProvider(k/1000)).state)
    checks=[]
    for outer in F:
        frame=pd.read_csv(EXP/'results'/f'outer_subject_{outer:02d}/inner_selection/tracker/tracker_selection_oof.csv')
        selected,metrics=c0_objectives(frame,outer)
        receipt=json.loads((C0DEV/f'receipts/outer_subject_{outer:02d}_selection.json').read_text())
        maximum=max(float(np.max(np.abs(metrics[r['candidate_id']]-np.array(r['subject_rmse'])))) for r in receipt['candidate_objectives'])
        old=json.loads((C0DEV/f'locks/outer_subject_{outer:02d}.json').read_text())['base_envelope']['scientific_lock']['selection']
        assert maximum<=1e-14 and selected.selected_alpha==old['selected_alpha'],'STOP C0 kernel differs'
        checks.append(dict(outer=outer,all_1001_objectives_max_error=maximum,selected_alpha_equal=True))
        print('C0 historical numerical identity',outer,maximum,flush=True)
    dump(HERE/'reports/C0_KERNEL_VERIFICATION.json',dict(status='PASS',synthetic_all_alphas_exact=True,historical=checks))

def select_fold(outer):
    out=HERE/'results'/f'outer_subject_{outer:02d}';oldroot=EXP/'results'/f'outer_subject_{outer:02d}'
    assert (out/'PROXIES_COMPLETE.json').exists()
    if (out/'CORE_FOLD_COMPLETE.json').exists():return
    frame=pd.read_csv(out/'inner_selection/tracker/tracker_selection_oof.csv')
    tracker_path=out/'inner_selection/tracker/tracker_selection.json'
    if tracker_path.exists():
        tracker=json.loads(tracker_path.read_text());config=nr.ChangeAwareTrackerConfig.from_mapping(tracker['parameters'])
    else:config,_,tracker=tracker_selection(frame,out/'inner_selection/tracker')
    comparator=nr._comparator_selection(oof=frame,tracker_config=config,comparator_lock=json.loads((EXP/'configs/comparator_grid_lock.json').read_text()),outer_root=out)
    selected,metrics=c0_objectives(frame,outer)
    dump(out/'inner_selection/c0/selection.json',dict(execution_class='REAL_STACKING_REPAIR',selection=selected.to_dict(),outer_subject=outer,uses_outer=False,development_input_sha256=sha(out/'inner_selection/tracker/tracker_selection_oof.csv')))
    pd.DataFrame(metrics).to_csv(out/'inner_selection/c0/candidate_subject_rmse.csv',index=False)
    oldlock=json.loads((oldroot/'selection_lock.json').read_text());lock=copy.deepcopy(oldlock)
    lock['tracker']=tracker;lock['comparators']=comparator;lock['repair_version']=HERE.name;lock['parent_selection_lock_sha256']=sha(oldroot/'selection_lock.json');lock.pop('stable_selection_sha256',None)
    dump(out/'selection_lock.json',lock)
    # All development choices have now been saved, before opening outer outcomes.
    old=pd.read_csv(next((oldroot/'outer_evaluation').glob('*trajectories.csv')))
    keys=['outer_subject_id','seed','session_id','time_index'];fast=pd.read_csv(out/'corrected_outer_fast.csv')
    trajectory=old.drop(columns='fast').merge(fast,on=keys,validate='one_to_one').sort_values(keys).reset_index(drop=True)
    methods=nr._selected_method_parameters(comparator);c0states=[]
    for (_,sid),g in trajectory.groupby(['seed','session_id'],sort=True):
        f=g.fast.to_numpy(float);s=g.slow.to_numpy(float)
        trajectory.loc[g.index,'tracker']=nr.ChangeAwareTracker(config).track(f,s).state
        for family,roles in methods.items():
            for role,sp in roles.items():
                p=sp['parameters'];values=nr._ema(f,int(sid),float(p['alpha'])) if family=='ema' else nr._kalman(f,int(sid),float(p['q']),float(p['r']))
                trajectory.loc[g.index,f'{family}_{role}']=values
        c0states.extend(run_session(f,s,ConstantAlphaProvider(selected.selected_alpha)).state)
    trajectory['selection_lock_sha256']=sha(out/'selection_lock.json')
    trajectory=trajectory[old.columns];assert np.array_equal(trajectory.slow,old.sort_values(keys).slow)
    op=out/'outer_evaluation';op.mkdir(exist_ok=True);trajectory.to_csv(op/'p0_trajectories.csv',index=False)
    c0=trajectory[keys+['target_perclos','tracker']].rename(columns={'tracker':'full_state'});c0['c0_state']=c0states;c0['c0_alpha']=selected.selected_alpha;c0.to_csv(op/'c0_trajectories.csv',index=False)
    dump(out/'CORE_FOLD_COMPLETE.json',dict(outer=outer,p0_sha256=sha(op/'p0_trajectories.csv'),c0_sha256=sha(op/'c0_trajectories.csv')))

if __name__=='__main__':
    if sys.argv[1]=='verify-c0':validate_c0_batch()
    elif sys.argv[1]=='fold':select_fold(int(sys.argv[2]))
