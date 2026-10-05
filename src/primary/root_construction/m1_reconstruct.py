"""Original M1-B endpoints and reducers with independent branches batched."""
from common import *
import ast,copy
from types import SimpleNamespace
M1=EXP/'m1_lite_implementation_freeze_20260905_v1';OLD=EXP/'m1_production_20260909_v1'
C0SRC=EXP/'c0_release_candidate_20260902_development_selection_lock_serialization_v1/src/c0'
sys.path[:0]=[str(M1/'src'),str(C0SRC.parent)]
from m1_lite import event_universe as eu
from m1_lite.endpoints import primary_a,supportive_b_late,reduce_event_ledger
from m1_lite.inference import bca_mean
from m1_lite.production import run_event_causal_prefix
from src.models.change_aware_tracker import ChangeAwareTracker,ChangeAwareTrackerConfig
from c0.recursion import run_session,ConstantAlphaProvider

def roots_from_original_loop(inputs):
    # Execute the unchanged original inventory+eligibility AST. Stop before old observed count assertions.
    tree=ast.parse((M1/'src/m1_lite/event_universe.py').read_text())
    fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='build_event_universe')
    stop=next(i for i,n in enumerate(fn.body) if isinstance(n,ast.If))
    body=copy.deepcopy(fn.body[:stop]);body.append(ast.Return(value=ast.Tuple(elts=[ast.Name(id='eligible',ctx=ast.Load()),ast.Name(id='excluded',ctx=ast.Load())],ctx=ast.Load())))
    new=ast.FunctionDef(name='eligibility',args=ast.arguments(posonlyargs=[],args=[ast.arg(arg='inputs')],kwonlyargs=[],kw_defaults=[],defaults=[]),body=body,decorator_list=[])
    mod=ast.fix_missing_locations(ast.Module(body=[new],type_ignores=[]));ns=dict(vars(eu));exec(compile(mod,str(M1/'src/m1_lite/event_universe.py'),'exec'),ns)
    return ns['eligibility'](inputs)

def batch_full(f,slow,p):
    k,n=f.shape;prev=np.full(k,slow[0]);states=np.empty((k,n));alpha=np.empty((k,n));support=np.array([p['alpha_outlier'],p['alpha_stable'],p['alpha_transition']]);wsize=int(p['past_window'])
    for t in range(n):
        h=f[:,max(0,t-wsize):t];m=h.shape[1]
        if m<2:slope=np.zeros(k);cons=np.zeros(k);var=np.zeros(k)
        else:
            x=np.arange(m,dtype=float);x-=np.mean(x);means=np.mean(h,axis=1)
            slope=np.sum(x[None,:]*(h-means[:,None]),axis=1)/max(float(np.sum(x**2)),1e-12)
            dif=np.diff(h,axis=1);nz=np.abs(dif)>1e-12;count=np.sum(nz,axis=1)
            cons=np.abs(np.divide(np.sum(np.where(nz,np.sign(dif),0.),axis=1),count,out=np.zeros(k),where=count>0));var=np.var(h,axis=1)
        raw=f[:,t]-prev;d=f[:,t]-slow[t]
        g=np.where((np.abs(slope)<=1e-12)|(np.abs(raw)<=1e-12),.5,np.where(np.sign(slope)==np.sign(raw),1.,0.))
        i=1.-np.exp(-np.abs(raw)/p['innovation_scale']);d=1.-np.exp(-np.abs(d)/p['disagreement_scale']);l=1.-np.exp(-np.abs(slope)/p['slope_scale']);v=1.-np.exp(-np.maximum(var,0.)/p['variance_scale'])
        transition=p['transition_bias']+1.4*i+1.1*d+1.*l+1.*cons+1.2*g-.5*v
        outlier=p['outlier_bias']+1.6*i+1.2*d+1.1*(1.-cons)+1.1*(1.-g)+.5*v-.4*l
        stable=p['stable_bias']+1.5*(1.-i)+1.*(1.-d)+.8*(1.-l)+.8*(1.-v)+.3*cons
        scores=np.stack([outlier,stable,transition],axis=1);shift=scores/p['temperature'];shift-=np.max(shift,axis=1)[:,None];ex=np.exp(shift);weights=ex/np.sum(ex,axis=1)[:,None]
        a=np.matmul(weights[:,None,:],np.broadcast_to(support,(k,3))[:,:,None])[:,0,0]
        prev=np.clip(a*f[:,t]+(1.-a)*prev,0.,1.);states[:,t]=prev;alpha[:,t]=a
    return states,alpha

def batch_c0(f,slow,a):
    prev=np.full(len(f),slow[0]);states=np.empty_like(f)
    for t in range(f.shape[1]):prev=np.clip(a*f[:,t]+(1.-a)*prev,0.,1.);states[:,t]=prev
    return states

def main():
    assert (HERE/'results/e1b/DEVELOPMENT_COMPLETE.json').exists()
    out=HERE/'results/m1';out.mkdir(parents=True,exist_ok=True)
    inputs={s:pd.read_csv(outerpath(s),usecols=['outer_subject_id','seed','session_id','time_index','fast','slow']) for s in range(1,22)}
    roots,excluded=roots_from_original_loop(inputs);historical=load(M1/'M1_LITE_EVENT_UNIVERSE_MANIFEST.json')['eligible_roots']
    roots=sorted(roots);oldset=set(map(tuple,historical));newset=set(map(tuple,roots));rows=[]
    for s in range(1,22):
        a={r for r in oldset if r[0]==s};b={r for r in newset if r[0]==s}
        if s not in F:assert a==b,'STOP unaffected roots changed'
        rows.append(dict(outer=s,historical_roots=len(a),corrected_eligible_roots=len(b),added=len(b-a),removed=len(a-b),same=a==b))
    pd.DataFrame(rows).to_csv(HERE/'reports/M1_ROOT_AUDIT.csv',index=False)
    dump(out/'roots.json',dict(eligible_roots=roots,excluded=excluded,added=sorted(newset-oldset),removed=sorted(oldset-newset),eligibility_source_sha256=sha(M1/'src/m1_lite/event_universe.py'),eligibility_AST_unchanged=True))
    print(f'M1 ROOTS {len(historical)} -> {len(roots)}',flush=True)
    manifest=load(OLD/'output_manifest.json')
    for name in ['formal/primary_branch.jsonl','supportive/supportive_branch.jsonl','inference.json']:assert sha(OLD/name)==manifest['files'][name]['sha256']
    oldp=pd.read_json(OLD/'formal/primary_branch.jsonl',lines=True).drop(columns='classification');oldb=pd.read_json(OLD/'supportive/supportive_branch.jsonl',lines=True).drop(columns='classification');oldb['shape']='sustained'
    ledger=oldp.merge(oldb,on=['subject','session','t0','seed','sign','controller','shape'],how='left',validate='one_to_one').rename(columns={'subject':'outer_subject_id','session':'session_id'})
    ledger['primary_classification']='PRIMARY_CONFIRMATORY_COMPONENT';ledger['supportive_classification']='SUPPORTIVE_DESCRIPTIVE_ONLY'
    oldreduced=reduce_event_ledger(ledger,map(tuple,historical));oldci=bca_mean(oldreduced['theta'])
    assert max(abs(oldci[k]-load(OLD/'inference.json')[k]) for k in ['estimate','lower','upper'])<1e-12
    parts=[ledger[~ledger.outer_subject_id.isin(F)].copy()];verification=[]
    for s in F:
        dest=out/f'outer_subject_{s:02d}.csv'
        if dest.exists():parts.append(pd.read_csv(dest));continue
        p=load(lockpath(s))['tracker']['parameters'];a=c0alpha(s);resultrows=[]
        binding=SimpleNamespace(full_parameters={s:p},c0_alphas={s:a},full_source_path=ROOT/'src/models/change_aware_tracker.py',c0_source_root=C0SRC)
        for (seed,session),group in inputs[s].groupby(['seed','session_id'],sort=True):
            ordered=group.sort_values('time_index');f=ordered.fast.to_numpy(float);sl=ordered.slow.to_numpy(float);rr=[r[2] for r in roots if r[:2]==[s,int(session)]];assert rr
            clean=ChangeAwareTracker(ChangeAwareTrackerConfig.from_mapping(p)).track(f,sl).state;c0clean=run_session(f,sl,ConstantAlphaProvider(a)).state
            for sign in [-1,1]:
                for shape,duration in [('transient',1),('sustained',20)]:
                    perturbed=np.broadcast_to(f,(len(rr),len(f))).copy()
                    for j,t0 in enumerate(rr):perturbed[j,t0:t0+duration]+=sign*.05
                    assert np.min(perturbed)>=0 and np.max(perturbed)<=1
                    states,alphas=batch_full(perturbed,sl,p);cstates=batch_c0(perturbed,sl,a)
                    # Original causal-prefix engine checks early/middle/late roots in every corrected seed-session/sign/shape.
                    for j in sorted({0,len(rr)//2,len(rr)-1}):
                        t0=rr[j];exact=run_event_causal_prefix(binding,subject=s,fast=f,slow=sl,t0=int(t0),sign=sign,shape=shape);end=t0+270
                        error=max(float(np.max(np.abs(states[j,:end]-exact.branches['Full_perturbed'].state))),float(np.max(np.abs(alphas[j,:end]-exact.branches['Full_perturbed'].alpha))),float(np.max(np.abs(cstates[j,:end]-exact.branches['C0_perturbed'].state))))
                        assert error<2e-14,'STOP M1 canonical trajectory mismatch'
                        for ctrl,base,current in [('Full',clean,states[j]),('C0',c0clean,cstates[j])]:assert abs(primary_a(base,current,t0)-exact.primary[ctrl])<1e-10
                        verification.append(dict(outer=s,seed=int(seed),session=int(session),sign=sign,shape=shape,t0=t0,max_trajectory_alpha_error=error))
                    for j,t0 in enumerate(rr):
                        for ctrl,base,current in [('Full',clean,states[j]),('C0',c0clean,cstates[j])]:
                            assert np.max(np.abs(base[:t0]-current[:t0]))<2e-14
                            resultrows.append(dict(outer_subject_id=s,session_id=int(session),t0=t0,seed=int(seed),sign=sign,controller=ctrl,shape=shape,A=primary_a(base,current,t0),B_late=supportive_b_late(base,current,t0,sign) if shape=='sustained' else None,primary_classification='PRIMARY_CONFIRMATORY_COMPONENT',supportive_classification='SUPPORTIVE_DESCRIPTIVE_ONLY'))
            print(f'M1 outer {s} seed {seed} session {session} COMPLETE',flush=True)
        part=pd.DataFrame(resultrows);part.to_csv(dest,index=False);parts.append(part)
        pd.DataFrame([r for r in verification if r['outer']==s]).to_csv(out/f'canonical_verification_{s:02d}.csv',index=False)
    corrected=pd.concat(parts,ignore_index=True);corrected.to_csv(out/'branch_results.csv',index=False)
    reduced=reduce_event_ledger(corrected,map(tuple,roots));ci=bca_mean(reduced['theta'])
    for version,r in [('old',oldreduced),('corrected',reduced)]:
        r['subject_primary'].to_csv(out/f'{version}_subject_primary.csv',index=False);r['subject_supportive'].to_csv(out/f'{version}_subject_supportive.csv',index=False)
        pd.DataFrame(dict(outer_subject_id=range(1,22),theta=r['theta'])).to_csv(out/f'{version}_theta.csv',index=False)
    dump(out/'inference.json',ci);dump(out/'COMPLETE.json',dict(status='PASS',roots=len(roots),rows=len(corrected),negative_subjects=int(np.sum(reduced['theta']<0)),positive_subjects=int(np.sum(reduced['theta']>0)),source_sha256=sha(Path(__file__)),target_read_count=0))
    print('M1 COMPLETE',ci,flush=True)
if __name__=='__main__':main()
