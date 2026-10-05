"""Bounded synchronous W04 analysis of accepted primary M1 areas and group vectors."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
from scipy.stats import bootstrap

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
EXP = ROOT / 'major_revision_experiments_20260728'
W02 = ROOT / 'codex_workspace/W02_M1_decomposition'
W03 = ROOT / 'codex_workspace/W03_claim_retention'
FREEZE = EXP / 'm1_lite_implementation_freeze_20260905_v1'
DOWN = EXP / 'corrected_downstream_reconstruction_20260912_v1'
REVIEW = EXP / 'reviewer_mechanistic_controls_20260912_v1'
sys.path[:0] = [str(ROOT), str(FREEZE / 'src')]
from m1_lite.endpoints import LEDGER_KEYS, reduce_event_ledger
from m1_lite.inference import bca_mean, BOOTSTRAP_SEED, BOOTSTRAP_RESAMPLES, CONFIDENCE_LEVEL
from src.data.manifest import SUBJECT_SESSIONS
from src.utils.serialization import stable_hash

EPS = 1e-12
BRANCH = 'primary_corrected_192'
UNIVERSE = 'primary_fixed_roots'
ORIENTATION = 'DeltaConstruction_g = Theta_equal,g - Theta_fixed,g'
SIGN_INTERPRETATION = 'negative: equal total dose gives lower M1 interaction; positive: higher'
GROUPS = list(range(1, 22))
SCIENCE_FILES = [
    'W04_GROUP_CONTRASTS.csv', 'W04_POPULATION_SUMMARY.csv', 'W04_POPULATION_CELLS.csv',
    'W04_LEAVE_ONE_GROUP.csv', 'W04_BOOTSTRAP_CONTRAST.npy', 'W04_BOOTSTRAP_GROUP_INDICES.npy',
    'W04_ALGEBRA_AUDIT.json', 'W04_SOURCE_MANIFEST.json', 'W04_INFERENCE_RECEIPT.json',
    'W04_INTEGRITY_AUDIT.json', 'W04_RECOVERED_CONTRACT.json',
]


def require(ok, message):
    if not ok:
        raise ValueError(message)


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def dump(path, value):
    value = dict(value, orientation=ORIENTATION, sign_interpretation=SIGN_INTERPRETATION)
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n', encoding='utf-8')


def read(path, **kwargs):
    return pd.read_csv(path, float_precision='round_trip', **kwargs)


def error(a, b):
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    require(a.shape == b.shape and np.isfinite(a).all() and np.isfinite(b).all(), 'Invalid algebra operands')
    return float(np.max(np.abs(a-b), initial=0.0))


def close(a, b, message):
    discrepancy = error(a, b)
    require(discrepancy <= EPS, f'{message}: {discrepancy}')
    return discrepancy


def signs(values):
    values = np.asarray(values, dtype=np.float64)
    require(np.isfinite(values).all(), 'Nonfinite sign input')
    return np.where(values > EPS, '+', np.where(values < -EPS, '-', '0'))


def checked_groups(frame, group_column='group'):
    require(not frame.duplicated(group_column).any(), 'Duplicate analysis group')
    result = frame.sort_values(group_column).reset_index(drop=True)
    require(result[group_column].tolist() == GROUPS, 'Analysis groups must be exactly 1..21')
    return result


def primary_view(frame, view):
    return checked_groups(frame.loc[frame.analysis_layer.eq(BRANCH) & frame.universe.eq(UNIVERSE) &
                                    frame.view.eq(view)].copy())


class Bindings:
    def __init__(self):
        self.records = {}

    def check(self, path, expected=None, authority='W04 pre-computation inspection snapshot'):
        path = Path(path).resolve()
        actual = sha(path)
        if expected is not None:
            require(actual == expected, f'Source hash mismatch: {path}')
        previous = self.records.get(str(path))
        require(previous is None or previous['sha256'] == actual, f'Source changed: {path}')
        self.records[str(path)] = dict(sha256=actual, bytes=path.stat().st_size,
                                      existing_hash_verified=expected is not None,
                                      binding_authority=str(authority))
        return path

    def mapped(self, path, mapping, authority):
        key = str(Path(path).resolve())
        require(key in mapping, f'Missing accepted source binding: {key}')
        value = mapping[key]
        return self.check(path, value['sha256'] if isinstance(value, dict) else value, authority)

    def unchanged(self):
        require(all(sha(p) == v['sha256'] for p, v in self.records.items()), 'Consumed frozen source changed')


def validate_ledger(frame, roots):
    require(set(LEDGER_KEYS + ['A']).issubset(frame.columns), 'Missing primitive ledger fields')
    require(not frame.empty and not frame.duplicated(LEDGER_KEYS).any(), 'Empty/duplicate branch keys')
    require(np.isfinite(frame.A).all() and frame.A.ge(0).all(), 'Invalid accepted response area')
    require(set(frame.outer_subject_id) == set(GROUPS), 'Ledger group mismatch')
    require(set(frame.seed) == {42, 123, 2026} and set(frame.sign) == {-1, 1}, 'Seed/sign mismatch')
    require(set(frame.controller) == {'Full', 'C0'} and set(frame['shape']) == {'transient', 'sustained'},
            'Controller/shape mismatch')
    observed = sorted(map(tuple, frame[['outer_subject_id', 'session_id', 't0']].drop_duplicates().to_numpy()))
    require(observed == roots, 'Missing/extra accepted roots')
    require(len(frame) == 24*len(roots), 'Incomplete Cartesian branch coverage')
    require(frame.groupby(LEDGER_KEYS[:3]).size().eq(24).all(), 'Incomplete root tuple')


def reduced_components(frame, roots):
    validate_ledger(frame, roots)
    data = frame[LEDGER_KEYS + ['A']].copy()
    # Only the accepted primary reducer is consumed. These finite placeholders
    # let its unmodified supportive block run; no supportive output is used.
    data['B_late'] = 0.0
    data['primary_classification'] = 'PRIMARY_CONFIRMATORY_COMPONENT'
    data['supportive_classification'] = 'SUPPORTIVE_DESCRIPTIVE_ONLY'
    reduced = reduce_event_ledger(data, roots)
    pivot = reduced['subject_primary'].pivot(index='outer_subject_id', columns=['controller', 'shape'], values='A')
    cells = np.column_stack([pivot[key].to_numpy() for key in
                            [('Full', 'transient'), ('Full', 'sustained'), ('C0', 'transient'), ('C0', 'sustained')]])
    return cells, reduced['theta']


def algebra(fixed, equal, indices):
    delta = equal.Theta.to_numpy() - fixed.Theta.to_numpy()
    decomposed = (equal.D_S.to_numpy()-equal.D_T.to_numpy()) - (fixed.D_S.to_numpy()-fixed.D_T.to_numpy())
    sustained = equal.D_S.to_numpy()-fixed.D_S.to_numpy()
    primitive = (equal.A_Full_S.to_numpy()-equal.A_C0_S.to_numpy()) - (fixed.A_Full_S.to_numpy()-fixed.A_C0_S.to_numpy())
    transient_delta = equal.D_T.to_numpy()-fixed.D_T.to_numpy()
    operands = [(delta, equal.Theta.to_numpy()-fixed.Theta.to_numpy()),
                (delta, decomposed), (equal.D_T.to_numpy(), fixed.D_T.to_numpy()),
                (delta, sustained), (delta, primitive)]
    identities = {}
    for number, (a, b) in enumerate(operands, 1):
        identities[f'identity_{number}'] = dict(
            group_max_abs_error=close(a, b, f'Group identity {number}'),
            population_abs_error=close(np.mean(a), np.mean(b), f'Population identity {number}'),
            bootstrap_max_abs_error=close(a[indices].mean(axis=1), b[indices].mean(axis=1),
                                          f'Bootstrap identity {number}'))
    draws = delta[indices].mean(axis=1)
    # Independent arithmetic path: difference of resampled construction means.
    paired_difference = equal.Theta.to_numpy()[indices].mean(axis=1)-fixed.Theta.to_numpy()[indices].mean(axis=1)
    primitive_difference = (equal.A_Full_S.to_numpy()[indices].mean(axis=1)-equal.A_C0_S.to_numpy()[indices].mean(axis=1)) - (
        fixed.A_Full_S.to_numpy()[indices].mean(axis=1)-fixed.A_C0_S.to_numpy()[indices].mean(axis=1))
    audit = dict(schema='paper2.w04.algebra.v1', tolerance=EPS, identities=identities,
                 transient_cell_max_abs_error=max(close(equal.A_Full_T, fixed.A_Full_T, 'Full transient'),
                                                  close(equal.A_C0_T, fixed.A_C0_T, 'C0 transient')),
                 transient_D_T_max_abs_error=error(transient_delta, np.zeros(21)),
                 direct_vs_decomposed_max_abs_error=error(delta, decomposed),
                 sustained_only_max_abs_error=error(delta, sustained),
                 primitive_sustained_max_abs_error=error(delta, primitive),
                 population_difference_of_means_error=close(np.mean(delta), np.mean(equal.Theta)-np.mean(fixed.Theta),
                                                            'Population difference of means'),
                 bootstrap_difference_of_Theta_means_max_abs_error=close(draws, paired_difference, 'Shared Theta draws'),
                 bootstrap_primitive_cell_means_max_abs_error=close(draws, primitive_difference, 'Shared primitive draws'),
                 bootstrap_sustained_only_max_abs_error=error(draws, sustained[indices].mean(axis=1)),
                 bootstrap_resamples=10000, paired_indices_shape=list(indices.shape),
                 extra_inferential_test=False)
    audit['bootstrap_identity_max_abs_error'] = max(
        [v['bootstrap_max_abs_error'] for v in identities.values()] +
        [audit['bootstrap_difference_of_Theta_means_max_abs_error'], audit['bootstrap_primitive_cell_means_max_abs_error']])
    return delta, sustained, decomposed, draws, audit


def leave_one(values):
    require(np.asarray(values).shape == (21,) and np.isfinite(values).all(), 'Leave-one audit requires 21 groups')
    means = np.asarray([np.mean(np.delete(values, i), dtype=np.float64) for i in range(21)])
    direction = signs([np.mean(values)])[0]
    return means, signs(means) != direction


def save_csv(output_dir, name, frame):
    frame = frame.copy()
    frame['contrast_orientation'] = 'Theta_equal - Theta_fixed'
    frame['sign_interpretation'] = SIGN_INTERPRETATION
    frame['analysis_layer'] = BRANCH
    frame['universe'] = UNIVERSE
    frame.to_csv(output_dir/name, index=False, float_format='%.17g')


def main(output_dir=HERE):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    bind = Bindings()
    bind.check(HERE/'W04_SCIENTIFIC_CONTRACT.md')
    bind.check(Path(__file__))
    d3 = load(bind.check(W03/'W03_delivery_manifest.json'))['files']
    def accepted(folder, name, delivery):
        require(name in delivery, f'Accepted file absent: {name}')
        return bind.check(folder/name, delivery[name]['sha256'], folder/('W02_delivery_manifest.json' if folder == W02 else 'W03_delivery_manifest.json'))
    m3 = load(accepted(W03, 'W03_SOURCE_MANIFEST.json', d3))['sources']
    d2 = load(bind.mapped(W02/'W02_delivery_manifest.json', m3, W03/'W03_SOURCE_MANIFEST.json'))['files']
    consumed2 = ['W02_frozen_contract.json', 'W02_source_manifest.json', 'W02_acceptance.json', 'W02_integrity_audit.json',
                 'W02_reproducibility.json', 'W02_group_components.csv', 'W02_population_summary.csv',
                 'W02_frozen_config_identity.csv', 'W02_bootstrap_group_indices.npy', 'W02_root_audit.csv',
                 'W02_additional_provenance_audit.json']
    for name in consumed2:
        accepted(W02, name, d2)
        if str(W02/name) in m3:
            bind.mapped(W02/name, m3, W03/'W03_SOURCE_MANIFEST.json')
    for name in ['W03_acceptance.json', 'W03_reproducibility.json', 'W03_GROUP_TRANSITIONS.csv', 'W03_SCIENTIFIC_CONTRACT.md']:
        accepted(W03, name, d3)
    for folder, prefix in [(W02, 'W02'), (W03, 'W03')]:
        require(load(folder/f'{prefix}_acceptance.json')['status'] == 'PASS', 'Upstream workflow not accepted')
        receipt = load(folder/f'{prefix}_reproducibility.json')
        require(receipt['status'] == 'PASS' and receipt['synchronous_reruns'] == 1, 'Upstream reproduction failed')
        for name in consumed2 if folder == W02 else ['W03_GROUP_TRANSITIONS.csv', 'W03_SOURCE_MANIFEST.json']:
            if name in receipt['original_hashes']:
                bind.check(folder/name, receipt['original_hashes'][name], folder/f'{prefix}_reproducibility.json')
    source_map = load(W02/'W02_source_manifest.json')['sources']
    def upstream(path):
        return bind.mapped(path, source_map, W02/'W02_source_manifest.json')
    for module in ['endpoints.py', 'inference.py', 'contract.py', 'errors.py', '__init__.py']:
        path = FREEZE/'src/m1_lite'/module
        if str(path) in source_map: upstream(path)
        else: bind.check(path)
    upstream(ROOT/'src/data/manifest.py')
    bind.check(ROOT/'src/utils/serialization.py')
    contract = load(W02/'W02_frozen_contract.json')
    frozen = load(upstream(FREEZE/'M1_LITE_FROZEN_PROTOCOL.json'))
    require(contract['primary_analysis_layer'] == BRANCH and contract['roots'] == 13654, 'Primary branch/root drift')
    require(contract['group_recording_mapping'] == {str(k): v for k, v in SUBJECT_SESSIONS.items()}, 'Recording mapping drift')
    require(contract['intervention']['seeds'] == [42, 123, 2026] and contract['intervention']['signs'] == [-1, 1], 'Intervention identity drift')
    require(contract['endpoint'] == frozen['endpoints']['primary'] and contract['reducer'] == frozen['reducers'] and
            contract['timing'] == frozen['timing'], 'Endpoint/reducer/horizon drift')
    require(frozen['inference']['rng_seed'] == BOOTSTRAP_SEED == 20260905 and
            frozen['inference']['bootstrap_resamples'] == BOOTSTRAP_RESAMPLES == 10000 and
            CONFIDENCE_LEVEL == .95 and frozen['inference']['bit_generator'] == 'PCG64', 'Inference settings drift')
    root_contract = load(upstream(EXP/'boundary_stage23_contract_20260924_v1/M1_ROOT_CONTRACT.json'))
    root_ref = root_contract['primary_root_universe']
    roots_path = bind.check(root_ref['path'], root_ref['sha256'], 'current M1_ROOT_CONTRACT.json')
    roots = [tuple(map(int, v)) for v in load(roots_path)['eligible_roots']]
    require(len(roots) == len(set(roots)) == 13654 and roots == sorted(roots), 'Root identity/count drift')
    require(all(session in SUBJECT_SESSIONS[g] and 20 <= onset <= 615 for g, session, onset in roots), 'Root recording/onset drift')
    require(set(session for _, session, _ in roots) == set(range(23)), 'Root recording universe drift')
    root_audit = read(W02/'W02_root_audit.csv')
    require(int(root_audit.eligible_roots.sum()) == 13654 and int(root_audit.structural_roots.sum()) == 13708 and
            int(root_audit.excluded_roots.sum()) == 54 and root_audit.both_signs_all_seeds_verified.all(), 'Accepted eligibility audit drift')
    configs = checked_groups(read(W02/'W02_frozen_config_identity.csv').query('analysis_layer == @BRANCH'))
    for row in configs.itertuples():
        lock = load(upstream(Path(row.Full_source_lock)))
        require(lock['tracker']['candidate_id'] == row.Full_candidate_id and
                stable_hash(lock['tracker']['parameters']) == row.Full_config_sha256 and
                lock['tracker']['parameters'] == json.loads(row.Full_parameters_json), 'Full controller identity mismatch')
        c0 = load(upstream(Path(row.C0_source_lock)))
        selection = c0['selection'] if 'selection' in c0 else c0['base_envelope']['scientific_lock']['selection']
        require(selection['selected_alpha'] == row.C0_alpha and
                selection.get('selected_candidate_id', f'alpha_{row.C0_alpha:.3f}') == row.C0_candidate_id, 'C0 identity mismatch')
        require(json.loads(row.recordings_json) == SUBJECT_SESSIONS[row.group] and
                lock['proxy_roles']['fast_candidate_id'] == row.fast_candidate_id and
                lock['proxy_roles']['slow_candidate_id'] == row.slow_candidate_id, 'Recording/proxy identity mismatch')
    upstream(REVIEW/'src/m1_controls.py')
    upstream(REVIEW/'reports/OUTPUT_HASHES.json')
    output_hashes = load(REVIEW/'reports/OUTPUT_HASHES.json')
    legacy_receipts = {}
    for name in ['COMPLETE.json', 'ROOT_BINDING.json', 'TRANSIENT_PROVENANCE.json']:
        path = bind.mapped(REVIEW/'results/m1'/name, output_hashes, REVIEW/'reports/OUTPUT_HASHES.json')
        legacy_receipts[name] = load(path)
    require(legacy_receipts['COMPLETE.json']['status'] == 'PASS' and legacy_receipts['COMPLETE.json']['transient_reused'] and
            legacy_receipts['ROOT_BINDING.json']['root_file_sha256'] == root_ref['sha256'] and
            legacy_receipts['ROOT_BINDING.json']['new_roots'] == legacy_receipts['ROOT_BINDING.json']['removed_roots'] == 0,
            'Accepted equal-construction receipt drift')
    fixed_path = upstream(DOWN/'results/m1/branch_results.csv')
    equal_path = upstream(REVIEW/'results/m1/equal_branch_results.csv')
    require(legacy_receipts['TRANSIENT_PROVENANCE.json']['source_sha256'] == sha(fixed_path), 'Transient source binding drift')
    fixed_events, equal_events = [read(path, usecols=LEDGER_KEYS+['A']) for path in [fixed_path, equal_path]]
    validate_ledger(fixed_events, roots); validate_ledger(equal_events, roots)
    a, b = [events.loc[events['shape'].eq('transient')].sort_values(LEDGER_KEYS).reset_index(drop=True)
            for events in [fixed_events, equal_events]]
    require(a[LEDGER_KEYS].equals(b[LEDGER_KEYS]), 'Transient keys differ')
    transient_event_error = close(a.A, b.A, 'Transient accepted event areas')
    all_components = read(W02/'W02_group_components.csv')
    fixed, equal = [primary_view(all_components, view) for view in ['fixed_amplitude', 'equal_total_dose']]
    recovery_errors = {}
    for name, events, comp in [('fixed', fixed_events, fixed), ('equal', equal_events, equal)]:
        cells, theta = reduced_components(events, roots)
        recovery_errors[name] = dict(
            event_reaggregation_cells_max_abs_error=close(cells, comp[['A_Full_T','A_Full_S','A_C0_T','A_C0_S']].to_numpy(), 'Accepted W02 cells'),
            event_reaggregation_Theta_max_abs_error=close(theta, comp.Theta, 'Accepted W02 Theta'))
        archived = read(upstream(REVIEW/f'results/m1/{name}_theta.csv'))
        require(archived.outer_subject_id.tolist() == GROUPS, 'Archived Theta group identities differ')
        recovery_errors[name]['archived_Theta_max_abs_error'] = close(comp.Theta, archived.theta, 'Corrected archived Theta')
    population = read(W02/'W02_population_summary.csv')
    for view, comp in [('fixed_amplitude', fixed), ('equal_total_dose', equal)]:
        row = population.loc[population.analysis_layer.eq(BRANCH) & population.universe.eq(UNIVERSE) & population.view.eq(view)]
        require(len(row) == 1 and row.iloc[0].N == 21 and row.iloc[0].roots == 13654, 'Accepted population identity drift')
        close(comp.Theta.mean(), row.iloc[0].Theta, 'Accepted W02 population mean')
    transitions = read(W03/'W03_GROUP_TRANSITIONS.csv')
    transitions = checked_groups(transitions.loc[transitions.branch.eq(BRANCH) & transitions.family.eq('M1') &
        transitions.universe.eq(UNIVERSE) & transitions.endpoint.eq('Theta') &
        transitions.source_construction.eq('fixed_amplitude') & transitions.destination_construction.eq('equal_total_dose')], 'group_id')
    close(fixed.Theta, transitions.source_value, 'W03 fixed vector')
    close(equal.Theta, transitions.destination_value, 'W03 equal vector')
    normalize_sign = lambda values: np.asarray([str(v).replace('−', '-') for v in values])
    require(np.array_equal(signs(fixed.Theta), normalize_sign(transitions.source_sign)) and
            np.array_equal(signs(equal.Theta), normalize_sign(transitions.destination_sign)), 'W03 sign identities drift')
    print('W04 primary roots, locks, accepted W02/W03 vectors and transient keys verified.', flush=True)
    indices = np.random.Generator(np.random.PCG64(BOOTSTRAP_SEED)).integers(0, 21, size=(10000, 21))
    require(np.array_equal(indices, np.load(W02/'W02_bootstrap_group_indices.npy', allow_pickle=False)), 'Accepted paired index matrix drift')
    delta, sustained, reconstructed, draws, audit = algebra(fixed, equal, indices)
    ci = bca_mean(delta)
    result = bootstrap((delta,), np.mean, vectorized=False, paired=False, n_resamples=10000,
                       confidence_level=.95, method='BCa', random_state=np.random.Generator(np.random.PCG64(BOOTSTRAP_SEED)))
    audit['frozen_scipy_draw_max_abs_error'] = close(draws, result.bootstrap_distribution, 'Frozen BCa index distribution')
    close([ci['lower'],ci['upper']], [result.confidence_interval.low,result.confidence_interval.high], 'Frozen BCa limits')
    audit['transient_event_area_max_abs_error'] = transient_event_error
    audit['identity_formulas'] = {
        'identity_1': 'DeltaConstruction = Theta_equal - Theta_fixed',
        'identity_2': 'DeltaConstruction = (D_S_equal-D_T_equal)-(D_S_fixed-D_T_fixed)',
        'identity_3': 'D_T_equal = D_T_fixed',
        'identity_4': 'DeltaConstruction = D_S_equal-D_S_fixed',
        'identity_5': 'DeltaConstruction = (A_Full_S_equal-A_C0_S_equal)-(A_Full_S_fixed-A_C0_S_fixed)'}
    group = pd.DataFrame(dict(group_id=GROUPS, Theta_fixed=fixed.Theta, Theta_equal=equal.Theta,
        DeltaConstruction=delta, sign_DeltaConstruction=signs(delta), D_S_fixed=fixed.D_S, D_S_equal=equal.D_S,
        D_T_fixed=fixed.D_T, D_T_equal=equal.D_T, transient_identity_error=equal.D_T-fixed.D_T,
        sustained_reconstruction=sustained, contrast_reconstruction_error=delta-sustained,
        Theta_fixed_sign=signs(fixed.Theta), Theta_equal_sign=signs(equal.Theta),
        W03_transition=transitions.transition, full_reconstruction=reconstructed))
    for tag, comp in [('fixed', fixed), ('equal', equal)]:
        for cell in ['A_Full_T', 'A_Full_S', 'A_C0_T', 'A_C0_S']:
            group[f'{cell}_{tag}'] = comp[cell]
    omitted, changed = leave_one(delta)
    q1, median, q3 = np.quantile(delta, [.25, .5, .75], method='linear')
    summary = dict(N_groups=21, N_recordings=23, roots=13654, mean_Theta_fixed=float(fixed.Theta.mean()),
        mean_Theta_equal=float(equal.Theta.mean()), mean_DeltaConstruction=ci['estimate'], CI_low=ci['lower'], CI_high=ci['upper'],
        positive_groups=int(np.count_nonzero(signs(delta)=='+')), negative_groups=int(np.count_nonzero(signs(delta)=='-')),
        zero_groups=int(np.count_nonzero(signs(delta)=='0')), median_DeltaConstruction=float(median), Q1=float(q1), Q3=float(q3),
        min=float(delta.min()), max=float(delta.max()), leave_one_group_min_mean=float(omitted.min()),
        leave_one_group_max_mean=float(omitted.max()), leave_one_direction_changed=bool(changed.any()))
    influence = pd.DataFrame(dict(omitted_group_id=GROUPS, remaining_N_groups=20, mean_DeltaConstruction=omitted,
                                 population_direction=signs(omitted), direction_changed=changed, descriptive_only=True))
    cell_summary = pd.DataFrame([dict(quantity=f'{col}_{tag}', mean=float(comp[col].mean()),
                                     status='descriptive; no component interval')
                                for tag, comp in [('fixed',fixed),('equal',equal)]
                                for col in ['A_Full_T','A_Full_S','A_C0_T','A_C0_S','D_S','D_T','Theta']])
    cell_summary = pd.concat([cell_summary,pd.DataFrame([dict(quantity='DeltaConstruction',mean=ci['estimate'],status='sole new inferential estimand')])],ignore_index=True)
    inference = dict(ci, schema='paper2.w04.direct_paired_bca.v1', endpoint='equal_group_mean_DeltaConstruction',
        frozen_helper_endpoint_label=ci['endpoint'], frozen_helper_schema=ci['schema_version'],
        independent_unit='filename-defined analysis group', group_count=21, input='already paired 21-vector',
        separate_construction_resampling=False, jackknife='leave_one_group_out', interval_scope='two-sided pointwise 95% BCa',
        bootstrap_distribution='W04_BOOTSTRAP_CONTRAST.npy', bootstrap_indices='W04_BOOTSTRAP_GROUP_INDICES.npy',
        population_summary=summary, sign_tolerance=EPS, quartile_method='NumPy linear',
        positive_group_ids=group.loc[group.sign_DeltaConstruction.eq('+'),'group_id'].tolist(),
        negative_group_ids=group.loc[group.sign_DeltaConstruction.eq('-'),'group_id'].tolist(),
        zero_group_ids=group.loc[group.sign_DeltaConstruction.eq('0'),'group_id'].tolist(),
        python_version=platform.python_version(), numpy_version=np.__version__, pandas_version=pd.__version__,
        new_scientific_replay=False, training=False, new_stage1b_inference=False)
    recovered = dict(contract, schema='paper2.w04.recovered_contract.v1', workflow=4,
        scientific_status='bounded retrospective perturbation-construction sensitivity analysis',
        inference_endpoint='equal_group_mean_DeltaConstruction', root_sha256=root_ref['sha256'],
        inference_unit='21 filename-defined analysis groups', frozen_group_ids=GROUPS,
        primary_state_archive_limitation=load(W02/'W02_integrity_audit.json')['primary_area_audit'])
    integrity = dict(schema='paper2.w04.integrity.v1', status='PASS', scientific_hard_gates=[],
        exactly_21_paired_groups=True, same_primary_branch=True, same_roots_recordings_seeds_signs=True,
        Full_C0_identities_verified=True, controller_identity_count=21, event_rows_per_construction=len(fixed_events),
        transient_keys_per_construction=len(a), root_sha256=root_ref['sha256'],
        W02_recovery_errors=recovery_errors, W03_pairing_and_vector_reproduction=True,
        transient_identity_verified=True, algebra_closed=True, bootstrap_pairing_verified=True,
        bootstrap_matches_frozen_helper=True, consumed_frozen_sources_unchanged=True,
        new_scientific_replays=0, upstream_training=0, controller_reselection=0, new_roots=0,
        new_p_values=0, new_inferential_estimands=1, new_stage1b_inference=False, search_regime_audit=False,
        polling_or_background_helpers=False, workflows_5_executed=False,
        primary_state_archive_limitation_disclosed=True, tolerance=EPS)
    bind.unchanged()
    for name, frame in [('W04_GROUP_CONTRASTS.csv',group),('W04_POPULATION_SUMMARY.csv',pd.DataFrame([summary])),
                        ('W04_POPULATION_CELLS.csv',cell_summary),('W04_LEAVE_ONE_GROUP.csv',influence)]:
        save_csv(output_dir,name,frame)
    np.save(output_dir/'W04_BOOTSTRAP_CONTRAST.npy',draws,allow_pickle=False)
    np.save(output_dir/'W04_BOOTSTRAP_GROUP_INDICES.npy',indices,allow_pickle=False)
    for name, value in [('W04_ALGEBRA_AUDIT.json',audit),('W04_INFERENCE_RECEIPT.json',inference),
                        ('W04_INTEGRITY_AUDIT.json',integrity),('W04_RECOVERED_CONTRACT.json',recovered),
                        ('W04_SOURCE_MANIFEST.json',dict(schema='paper2.w04.sources.v1',sources=bind.records,
                          scope='consumed primary sources rehashed; upstream W02/W03 manifests retain other provenance',
                          verifier='Codex machine/source audit; independent human scientific approval not asserted'))]:
        dump(output_dir/name,value)
    bind.unchanged()
    print(json.dumps(summary,indent=2),flush=True)
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir',type=Path,default=HERE)
    main(parser.parse_args().output_dir)
