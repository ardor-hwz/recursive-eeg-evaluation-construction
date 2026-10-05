"""W02: deterministic decomposition of accepted M1 outputs; no model replay.

Use the project's .venv-phase3 Python. All frozen inputs are read-only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
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
FREEZE = EXP / 'm1_lite_implementation_freeze_20260905_v1'
DOWN = EXP / 'corrected_downstream_reconstruction_20260912_v1'
REVIEW = EXP / 'reviewer_mechanistic_controls_20260912_v1'
CORE = EXP / 'stacking_grouping_core_repair_20260912_v1'
STAGE = EXP / 'boundary_stage23_execution_20260924_v1'
PROTOCOL = EXP / 'boundary_stage23_contract_20260924_v1_1/STAGE23_PROTOCOL_v1_1.json'
sys.path[:0] = [str(ROOT), str(FREEZE / 'src')]
from m1_lite.endpoints import LEDGER_KEYS, primary_a, reduce_event_ledger
from m1_lite.inference import bca_mean, BOOTSTRAP_SEED, BOOTSTRAP_RESAMPLES, CONFIDENCE_LEVEL
from src.data.manifest import SUBJECT_SESSIONS
from src.utils.serialization import stable_hash

EPS = 1e-12
VIEWS = ('fixed_amplitude', 'dose_normalized', 'equal_total_dose')
ALIASES = dict(zip(VIEWS, ('fixed', 'dose', 'equal')))
CELL_NAMES = ['A_Full_T', 'A_Full_S', 'A_C0_T', 'A_C0_S']
DERIVED = ['Delta_Full', 'Delta_C0', 'D_S', 'D_T', 'Theta']
SEEDS = (42, 123, 2026)
KEYS = ['outer_subject_id', 'seed', 'session_id', 'time_index']


def require(ok, message):
    if not ok:
        raise ValueError(message)


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n', encoding='utf-8')


def read(path, **kwargs):
    return pd.read_csv(path, float_precision='round_trip', **kwargs)


def error(first, second):
    first, second = np.asarray(first, float), np.asarray(second, float)
    require(first.shape == second.shape, 'Compared array shapes differ')
    require(np.isfinite(first).all() and np.isfinite(second).all(), 'Nonfinite audit value')
    return float(np.max(np.abs(first - second), initial=0.0))


def check_close(first, second, message, tolerance=EPS):
    value = error(first, second)
    require(value <= tolerance, f'{message}: maximum absolute error {value}')
    return value


class Bindings:
    def __init__(self):
        self.records = {}

    def check(self, path, expected=None, authority='inspection snapshot'):
        path = Path(path).resolve()
        actual = sha(path)
        if expected is not None:
            require(actual == expected, f'Frozen source hash mismatch: {path}')
        self.records[str(path)] = dict(sha256=actual, existing_receipt_verified=expected is not None,
                                      binding_authority=str(authority))
        return path

    def mapped(self, path, mapping, authority):
        key = str(Path(path).resolve())
        require(key in mapping, f'Existing binding absent: {key}')
        return self.check(path, mapping[key], authority)

    def unchanged(self):
        require(all(sha(path) == record['sha256'] for path, record in self.records.items()),
                'Consumed frozen source changed during analysis')


def validate_ledger(frame, roots, groups=tuple(range(1, 22)), sessions=SUBJECT_SESSIONS):
    require(set(LEDGER_KEYS + ['A']).issubset(frame.columns), 'Missing primitive ledger fields')
    require(not frame.empty and not frame.duplicated(LEDGER_KEYS).any(), 'Empty or duplicate branch keys')
    require(np.isfinite(frame.A).all() and (frame.A >= 0).all(), 'Invalid integrated absolute response')
    require(set(frame.outer_subject_id) == set(groups), 'Analysis-group coverage changed')
    roots = sorted(tuple(map(int, root)) for root in roots)
    require(len(roots) == len(set(roots)), 'Duplicate root identity')
    require(all(session in sessions[group] and 20 <= t0 <= 615 for group, session, t0 in roots),
            'Root violates group/recording/onset contract')
    observed = frame[['outer_subject_id', 'session_id', 't0']].drop_duplicates()
    require(sorted(map(tuple, observed.to_numpy())) == roots, 'Missing or extra accepted root')
    require(len(frame) == len(roots) * 24, 'Incomplete Cartesian branch coverage')
    require(set(frame.seed) == set(SEEDS) and set(frame.sign) == {-1, 1} and
            set(frame.controller) == {'Full', 'C0'} and set(frame['shape']) == {'transient', 'sustained'},
            'Branch identity changed')
    # With unique keys, correct domains and 24 rows per root, this proves complete pairing.
    counts = frame.groupby(['outer_subject_id', 'session_id', 't0']).size()
    require((counts == 24).all(), 'A root lacks a complete method/shape/sign/seed tuple')
    return frame


def primary_ledger(frame):
    data = frame[LEDGER_KEYS + ['A']].copy()
    # The supportive endpoint is outside W02. Finite placeholders allow the
    # unchanged full reducer to execute; no supportive result is consumed/reported.
    data['B_late'] = 0.0
    data['primary_classification'] = 'PRIMARY_CONFIRMATORY_COMPONENT'
    data['supportive_classification'] = 'SUPPORTIVE_DESCRIPTIVE_ONLY'
    return data


def reduce(frame, roots):
    validate_ledger(frame, roots)
    return reduce_event_ledger(primary_ledger(frame), roots)


def make_view(fixed, equal, view):
    require(view in VIEWS, 'Unknown perturbation view')
    if view == 'equal_total_dose':
        return equal.copy()
    result = fixed.copy()
    if view == 'dose_normalized':
        result.loc[result['shape'].eq('sustained'), 'A'] /= 20.0
    return result


def components(cells, theta):
    require(not cells.duplicated(['outer_subject_id', 'controller', 'shape']).any(), 'Duplicate group cell')
    pivot = cells.pivot(index='outer_subject_id', columns=['controller', 'shape'], values='A').sort_index()
    require(list(pivot.index) == list(range(1, 22)), 'Group tuple must be exactly 1..21')
    require(set(pivot.columns) == {(c, s) for c in ('Full', 'C0') for s in ('transient', 'sustained')},
            'Incomplete primitive group tuple')
    out = pd.DataFrame({'group': pivot.index})
    for name, key in zip(CELL_NAMES, [('Full', 'transient'), ('Full', 'sustained'),
                                     ('C0', 'transient'), ('C0', 'sustained')]):
        out[name] = pivot[key].to_numpy()
    out['Delta_Full'] = out.A_Full_S - out.A_Full_T
    out['Delta_C0'] = out.A_C0_S - out.A_C0_T
    out['D_S'] = out.A_Full_S - out.A_C0_S
    out['D_T'] = out.A_Full_T - out.A_C0_T
    out['Theta'] = np.asarray(theta)
    out['Theta_from_duration'] = out.Delta_Full - out.Delta_C0
    out['Theta_from_method'] = out.D_S - out.D_T
    check_close(out.Theta, out.Theta_from_duration, 'Interaction identity 1 failed')
    check_close(out.Theta, out.Theta_from_method, 'Interaction identity 2 failed')
    return out


def sign_counts(values):
    values = np.asarray(values, float)
    return dict(positive_groups=int((values > EPS).sum()), negative_groups=int((values < -EPS).sum()),
                zero_groups=int((np.abs(values) <= EPS).sum()))


def paired_bootstrap(comp, indices):
    require(indices.ndim == 2 and indices.shape[1] == 21 and indices.min() >= 0 and indices.max() < 21,
            'Invalid group bootstrap indices')
    means = comp[CELL_NAMES + ['Theta']].to_numpy()[indices].mean(axis=1)
    ft, fs, ct, cs, direct = means.T
    duration = (fs - ft) - (cs - ct)
    method = (fs - cs) - (ft - ct)
    check_close(direct, duration, 'Paired bootstrap duration identity failed')
    check_close(direct, method, 'Paired bootstrap method identity failed')
    # Confirm these are exactly the draws used by the frozen SciPy call.
    frozen = bootstrap((comp.Theta.to_numpy(),), np.mean, vectorized=False, paired=False,
                       n_resamples=BOOTSTRAP_RESAMPLES, confidence_level=CONFIDENCE_LEVEL,
                       method='BCa', random_state=np.random.Generator(np.random.PCG64(BOOTSTRAP_SEED)))
    distribution_error = check_close(direct, frozen.bootstrap_distribution, 'Frozen bootstrap draws differ')
    return np.column_stack((direct, duration, method)), dict(
        identity_4_duration_max_abs_error=error(direct, duration),
        identity_4_method_max_abs_error=error(direct, method),
        frozen_scipy_distribution_max_abs_error=distribution_error)


def sign_diagnostics(frame, comp, label):
    branch = frame.groupby(['outer_subject_id', 'session_id', 't0', 'controller', 'shape', 'sign'],
                           as_index=False, sort=True).A.mean()
    session = branch.groupby(['outer_subject_id', 'session_id', 'controller', 'shape', 'sign'],
                             as_index=False, sort=True).A.mean()
    groups = session.groupby(['outer_subject_id', 'controller', 'shape', 'sign'],
                             as_index=False, sort=True).A.mean()
    p = groups.pivot(index=['outer_subject_id', 'controller', 'shape'], columns='sign', values='A').reset_index()
    p = p.rename(columns={-1: 'A_minus', 1: 'A_plus'})
    p['minus_contribution'] = p.A_minus / 2
    p['plus_contribution'] = p.A_plus / 2
    p['A_sign_average'] = p.minus_contribution + p.plus_contribution
    p['relative_asymmetry'] = np.abs(p.A_plus - p.A_minus) / p.A_sign_average
    for row in p.itertuples():
        cell = f'A_{row.controller}_{"S" if row.shape == "sustained" else "T"}'
        check_close([row.A_sign_average], [comp.loc[comp.group.eq(row.outer_subject_id), cell].iloc[0]],
                    'Sign branches fail to reconstruct primitive response')
    bp = branch.pivot(index=['outer_subject_id', 'session_id', 't0', 'controller', 'shape'], columns='sign', values='A')
    mean = (bp[1] + bp[-1]) / 2
    ratio = np.abs(bp[1] - bp[-1]) / mean
    ratio = ratio.fillna(0.0)
    diagnostic = dict(**label, diagnostic_only=True, inferential_family_created=False,
                      group_relative_asymmetry_max=float(p.relative_asymmetry.max()),
                      root_seed_averaged_asymmetry_q95=float(ratio.quantile(.95)),
                      root_seed_averaged_asymmetry_q99=float(ratio.quantile(.99)),
                      root_seed_averaged_asymmetry_max=float(ratio.max()),
                      minus_branch_min=float(bp[-1].min()), plus_branch_min=float(bp[1].min()))
    for key, value in label.items():
        p[key] = value
    return p, diagnostic


def compare_cells(recomputed, archived):
    a = recomputed.sort_values(['outer_subject_id', 'controller', 'shape']).reset_index(drop=True)
    b = archived.sort_values(['outer_subject_id', 'controller', 'shape']).reset_index(drop=True)
    require(a[['outer_subject_id', 'controller', 'shape']].equals(b[['outer_subject_id', 'controller', 'shape']]),
            'Archived group cells do not pair')
    return check_close(a.A, b.A, 'Archived primitive response mismatch')


def root_audit(frame, roots):
    records = []; recovered = []
    for (group, seed, session), unit in frame.groupby(KEYS[:3], sort=True):
        unit = unit.sort_values('time_index')
        require(np.array_equal(unit.time_index, np.arange(885)), 'Incomplete recording indices')
        fast, slow = unit.fast.to_numpy(), unit.slow.to_numpy()
        require(np.isfinite(fast).all() and np.isfinite(slow).all() and
                np.all((fast >= 0) & (fast <= 1) & (slow >= 0) & (slow <= 1)), 'Invalid frozen proxy')
        for t0 in range(20, 616):
            valid = bool(np.all((fast[t0:t0+20] >= .05) & (fast[t0:t0+20] <= .95)))
            records.append(dict(group=int(group), seed=int(seed), session=int(session), t0=t0,
                                eligible_this_seed=valid))
    candidate = pd.DataFrame(records).groupby(['group', 'session', 't0']).eligible_this_seed.all()
    recovered = [tuple(map(int, index)) for index in candidate.index[candidate]]
    require(recovered == sorted(roots), 'Target-free headroom reconstruction differs from frozen roots')
    rows = []
    for (group, session), values in candidate.groupby(level=[0, 1]):
        rows.append(dict(group=group, session=session, structural_roots=len(values),
                         eligible_roots=int(values.sum()), excluded_roots=int((~values).sum()),
                         response_horizon=270, onset_min=20, onset_max=615,
                         overlapping_roots_allowed=True, both_signs_all_seeds_verified=True))
    return rows


def identities(bind, protocol, baseline_map):
    records = []
    for fold in protocol['folds']:
        group = fold['fold']; folder = CORE / 'results' if group in {1, 3, 5, 12, 13} else EXP / 'results'
        lockpath = folder / f'outer_subject_{group:02d}/selection_lock.json'
        bind.check(lockpath, fold['role_lock']['sha256'], PROTOCOL)
        bind.mapped(lockpath, baseline_map, REVIEW / 'reports/BASELINE_BINDINGS.json')
        lock = load(lockpath)
        require(lock['outer_data_loaded_for_selection'] is False and lock['tracker']['uses_outer'] is False,
                'Full configuration lock crosses outer selection boundary')
        if group in {1, 3, 5, 12, 13}:
            cpath = folder / f'outer_subject_{group:02d}/inner_selection/c0/selection.json'
            bind.mapped(cpath, baseline_map, REVIEW / 'reports/BASELINE_BINDINGS.json')
            selection = load(cpath)['selection']
        else:
            cpath = EXP / ('c0_development_selection_execution_retry_20260902_v4/events/'
                           f'C0_DEVSEL_AUTH_20260902T131152Z/locks/outer_subject_{group:02d}.json')
            bind.mapped(cpath, baseline_map, REVIEW / 'reports/BASELINE_BINDINGS.json')
            selection = load(cpath)['base_envelope']['scientific_lock']['selection']
        alpha = selection['selected_alpha']
        for layer, candidate, params, config, source in (
            ('primary_corrected_192', lock['tracker']['candidate_id'], lock['tracker']['parameters'],
             stable_hash(lock['tracker']['parameters']), lockpath),
            ('existing_posthoc_stage1b_771', fold['selected_candidate_id'], fold['selected_parameters'],
             fold['selected_config_sha256'], PROTOCOL)):
            require(stable_hash(params) == config, 'Full configuration digest mismatch')
            records.append(dict(analysis_layer=layer, group=group, Full_candidate_id=candidate,
                                Full_config_sha256=config, Full_parameters_json=json.dumps(params, sort_keys=True),
                                Full_source_lock=str(source), C0_alpha=alpha, C0_source_lock=str(cpath),
                                C0_candidate_id=selection.get('selected_candidate_id', f'alpha_{alpha:.3f}'),
                                fast_candidate_id=lock['proxy_roles']['fast_candidate_id'],
                                slow_candidate_id=lock['proxy_roles']['slow_candidate_id'],
                                recordings_json=json.dumps(SUBJECT_SESSIONS[group])))
    return records


def stage_state_audit(bind, protocol, full_receipt, replay_receipt, c0, input_frame):
    parts = []; audits = []
    constructions = {'fixed_transient': (.05, 1, 'transient'), 'fixed_sustained': (.05, 20, 'sustained'),
                     'equal_dose_sustained': (.0025, 20, 'sustained')}
    for fold in protocol['folds']:
        group = fold['fold']; folder = STAGE / 'm1/folds' / f'outer_subject_{group:02d}'
        receipt_path = folder / 'M1_FOLD_COMPLETE.json'
        bind.mapped(receipt_path, replay_receipt['outputs'], STAGE / 'M1_REPLAY_COMPLETE.json')
        receipt = load(receipt_path)
        require(receipt['status'] == 'COMPLETE' and receipt['config_sha256'] == fold['selected_config_sha256'],
                'Stage M1 Full configuration identity changed')
        require(receipt['root_universe_sha256'] == replay_receipt['inputs']['root_universe'], 'Stage root identity drift')
        event_path = folder / 'event_metrics.csv'
        bind.check(event_path, receipt['outputs'][event_path.name], receipt_path)
        frame = read(event_path, usecols=LEDGER_KEYS + ['A', 'construction', 'config_sha256', 'amplitude', 'duration',
                                                   'branch_sha256', 'root_universe_sha256'])
        require(len(frame) == receipt['event_rows'] and set(frame.outer_subject_id) == {group}, 'Stage event accounting drift')
        require(frame.loc[frame.controller.eq('Full'), 'config_sha256'].eq(fold['selected_config_sha256']).all(),
                'Full branch configuration mismatch')
        require(frame.root_universe_sha256.eq(receipt['root_universe_sha256']).all(), 'Event root binding changed')
        full_path = STAGE / 'stage2_full' / f'outer_subject_{group:02d}/full_trajectory.csv'
        bind.mapped(full_path, full_receipt['outputs'], STAGE / 'STAGE2_FULL_COMPLETE.json')
        full = read(full_path, usecols=KEYS + ['state'])
        clean = full.merge(c0.loc[c0.outer_subject_id.eq(group)], on=KEYS, validate='one_to_one')
        require(len(clean) == 885 * 3 * len(SUBJECT_SESSIONS[group]), 'Clean state alignment incomplete')
        for name, digest in receipt['outputs'].items():
            if not name.endswith('.npz'):
                continue
            match = re.fullmatch(r'branch_seed(\d+)_session(\d+)_sign([+-]\d)_(.+)\.npz', name)
            require(match is not None, 'Unrecognized frozen branch filename')
            seed, session, sign = map(int, match.groups()[:3]); construction = match.group(4)
            amp, duration, shape = constructions[construction]
            path = bind.check(folder / name, digest, receipt_path)
            rows = frame.loc[frame.seed.eq(seed) & frame.session_id.eq(session) & frame.sign.eq(sign) &
                             frame.construction.eq(construction)]
            require(rows.branch_sha256.eq(digest).all() and rows.amplitude.eq(amp).all() and
                    rows.duration.eq(duration).all() and rows['shape'].eq(shape).all(), 'Branch metadata drift')
            unit = clean.loc[clean.seed.eq(seed) & clean.session_id.eq(session)].sort_values('time_index')
            require(np.array_equal(unit.time_index, np.arange(885)), 'Stage clean recording incomplete')
            f = input_frame.loc[input_frame.outer_subject_id.eq(group) & input_frame.seed.eq(seed) &
                               input_frame.session_id.eq(session)].sort_values('time_index').fast.to_numpy()
            with np.load(path, allow_pickle=False) as array:
                roots = array['roots']; require(roots.ndim == 1 and len(roots) == len(np.unique(roots)), 'Invalid state root keys')
                require(float(array['amplitude']) == amp and int(array['duration']) == duration and int(array['sign']) == sign,
                        'State archive perturbation metadata mismatch')
                require(np.all(roots + 270 <= 885), 'State archive crosses recording boundary')
                perturbed_inputs = f[roots[:, None] + np.arange(duration)] + sign * amp
                require(np.all((perturbed_inputs >= 0) & (perturbed_inputs <= 1)), 'Perturbation requires input clipping')
                boundary = 0; maximum = 0.0
                for controller, key, clean_column in [('Full', 'full_state', 'state'), ('C0', 'c0_state', 'c0_state')]:
                    state = array[key]
                    require(state.shape == (len(roots), 270) and np.isfinite(state).all() and
                            np.all((state >= 0) & (state <= 1)), 'Invalid archived perturbed state')
                    baseline = unit[clean_column].to_numpy()[roots[:, None] + np.arange(270)]
                    values = np.abs(state - baseline).sum(axis=1, dtype=np.float64) / .05
                    expected = rows.loc[rows.controller.eq(controller)].set_index('t0').loc[roots, 'A'].to_numpy()
                    # Archived fast scalar accumulator and NumPy sum differ by floating point ordering.
                    maximum = max(maximum, check_close(values, expected, 'State-derived response area drift', 1e-10))
                    for j in sorted({0, len(roots)//2, len(roots)-1}):
                        require(abs(primary_a(baseline[j], state[j], 0) - values[j]) <= EPS, 'Canonical area formula drift')
                    boundary += int(np.count_nonzero((state == 0) | (state == 1)))
                audits.append(dict(group=group, seed=seed, session=session, sign=sign, construction=construction,
                                   roots=len(roots), response_area_rows=2*len(roots), area_max_abs_error=maximum,
                                   boundary_state_points=boundary, input_clipped_points=0, denominator=.05, horizon=270))
        parts.append(frame)
        print(f'Accepted state-area audit: group {group:02d}', flush=True)
    return pd.concat(parts, ignore_index=True), audits


def main(output_dir=HERE):
    output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    bind = Bindings()
    maps = {name: load(bind.check(path)) for name, path in {
        'outputs': REVIEW / 'reports/OUTPUT_HASHES.json', 'baseline': REVIEW / 'reports/BASELINE_BINDINGS.json',
        'sources': REVIEW / 'reports/SOURCE_HASHES.json', 'down': DOWN / 'reports/RESULT_HASHES.json',
        'down_sources': DOWN / 'reports/DOWNSTREAM_IMPLEMENTATION_HASHES.json'}.items()}
    final = load(bind.check(STAGE / 'FINAL_VERIFICATION.json'))
    require(final['status'] == 'PASS' and final['historical_fallback'] is False, 'Current Stage23 release is not accepted')
    final_map_path = bind.check(STAGE / 'FINAL_HASHES.json', final['final_hashes_sha256'], STAGE / 'FINAL_VERIFICATION.json')
    final_map = load(final_map_path)
    stage_receipt = load(bind.check(STAGE / 'M1_COMPLETE.json'))
    replay_path = bind.check(STAGE / 'M1_REPLAY_COMPLETE.json', stage_receipt['inputs']['m1_replay_receipt'], STAGE / 'M1_COMPLETE.json')
    replay_receipt = load(replay_path)
    require(stage_receipt['status'] == 'COMPLETE' and replay_receipt['status'] == 'COMPLETE', 'M1 release incomplete')
    bind.check(PROTOCOL, stage_receipt['protocol_sha256'], STAGE / 'M1_COMPLETE.json')
    bind.mapped(PROTOCOL, final_map, final_map_path)
    protocol = load(PROTOCOL)
    root_contract_path = EXP / 'boundary_stage23_contract_20260924_v1/M1_ROOT_CONTRACT.json'
    bind.check(root_contract_path, protocol['m1_root_contract_sha256'], PROTOCOL)
    root_contract = load(root_contract_path)
    root_path = bind.check(root_contract['primary_root_universe']['path'], root_contract['primary_root_universe']['sha256'], root_contract_path)
    roots = [tuple(map(int, root)) for root in load(root_path)['eligible_roots']]
    require(len(roots) == 13654 and len(set(roots)) == 13654 and roots == sorted(roots), 'Corrected root authority drift')
    frozen_bindings = load(bind.check(FREEZE / 'M1_LITE_SOURCE_BINDINGS.json'))
    frozen_path = bind.check(FREEZE / 'M1_LITE_FROZEN_PROTOCOL.json', frozen_bindings['protocol']['raw_sha256'],
                             FREEZE / 'M1_LITE_SOURCE_BINDINGS.json')
    frozen = load(frozen_path)
    require(frozen['primary_interaction']['formula'] == 'Theta_s=(A_Full,S-A_Full,T)-(A_C0,S-A_C0,T)',
            'Authoritative M1 interaction orientation differs')
    require(frozen['inference']['rng_seed'] == BOOTSTRAP_SEED == 20260905 and
            frozen['inference']['bootstrap_resamples'] == BOOTSTRAP_RESAMPLES == 10000, 'Frozen inference drift')
    for module in ('endpoints.py', 'inference.py', 'event_universe.py', 'contract.py', 'branches.py', 'production.py'):
        bind.mapped(FREEZE / 'src/m1_lite' / module, maps['sources'], REVIEW / 'reports/SOURCE_HASHES.json')
    for source in (REVIEW / 'src/m1_controls.py', ROOT / 'src/models/change_aware_tracker.py'):
        bind.mapped(source, maps['sources'] if 'controls' in source.name else maps['baseline'],
                    REVIEW / 'reports/SOURCE_HASHES.json')
    bind.mapped(DOWN / 'src/m1_reconstruct.py', maps['down_sources'], DOWN / 'reports/DOWNSTREAM_IMPLEMENTATION_HASHES.json')
    for source in (STAGE / '05_m1_replay.py', STAGE / '05b_m1_reduce.py', ROOT / 'src/data/manifest.py'):
        bind.check(source)
    config_rows = identities(bind, protocol, maps['baseline'])
    for ref in (protocol['authoritative_corrected_trajectory'], protocol['corrected_c0_trajectory']):
        bind.check(ref['path'], ref['sha256'], PROTOCOL)
    input_frame = read(protocol['authoritative_corrected_trajectory']['path'], usecols=KEYS + ['fast', 'slow'])
    c0 = read(protocol['corrected_c0_trajectory']['path'], usecols=KEYS + ['c0_state'])
    roots_audit = root_audit(input_frame, roots)
    fixed_path = bind.mapped(DOWN / 'results/m1/branch_results.csv', maps['down'], DOWN / 'reports/RESULT_HASHES.json')
    fixed = read(fixed_path, usecols=LEDGER_KEYS + ['A'])
    equal_path = bind.mapped(REVIEW / 'results/m1/equal_branch_results.csv', maps['outputs'], REVIEW / 'reports/OUTPUT_HASHES.json')
    equal = read(equal_path, usecols=LEDGER_KEYS + ['A'])
    validate_ledger(fixed, roots); validate_ledger(equal, roots)
    alignment = LEDGER_KEYS
    a = fixed.loc[fixed['shape'].eq('transient')].sort_values(alignment).reset_index(drop=True)
    b = equal.loc[equal['shape'].eq('transient')].sort_values(alignment).reset_index(drop=True)
    require(a[alignment].equals(b[alignment]), 'Equal-dose transient pairing drift')
    equal_transient_error = check_close(a.A, b.A, 'Identity 6 primary equal-dose transient drift')
    canonical_checks = []
    for group in (1, 3, 5, 12, 13):
        path = bind.mapped(DOWN / f'results/m1/canonical_verification_{group:02d}.csv', maps['down'], DOWN / 'reports/RESULT_HASHES.json')
        table = read(path); canonical_checks.extend(table.max_trajectory_alpha_error.tolist())
    full_receipt = load(bind.check(STAGE / 'STAGE2_FULL_COMPLETE.json', replay_receipt['inputs']['stage2_full'], replay_path))
    stage_frame, area_audits = stage_state_audit(bind, protocol, full_receipt, replay_receipt, c0, input_frame)
    bind.mapped(STAGE / 'm1_effects.csv', stage_receipt['outputs'], STAGE / 'M1_COMPLETE.json')
    bind.mapped(STAGE / 'm1_group_cells.csv', stage_receipt['outputs'], STAGE / 'M1_COMPLETE.json')
    effects = read(STAGE / 'm1_effects.csv'); archived_stage_cells = read(STAGE / 'm1_group_cells.csv')
    indices = np.random.Generator(np.random.PCG64(BOOTSTRAP_SEED)).integers(0, 21, size=(10000, 21))
    groups_out = []; summaries = []; distributions = []; sign_tables = []; diagnostics = []; algebra = []; draws = []
    identity_rows = []
    strata = [('primary_corrected_192', 'primary_fixed_roots', roots, fixed, equal)]
    stage_fixed = stage_frame.loc[stage_frame.construction.isin(['fixed_transient', 'fixed_sustained'])]
    stage_equal = stage_frame.loc[stage_frame.construction.isin(['fixed_transient', 'equal_dose_sustained'])]
    strata.extend([('existing_posthoc_stage1b_771', universe, subset,
                    stage_fixed.loc[stage_fixed.t0.ge(60)] if 'ge_60' in universe else stage_fixed,
                    stage_equal.loc[stage_equal.t0.ge(60)] if 'ge_60' in universe else stage_equal)
                   for universe, subset in [('primary_fixed_roots', roots),
                                             ('common_complete_history_t0_ge_60', [r for r in roots if r[2] >= 60])]])
    for layer, universe, subset, branch_fixed, branch_equal in strata:
        for view in VIEWS:
            label = dict(analysis_layer=layer, universe=universe, view=view)
            ledger = make_view(branch_fixed, branch_equal, view)
            reduced = reduce(ledger, subset); cells = reduced['subject_primary']; theta = reduced['theta']
            comp = components(cells, theta); ci = bca_mean(theta)
            if layer == 'primary_corrected_192':
                name = ALIASES[view]
                cell_path = bind.mapped(REVIEW / f'results/m1/{name}_subject_cells.csv', maps['outputs'], REVIEW / 'reports/OUTPUT_HASHES.json')
                theta_path = bind.mapped(REVIEW / f'results/m1/{name}_theta.csv', maps['outputs'], REVIEW / 'reports/OUTPUT_HASHES.json')
                ci_path = bind.mapped(REVIEW / f'results/m1/{name}_inference.json', maps['outputs'], REVIEW / 'reports/OUTPUT_HASHES.json')
                cell_error = compare_cells(cells, read(cell_path)); theta_error = check_close(theta, read(theta_path).theta, 'Archived primary Theta drift')
                original_ci = load(ci_path)
                for key, value in [('rng_seed', 20260905), ('bootstrap_resamples', 10000), ('subject_count', 21)]:
                    require(original_ci[key] == value, 'Archived primary inference contract drift')
                ci_error = check_close([ci[k] for k in ('estimate', 'lower', 'upper')],
                                       [original_ci[k] for k in ('estimate', 'lower', 'upper')], 'Archived primary BCa drift')
            else:
                archived = archived_stage_cells.loc[archived_stage_cells.universe.eq(universe) &
                                                     archived_stage_cells.construction.eq(view)]
                cell_error = compare_cells(cells, archived)
                rows = effects.loc[effects.universe.eq(universe) & effects.construction.eq(view)]
                group_rows = rows.loc[rows.row_level.eq('group_interaction')].sort_values('fold')
                population = rows.loc[rows.row_level.eq('population_interaction')].iloc[0]
                theta_error = check_close(theta, group_rows.theta, 'Archived Stage Theta drift')
                ci_error = check_close([ci['estimate'], ci['lower'], ci['upper']],
                                       [population.theta, population.lower_bca, population.upper_bca], 'Archived Stage BCa drift')
                require(sign_counts(theta) == dict(positive_groups=int(population.positive_group_count),
                                                   negative_groups=int(population.negative_group_count),
                                                   zero_groups=int(population.zero_group_count)), 'Archived Stage sign counts drift')
            draw, boot_errors = paired_bootstrap(comp, indices); draws.append(draw)
            sign_table, diagnostic = sign_diagnostics(ledger, comp, label)
            sign_tables.append(sign_table); diagnostics.append(diagnostic)
            mean = comp[CELL_NAMES + DERIVED].mean()
            summaries.append(dict(**label, N=21, roots=len(subset), **mean.to_dict(), lower_bca=ci['lower'], upper_bca=ci['upper'],
                                  **sign_counts(theta), interval_classification='positive' if ci['lower'] > 0 else 'negative' if ci['upper'] < 0 else 'spans_zero',
                                  interval_scope='existing pointwise 95% BCa for Theta only',
                                  bootstrap_seed=BOOTSTRAP_SEED, bootstrap_resamples=BOOTSTRAP_RESAMPLES))
            identity_rows.append(dict(**label, primitive_cells_max_abs_error=cell_error, archived_theta_max_abs_error=theta_error,
                                      archived_BCa_max_abs_error=ci_error, identity_3_population_max_abs_error=abs(float(mean.Theta)-ci['estimate']),
                                      **boot_errors))
            for row in comp.itertuples():
                algebra.append(dict(**label, group=row.group, identity_1_abs_error=abs(row.Theta-row.Theta_from_duration),
                                    identity_2_abs_error=abs(row.Theta-row.Theta_from_method)))
            for name in CELL_NAMES + DERIVED:
                values = comp[name].to_numpy()
                distributions.append(dict(**label, quantity=name, mean=float(np.mean(values)), min=float(np.min(values)),
                                          q1=float(np.quantile(values,.25)), median=float(np.median(values)),
                                          q3=float(np.quantile(values,.75)), max=float(np.max(values)),
                                          sd_between_groups=float(np.std(values,ddof=1)), **sign_counts(values),
                                          status='descriptive group distribution; no new interval or test'))
            for key, value in label.items(): comp[key] = value
            groups_out.append(comp)
            print('Decomposition accepted:', layer, universe, view, f'Theta={ci["estimate"]:.12g}', flush=True)
    group_table = pd.concat(groups_out, ignore_index=True)
    deterministic = []
    for layer, universe in group_table[['analysis_layer', 'universe']].drop_duplicates().itertuples(index=False):
        select = group_table.loc[group_table.analysis_layer.eq(layer) & group_table.universe.eq(universe)]
        f, d, e = [select.loc[select.view.eq(v)].sort_values('group') for v in VIEWS]
        for name in ('A_Full_S', 'A_C0_S'):
            check_close(f[name]/20, d[name], 'Identity 5 sustained normalization failed')
        for name in ('A_Full_T', 'A_C0_T'):
            check_close(f[name], d[name], 'Identity 5 transient normalization failed')
            check_close(f[name], e[name], 'Identity 6 equal-dose transient failed')
        correct_theta = f.D_S.to_numpy()/20 - f.D_T.to_numpy()
        deterministic.append(dict(analysis_layer=layer, universe=universe,
                                  identity_5_cell_max_abs_error=max(error(f.A_Full_S/20,d.A_Full_S),error(f.A_C0_S/20,d.A_C0_S)),
                                  identity_5_theta_max_abs_error=check_close(d.Theta,correct_theta,'Identity 5 final contrast drift'),
                                  identity_6_transient_max_abs_error=max(error(f.A_Full_T,e.A_Full_T),error(f.A_C0_T,e.A_C0_T)),
                                  invalid_theta_divide_20_max_abs_discrepancy=error(d.Theta,f.Theta/20)))
    bind.unchanged()
    contract = dict(schema='paper2.w02.recovered_contract.v1', workflow=2, question='Primitive M1 interaction decomposition and algebraic integrity',
                    primary_analysis_layer='primary_corrected_192', separately_retained_context='existing_posthoc_stage1b_771',
                    superseded_numerical_branch='pre-stacking-correction M1 Theta about 1.658920; not analyzed',
                    current_authority=str(ROOT / 'PAPER2_EVIDENCE_FOR_REFRAMING.md'),
                    formula=frozen['primary_interaction'], endpoint=frozen['endpoints']['primary'],
                    intervention=root_contract['intervention'], timing=frozen['timing'], reducer=frozen['reducers'],
                    independent_unit='21 filename-defined analysis groups; provider-authenticated biological identity unresolved',
                    group_recording_mapping={str(k):v for k,v in SUBJECT_SESSIONS.items()}, roots=13654, structural_roots=13708,
                    excluded_roots=54, overlapping_roots_allowed=True, stage_existing_mature_roots=12734,
                    inference=frozen['inference'], normalized_area_unit='released feature-sequence indices; no conversion to seconds',
                    population_aggregation='equal arithmetic mean of 21 complete group interactions',
                    scientific_status='fixed original protocol-designated primary interaction; W02 decomposition retrospective; other views/posthoc strata exploratory',
                    component_intervals='not computed; descriptive point estimates and group distributions only',
                    target_columns_read=False, upstream_refit=False, model_replay=False,
                    workflows_3_4_5_executed=False, independent_dataset_experiment=False)
    integrity = dict(status='PASS', hard_gate=None, tolerance=EPS, state_area_tolerance=1e-10,
                     group_identity_audit=identity_rows, deterministic_view_identities=deterministic,
                     primary_equal_transient_branch_max_abs_error=equal_transient_error,
                     primary_area_audit=dict(level='accepted event areas plus frozen formula/source and canonical-check receipts',
                                             complete_perturbed_state_arrays_available=False, new_replay_performed=False,
                                             current_area_recalculation_from_states='unavailable without forbidden replay',
                                             corrected_fold_canonical_state_checks=len(canonical_checks),
                                             corrected_fold_canonical_state_max_abs_error=max(canonical_checks)),
                     stage_area_audit=dict(level='all accepted perturbed state arrays; target-free clean states',
                                          archives=len(area_audits), area_rows=sum(r['response_area_rows'] for r in area_audits),
                                          max_abs_error=max(r['area_max_abs_error'] for r in area_audits),
                                          boundary_state_points=sum(r['boundary_state_points'] for r in area_audits),
                                          input_clipped_points=0), consumed_frozen_sources_unchanged=True,
                     new_scientific_replays=0, inference_N=21, new_p_values=0, new_component_CIs=0,
                     direct_perturbation_construction_inference=False, search_audit=False)
    for name, frame in [('group_components',group_table),('population_summary',pd.DataFrame(summaries)),
                        ('group_distributions',pd.DataFrame(distributions)),('sign_branch_groups',pd.concat(sign_tables,ignore_index=True)),
                        ('sign_diagnostics',pd.DataFrame(diagnostics)),('algebraic_group_audit',pd.DataFrame(algebra)),
                        ('frozen_config_identity',pd.DataFrame(config_rows)),('root_audit',pd.DataFrame(roots_audit)),
                        ('response_area_audit',pd.DataFrame(area_audits))]:
        frame.to_csv(output_dir / f'W02_{name}.csv', index=False, float_format='%.17g')
    np.save(output_dir/'W02_bootstrap_group_indices.npy',indices,allow_pickle=False)
    np.save(output_dir/'W02_bootstrap_identity_draws.npy',np.stack(draws),allow_pickle=False)
    dump(output_dir/'W02_frozen_contract.json',contract)
    dump(output_dir/'W02_integrity_audit.json',integrity)
    dump(output_dir/'W02_source_manifest.json',dict(schema='paper2.w02.source_manifest.v1', sources=bind.records,
                                                 verifier='Codex local automated verification; human verification not claimed'))
    dump(output_dir/'W02_summary.json',dict(status='PASS', workflow=2, scientific_contract=contract,
                                          results=summaries, integrity=integrity,
                                          bootstrap_draw_order=[{k:r[k] for k in ('analysis_layer','universe','view')} for r in summaries],
                                          bootstrap_draw_columns=['Theta_direct','Theta_duration','Theta_method'],
                                          software=dict(python=sys.version.split()[0],numpy=np.__version__,pandas=pd.__version__,scipy=scipy.__version__),
                                          request_attachment_ends_with_incomplete_section_12=True))
    print('W02 ANALYSIS COMPLETE: accepted outputs only; no replay; stop at Workflow 2.',flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output-dir',type=Path,default=HERE)
    main(parser.parse_args().output_dir)
