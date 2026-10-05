"""W03: accepted evidence recovery and descriptive sign transitions only."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
W01 = ROOT / 'codex_workspace/W01_C0_vs_Cmean'
W02 = ROOT / 'codex_workspace/W02_M1_decomposition'
EPS = 1e-12
GROUPS = set(range(1, 22))
BRANCHES = ('primary_corrected_192', 'existing_posthoc_stage1b_771')
VIEWS = ('fixed_amplitude', 'dose_normalized', 'equal_total_dose')
SIGNS = ('+', '−', '0')
SCIENTIFIC_FILES = (
    'W03_CLAIM_STATE_TABLE.csv', 'W03_RETENTION_TABLE.csv',
    'W03_GROUP_TRANSITIONS.csv', 'W03_TRANSITION_MATRICES.json',
    'W03_HETEROGENEITY.csv', 'W03_REPORT_AGREEMENT.csv',
    'W03_SOURCE_MANIFEST.json', 'W03_integrity_audit.json',
)


def require(ok, message):
    if not ok:
        raise ValueError('SCIENTIFIC HARD GATE: ' + message)


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def dump(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True,
                                    allow_nan=False) + '\n', encoding='utf-8')


def read(path):
    return pd.read_csv(path, float_precision='round_trip')


def sign(value):
    require(math.isfinite(float(value)), 'nonfinite value')
    return '+' if value > EPS else '−' if value < -EPS else '0'


def direction(value):
    return {'+': 'POSITIVE', '−': 'NEGATIVE', '0': 'ZERO'}[sign(value)]


def support(low, high):
    if low is None and high is None:
        return 'NO_INTERVAL_DEFINED'
    require(low is not None and high is not None and math.isfinite(low) and
            math.isfinite(high) and low <= high, 'invalid interval')
    return 'POSITIVE_SUPPORTED' if low > 0 else 'NEGATIVE_SUPPORTED' if high < 0 else 'SPANS_ZERO'


def majority(positive, negative, zero):
    n = positive + negative + zero
    require(n == 21, 'majority must use 21 groups')
    if positive == n:
        return 'ALL_POSITIVE'
    if negative == n:
        return 'ALL_NEGATIVE'
    return 'POSITIVE_MAJORITY' if positive > negative else 'NEGATIVE_MAJORITY' if negative > positive else 'TIED_MAJORITY'


def majority_polarity(category):
    return 1 if category in ('ALL_POSITIVE', 'POSITIVE_MAJORITY') else -1 if category in ('ALL_NEGATIVE', 'NEGATIVE_MAJORITY') else 0


def validate_groups(frame, group_column='group'):
    values = frame[group_column].tolist()
    require(len(values) == 21 and len(set(values)) == 21 and set(values) == GROUPS,
            'missing, duplicate or incorrect matched groups')
    return frame.sort_values(group_column).reset_index(drop=True)


def equal_number(actual, expected, label, tolerance=EPS):
    require(math.isfinite(float(actual)) and math.isfinite(float(expected)) and
            abs(float(actual) - float(expected)) <= tolerance, label)


class Sources:
    def __init__(self):
        self.records = {}

    def add(self, path, expected=None, authority='W03 inspection snapshot'):
        path = Path(path).resolve()
        digest = sha(path)
        if expected is not None:
            require(digest == expected, 'accepted source hash drift: ' + str(path))
        if str(path) in self.records:
            require(self.records[str(path)]['sha256'] == digest, 'source changed during recovery')
        self.records[str(path)] = dict(sha256=digest, bytes=path.stat().st_size,
                                      binding_authority=str(authority), existing_hash_verified=expected is not None)

    def unchanged(self):
        for path, record in self.records.items():
            require(sha(path) == record['sha256'], 'consumed source modified: ' + path)


def bind_sources():
    sources = Sources()
    for path in (W01/'W01_acceptance.json', W01/'W01_reproducibility.json',
                 W02/'W02_acceptance.json', W02/'W02_reproducibility.json', W02/'W02_delivery_manifest.json'):
        sources.add(path)
        require(load(path)['status'] == 'PASS', 'upstream acceptance/reproducibility not PASS')
    for filename, digest in load(W01/'W01_reproducibility.json')['output_sha256'].items():
        sources.add(W01/filename, digest, W01/'W01_reproducibility.json')
    for filename, record in load(W02/'W02_delivery_manifest.json')['files'].items():
        sources.add(W02/filename, record['sha256'], W02/'W02_delivery_manifest.json')
    for filename, digest in load(W02/'W02_reproducibility.json')['original_hashes'].items():
        sources.add(W02/filename, digest, W02/'W02_reproducibility.json')
    a1 = load(W01/'W01_acceptance.json')
    sources.add(W01/'analyze_w01.py', a1['final_analysis_sha256'], W01/'W01_acceptance.json')
    sources.add(W01/'test_w01.py', a1['test_script_sha256'], W01/'W01_acceptance.json')
    for path in (W01/'W01_SCIENTIFIC_CONTRACT.md', W01/'W01_C0_VS_CMEAN_REPORT.md',
                 ROOT/'PAPER2_EVIDENCE_FOR_REFRAMING.md', ROOT/'PAPER2_FINAL_SCIENTIFIC_CONSTITUTION.md',
                 HERE/'W03_SCIENTIFIC_CONTRACT.md'):
        sources.add(path)
    upstream = load(W02/'W02_source_manifest.json')['sources']
    for name in ('manifest.py', 'M1_ROOT_CONTRACT.json', 'STAGE23_PROTOCOL_v1_1.json'):
        matches = [(p, r) for p, r in upstream.items() if Path(p).name == name]
        require(len(matches) == 1, 'ambiguous current contract source ' + name)
        path, record = matches[0]
        sources.add(path, record['sha256'], W02/'W02_source_manifest.json')
    require(load(W01/'W01_summary.json')['statistical_rule']['sign_tolerance'] == EPS and
            load(W02/'W02_integrity_audit.json')['tolerance'] == EPS, 'accepted sign tolerance mismatch')
    tree = ast.parse((ROOT/'src/data/manifest.py').read_text(encoding='utf-8'))
    mapping = next(ast.literal_eval(node.value) for node in tree.body
                   if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)
                   and node.target.id == 'SUBJECT_SESSIONS')
    recovered = {int(k): v for k, v in load(W02/'W02_frozen_contract.json')['group_recording_mapping'].items()}
    require(mapping == recovered and set(mapping) == GROUPS, 'current accepted group mapping drift')
    c1, c2 = read(W01/'W01_frozen_config_identity.csv'), read(W02/'W02_frozen_config_identity.csv')
    identity_columns = ['Full_candidate_id', 'Full_config_sha256', 'Full_parameters_json',
                        'fast_candidate_id', 'slow_candidate_id', 'C0_alpha']
    for branch in BRANCHES:
        a = validate_groups(c1[c1.analysis_layer.eq(branch)])
        b = validate_groups(c2[c2.analysis_layer.eq(branch)])
        for column in identity_columns:
            require(a[column].tolist() == b[column].tolist(), 'W01/W02 branch identity mismatch ' + column)
    return sources


def report_numbers(cell):
    return re.findall(r'[+−-]?\d+\.\d+(?:[eE][+-]?\d+)?', cell)


def check_printed(actual, printed, label):
    token = printed.replace('−', '-')
    digits = len(token.split('.')[1]) if 'e' not in token.lower() else 9
    equal_number(actual, float(token), 'accepted machine/report disagreement: '+label,
                 0.5 * 10**(-digits) + 1e-15)
    return dict(label=label, accepted_value=float(actual), report_value=float(token),
                printed_decimal_places=digits, absolute_error=abs(float(actual)-float(token)), status='PASS')


def check_reports(t1, t2):
    checks = []
    branch = None
    effect_table = False
    w01_rows = 0
    for line in (W01/'W01_C0_VS_CMEAN_REPORT.md').read_text(encoding='utf-8').splitlines():
        if line.startswith('## Corrected original'):
            branch = BRANCHES[0]
        if line.startswith('## Existing Stage-1b'):
            branch = BRANCHES[1]
        if line.startswith('| Endpoint/view | Full−C0'):
            effect_table = True
            continue
        if not line.startswith('| '):
            effect_table = False
        if not line.startswith('| ') or branch is None:
            continue
        cells = [x.strip() for x in line.strip('|').split('|')]
        if not effect_table or len(cells) != 5 or '/' not in cells[0] or not cells[0].startswith(('RMSE/', 'MAFD/')):
            continue
        endpoint, weighting = cells[0].split('/')
        construction = 'Cmean_primary' if weighting == 'primary' else 'Cmean_pooled'
        row = t1[(t1.analysis_layer.eq(branch)) & (t1.endpoint.eq(endpoint.lower())) &
                 (t1.cmean_definition.eq(construction))]
        require(len(row) == 1, 'W01 report identity mismatch')
        row = row.iloc[0]
        for cell_index, prefix in ((1, 'Full_vs_C0'), (3, 'Full_vs_Cmean')):
            tokens = report_numbers(cells[cell_index])
            require(len(tokens) == 3, 'W01 report parse failed')
            for field, token in zip(('mean', 'lower', 'upper'), tokens):
                checks.append(check_printed(row[prefix+'_'+field], token, f'{branch}/{endpoint}/{construction}/{prefix}/{field}'))
            counts = '/'.join(str(int(row[prefix+'_'+name+'_groups'])) for name in ('positive', 'negative', 'zero'))
            require(cells[cell_index+1] == counts, 'W01 report sign-count disagreement')
        w01_rows += 1
    require(w01_rows == 8, 'W01 accepted report rows missing')
    header = ''
    w02_rows = 0
    for line in (W02/'W02_M1_DECOMPOSITION_REPORT.md').read_text(encoding='utf-8').splitlines():
        if line.startswith('| view | Theta |'):
            header = 'primary'
            continue
        if line.startswith('| universe | view | D_S |'):
            header = 'stage'
            continue
        if not line.startswith('| '):
            header = ''
            continue
        cells = [x.strip() for x in line.strip('|').split('|')]
        if header == 'primary' and len(cells) == 5 and cells[0] in VIEWS:
            branch, universe, view = BRANCHES[0], 'primary_fixed_roots', cells[0]
            estimate, interval, counts = cells[1:4]
        elif header == 'stage' and len(cells) == 7 and cells[1] in VIEWS:
            branch, universe, view = BRANCHES[1], cells[0], cells[1]
            estimate, interval, counts = cells[4:7]
        else:
            continue
        found = t2[t2.analysis_layer.eq(branch) & t2.universe.eq(universe) & t2.view.eq(view)]
        require(len(found) == 1, 'W02 report identity mismatch')
        row = found.iloc[0]
        tokens = report_numbers(estimate) + report_numbers(interval)
        require(len(tokens) == 3, 'W02 report parse failed')
        for field, token in zip(('Theta', 'lower_bca', 'upper_bca'), tokens):
            checks.append(check_printed(row[field], token, f'{branch}/{universe}/{view}/{field}'))
        expected = '/'.join(str(int(row[name+'_groups'])) for name in ('positive', 'negative', 'zero'))
        require(counts == expected, 'W02 report sign-count disagreement')
        w02_rows += 1
    require(w02_rows == 9, 'W02 accepted report rows missing')
    return checks


def make_claim(branch, family, universe, endpoint, construction, values, accepted_mean, low, high, counts,
               group_source, summary_source):
    values = validate_groups(values)
    require(np.isfinite(values.value).all(), 'nonfinite recovered group vector')
    vector = values.value.to_numpy()
    equal_number(float(np.mean(vector)), accepted_mean, 'accepted group mean not reproduced')
    labels = [sign(x) for x in vector]
    actual_counts = [labels.count(s) for s in SIGNS]
    require(actual_counts == list(counts), 'accepted sign counts not reproduced')
    pos, neg, zero = actual_counts
    claim_id = f'{branch}__{family}__{universe}__{endpoint}__{construction}'
    state = dict(claim_id=claim_id, branch=branch, family=family, universe=universe,
                 endpoint_or_view=construction if family == 'M1' else endpoint, endpoint=endpoint,
                 construction=construction, population_mean=float(accepted_mean), mean_direction=direction(accepted_mean),
                 ci_low=float(low), ci_high=float(high), ci_support_category=support(low, high),
                 positive_groups=pos, negative_groups=neg, zero_groups=zero,
                 majority_direction=majority(pos, neg, zero), N_groups=21,
                 group_vector_source=str(group_source), interval_source=str(summary_source),
                 scope='corrected original' if branch == BRANCHES[0] else 'existing post hoc Stage-1b',
                 weighting_role='sensitivity' if construction == 'Cmean_pooled' else 'main within branch')
    q1, median, q3 = map(float, np.quantile(vector, [.25, .5, .75], method='linear'))
    iqr = q3-q1
    margin = abs(pos-neg)
    flips_needed = margin//2+1 if pos != neg else (1 if pos+neg else None)
    hetero = dict(claim_id=claim_id, branch=branch, family=family, universe=universe,
                  endpoint_or_view=state['endpoint_or_view'], construction=construction, N_groups=21,
                  mean=float(np.mean(vector)), min=float(np.min(vector)), Q1=q1, median=median, Q3=q3,
                  max=float(np.max(vector)), IQR=iqr, median_absolute_deviation=float(np.median(np.abs(vector-median))),
                  positive_groups=pos, negative_groups=neg, zero_groups=zero,
                  mean_direction=state['mean_direction'], median_direction=direction(median),
                  mean_median_direction_disagreement=state['mean_direction'] != direction(median),
                  directional_mean_with_ci_spanning_zero=state['mean_direction'] != 'ZERO' and state['ci_support_category']=='SPANS_ZERO',
                  majority_margin=margin, minimum_nonzero_sign_flips_to_opposite_majority=flips_needed,
                  one_sign_flip_reverses_majority=flips_needed == 1,
                  abs_mean_divided_by_IQR=abs(float(accepted_mean))/iqr if iqr>0 else None,
                  positive_value_sum=float(sum(v for v,s in zip(vector, labels) if s=='+')),
                  negative_value_sum=float(sum(v for v,s in zip(vector, labels) if s=='−')),
                  largest_absolute_value_group=int(values.iloc[int(np.argmax(np.abs(vector)))].group))
    return state, hetero, {int(g):float(v) for g,v in zip(values.group, values.value)}


def compare(source, destination, source_vector, destination_vector, role):
    require(set(source_vector) == set(destination_vector) == GROUPS, 'unmatched comparison groups')
    require(all(source[k] == destination[k] for k in ('branch','family','universe','endpoint')),
            'cross-branch/endpoint/root-support comparison prohibited')
    comparison_id = source['claim_id']+'__TO__'+destination['construction']
    matrix = [[0]*3 for _ in range(3)]
    rows = []
    retained, flipped, zero_ids = [], [], []
    for group in sorted(GROUPS):
        a, b = source_vector[group], destination_vector[group]
        sa, sb = sign(a), sign(b)
        matrix[SIGNS.index(sa)][SIGNS.index(sb)] += 1
        bucket = 'any_zero' if '0' in (sa,sb) else 'retained_nonzero' if sa==sb else 'flipped_nonzero'
        {'retained_nonzero':retained,'flipped_nonzero':flipped,'any_zero':zero_ids}[bucket].append(group)
        rows.append(dict(comparison_id=comparison_id, branch=source['branch'], family=source['family'],
                         universe=source['universe'], endpoint=source['endpoint'], role=role,
                         source_construction=source['construction'], destination_construction=destination['construction'],
                         group_id=group, source_value=a, source_sign=sa, destination_value=b, destination_sign=sb,
                         transition=sa+'→'+sb, sign_retained=sa==sb, accounting_bucket=bucket))
    reversal = {source['mean_direction'],destination['mean_direction']} == {'POSITIVE','NEGATIVE'}
    ci_retained = 'NOT APPLICABLE' if 'NO_INTERVAL_DEFINED' in (source['ci_support_category'],destination['ci_support_category']) else 'YES' if source['ci_support_category']==destination['ci_support_category'] else 'NO'
    pair = dict(comparison_id=comparison_id, branch=source['branch'], family=source['family'],
                universe=source['universe'], endpoint=source['endpoint'], role=role,
                source_construction=source['construction'], destination_construction=destination['construction'],
                source_claim_id=source['claim_id'], destination_claim_id=destination['claim_id'],
                mean_direction_retained='YES' if source['mean_direction']==destination['mean_direction'] else 'NO',
                ci_support_retained=ci_retained,
                majority_direction_retained='YES' if majority_polarity(source['majority_direction'])==majority_polarity(destination['majority_direction']) else 'NO',
                majority_category_retained='YES' if source['majority_direction']==destination['majority_direction'] else 'NO',
                matched_sign_retained=len(retained), matched_sign_flipped=len(flipped), transitions_via_zero=len(zero_ids),
                matched_state_retained_including_zero=len(retained)+matrix[2][2],
                retained_sign_proportion=len(retained)/21, N_groups=21,
                flip_group_ids=json.dumps(flipped), retained_nonzero_group_ids=json.dumps(retained),
                zero_transition_group_ids=json.dumps(zero_ids),
                population_nonzero_direction_reversed=reversal,
                population_reversal_resisting_group_ids=json.dumps(retained) if reversal else 'NOT APPLICABLE')
    require(sum(map(sum,matrix)) == len(rows) == len(retained)+len(flipped)+len(zero_ids) == 21,
            'transition accounting failed')
    require([sum(x) for x in matrix] == [source[k] for k in ('positive_groups','negative_groups','zero_groups')], 'source transition marginals')
    require([sum(matrix[i][j] for i in range(3)) for j in range(3)] == [destination[k] for k in ('positive_groups','negative_groups','zero_groups')], 'destination transition marginals')
    for i,sa in enumerate(SIGNS):
        for j,sb in enumerate(SIGNS):
            pair['count_'+{'+' :'positive','−':'negative','0':'zero'}[sa]+'_to_'+{'+' :'positive','−':'negative','0':'zero'}[sb]] = matrix[i][j]
    m = dict(comparison_id=comparison_id, branch=source['branch'], family=source['family'],
             universe=source['universe'], endpoint=source['endpoint'], role=role,
             source_construction=source['construction'], destination_construction=destination['construction'],
             row_signs=list(SIGNS), column_signs=list(SIGNS), counts=matrix, N_groups=21,
             flip_group_ids=flipped, retained_nonzero_group_ids=retained,
             population_reversal_resisting_group_ids=retained if reversal else None)
    return pair, rows, m


def run(output_dir):
    output_dir = Path(output_dir).resolve()
    require(output_dir == HERE or output_dir == HERE/'reproducibility_run', 'outputs must stay in W03')
    output_dir.mkdir(parents=True, exist_ok=True)
    sources = bind_sources()
    t1, g1, metrics = read(W01/'W01_result_table.csv'), read(W01/'W01_paired_group_contrasts.csv'), read(W01/'W01_group_metrics.csv')
    t2, g2 = read(W02/'W02_population_summary.csv'), read(W02/'W02_group_components.csv')
    require(len(t1)==8 and len(g1)==168 and len(t2)==9 and len(g2)==189, 'accepted input cardinality drift')
    require(t1.N.eq(21).all() and t2.N.eq(21).all(), 'accepted independent N drift')
    require(set(t1.analysis_layer)==set(t2.analysis_layer)==set(BRANCHES), 'accepted branch identities drift')
    require(set(t1.endpoint)=={'rmse','mafd'} and set(t2.view)==set(VIEWS), 'accepted endpoint/view drift')
    expected_strata = {(BRANCHES[0],'primary_fixed_roots'),(BRANCHES[1],'primary_fixed_roots'),
                       (BRANCHES[1],'common_complete_history_t0_ge_60')}
    require(set(zip(t2.analysis_layer,t2.universe))==expected_strata, 'M1 root support identity drift')
    j1, j2 = load(W01/'W01_summary.json')['results'], load(W02/'W02_summary.json')['results']
    for frame, archived, keys in ((t1,j1,['analysis_layer','cmean_definition','endpoint']),
                                 (t2,j2,['analysis_layer','universe','view'])):
        require(not frame.duplicated(keys).any(), 'duplicate population summary')
        indexed = {tuple(row[k] for k in keys):row for row in archived}
        for row in frame.to_dict('records'):
            reference = indexed[tuple(row[k] for k in keys)]
            require(all(row[k] == reference[k] for k in row), 'accepted CSV/JSON disagreement')
    report_checks = check_reports(t1,t2)
    states, heterogeneity, vectors, pairs, transitions, matrices = {}, {}, {}, [], [], []
    w01_pairs = []
    def add_claim(*args):
        state, hetero, vector = make_claim(*args)
        cid = state['claim_id']
        if cid in states:
            require(states[cid]==state and vectors[cid]==vector, 'same C0 claim differs by Cmean weighting')
        states[cid],heterogeneity[cid],vectors[cid] = state,hetero,vector
        return state
    def add_pair(a,b,role):
        pair, rows, matrix = compare(a,b,vectors[a['claim_id']],vectors[b['claim_id']],role)
        pairs.append(pair); transitions.extend(rows); matrices.append(matrix)
        return pair
    for row in t1.to_dict('records'):
        branch, endpoint, construction = row['analysis_layer'],row['endpoint'],row['cmean_definition']
        groups = validate_groups(g1[g1.analysis_layer.eq(branch) & g1.endpoint.eq(endpoint) & g1.cmean_definition.eq(construction)])
        for method in ('Full','C0',construction):
            m = validate_groups(metrics[metrics.analysis_layer.eq(branch) & metrics.method.eq(method)],'outer_subject_id')
            accepted_column = method if method in ('Full','C0') else 'Cmean'
            require(groups[accepted_column].tolist()==m[endpoint].tolist(), 'W01 group metric identity mismatch')
        for comparator, effect in (('C0','Full_vs_C0'),('Cmean','Full_vs_Cmean')):
            require(np.all(np.abs(groups[effect]-(groups.Full-groups[comparator]))<=EPS), 'W01 Full-minus-comparator orientation mismatch')
        def claim(construction_name,prefix):
            return add_claim(branch,'comparator','accepted_outer',endpoint,construction_name,
                             groups[['group',prefix]].rename(columns={prefix:'value'}), row[prefix+'_mean'],
                             row[prefix+'_lower'],row[prefix+'_upper'],
                             [int(row[prefix+'_'+s+'_groups']) for s in ('positive','negative','zero')],
                             W01/'W01_paired_group_contrasts.csv',W01/'W01_result_table.csv')
        a,b = claim('C0','Full_vs_C0'),claim(construction,'Full_vs_Cmean')
        pair = add_pair(a,b,'main within branch' if construction=='Cmean_primary' else 'pooled weighting sensitivity')
        for field, accepted in (('mean_direction_retained','mean_direction_retained'),('ci_support_retained','ci_support_retained'),
                                ('majority_direction_retained','group_majority_retained')):
            require((pair[field]=='YES')==row[accepted], 'accepted W01 retention disagreement '+field)
        require(pair['matched_sign_retained']==row['group_same_sign'] and pair['matched_sign_flipped']==row['group_opposite_sign'] and
                pair['transitions_via_zero']==row['group_zero_transition'], 'accepted W01 sign retention not reproduced')
        w01_pairs.append(pair)
    m1_states = {}
    for row in t2.to_dict('records'):
        branch,universe,view = row['analysis_layer'],row['universe'],row['view']
        groups = validate_groups(g2[g2.analysis_layer.eq(branch) & g2.universe.eq(universe) & g2.view.eq(view)])
        require(np.all(np.abs(groups.Theta-((groups.A_Full_S-groups.A_Full_T)-(groups.A_C0_S-groups.A_C0_T)))<=EPS),
                'M1 Theta orientation mismatch')
        state = add_claim(branch,'M1',universe,'Theta',view,groups[['group','Theta']].rename(columns={'Theta':'value'}),
                          row['Theta'],row['lower_bca'],row['upper_bca'],
                          [int(row[s+'_groups']) for s in ('positive','negative','zero')],
                          W02/'W02_group_components.csv',W02/'W02_population_summary.csv')
        m1_states[(branch,universe,view)] = state
    for branch,universe in sorted(expected_strata):
        for source,dest in ((VIEWS[0],VIEWS[1]),(VIEWS[0],VIEWS[2]),(VIEWS[1],VIEWS[2])):
            add_pair(m1_states[(branch,universe,source)],m1_states[(branch,universe,dest)],
                     'main corrected M1' if branch==BRANCHES[0] else 'existing post hoc M1 context')
    require(len(states)==21 and len(pairs)==17 and len(transitions)==357 and len(matrices)==17,
            'W03 frozen output cardinality mismatch')
    require(len({p['comparison_id'] for p in pairs})==17, 'duplicate comparisons')
    for filename,data in (('W03_CLAIM_STATE_TABLE.csv',list(states.values())),('W03_HETEROGENEITY.csv',list(heterogeneity.values())),
                          ('W03_RETENTION_TABLE.csv',pairs),('W03_GROUP_TRANSITIONS.csv',transitions),
                          ('W03_REPORT_AGREEMENT.csv',report_checks)):
        pd.DataFrame(data).to_csv(output_dir/filename,index=False,float_format='%.17g',lineterminator='\n')
    dump(output_dir/'W03_TRANSITION_MATRICES.json',dict(schema='paper2.w03.transitions.v1',
         signs=list(SIGNS), orientation='source row to destination column', comparisons=matrices))
    sources.unchanged()
    dump(output_dir/'W03_SOURCE_MANIFEST.json',dict(schema='paper2.w03.sources.v1', sources=sources.records,
         verification_scope='consumed accepted W01/W02 outputs, acceptance/reproducibility/delivery receipts, current mapping and selected normative contracts; upstream full trajectory audit inherited from W01/W02'))
    audit = dict(schema='paper2.w03.integrity.v1',status='PASS',scientific_hard_gates=[],
                 N_groups=21,claim_states=21,comparisons=17,group_transition_rows=357,transition_matrices=17,
                 heterogeneity_rows=21,accepted_W01_report_rows_checked=8,accepted_W02_report_rows_checked=9,
                 report_numeric_fields_checked=len(report_checks),report_sign_counts_reproduced=True,
                 accepted_CSV_JSON_exact_agreement=True,accepted_W01_retention_reproduced=True,
                 accepted_W01_point_estimates_and_intervals_reproduced=True,
                 accepted_W02_Theta_and_intervals_reproduced=True,accepted_sign_counts_reproduced=True,
                 branch_config_identities_verified=42,Full_minus_comparator_orientation_verified=True,
                 Theta_orientation_verified=True,same_group_IDs_and_mapping_verified=True,
                 exactly_21_unique_groups_every_comparison=True,transition_marginals_verified=True,
                 retained_plus_flipped_plus_any_zero_equals_21=True,consumed_sources_unchanged=True,
                 consumed_files_checked=len(sources.records),sign_tolerance=EPS,
                 interval_method='copied accepted pointwise 95% BCa; no bootstrap executed',
                 primary_M1_verification_limit='accepted event areas/group cells and canonical-check receipts; complete primary perturbed states unavailable; no replay',
                 new_inferential_family=False,direct_perturbation_construction_difference_computed=False,
                 new_intervals=0,new_p_values=0,scientific_replays=0,training=False,
                 search_regime_analysis=False,later_workflows_executed=[],polling_or_waiting_created=False,
                 deterministic_rerun_receipt='W03_reproducibility.json',test_receipt='W03_test_results.json',
                 final_acceptance_receipt='W03_acceptance.json',software=dict(python='3.11',numpy=np.__version__,pandas=pd.__version__))
    dump(output_dir/'W03_integrity_audit.json',audit)
    print('W03 recovery PASS: 21 states, 17 comparisons, 357 paired rows; accepted hashes, means, intervals, signs and report tables verified.')
    return audit


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', type=Path, default=HERE)
    args = parser.parse_args()
    run(args.output_dir)
