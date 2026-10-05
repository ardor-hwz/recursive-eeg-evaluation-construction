"""Same bootstrap draws and weighting IDs, separate original IDs for splitting."""
from dataclasses import dataclass
import numpy as np
from sklearn.model_selection import GroupKFold

@dataclass(frozen=True)
class BootstrapSample:
    features: np.ndarray
    target: np.ndarray
    original_subject_ids: np.ndarray
    bootstrap_draw_ids: np.ndarray
    sampled_original_subjects: np.ndarray

def subject_bootstrap(features, target, subjects, seed):
    unique = np.unique(subjects)
    sampled = np.random.default_rng(seed).choice(unique, size=len(unique), replace=True)
    indices, groups = [], []
    for draw, subject in enumerate(sampled):
        current = np.flatnonzero(subjects == subject)
        indices.append(current)
        groups.append(np.full(len(current), -(draw + 1)))
    selected = np.concatenate(indices)
    return BootstrapSample(features[selected], target[selected], subjects[selected], np.concatenate(groups), sampled)

def stacking_splits(original_subject_ids, fold_count):
    original = np.asarray(original_subject_ids, dtype=int)
    for train, validation in GroupKFold(fold_count).split(np.zeros((len(original), 1)), groups=original):
        if set(original[train]) & set(original[validation]):
            raise RuntimeError('STOP: corrected stack original-subject overlap')
        yield train, validation
