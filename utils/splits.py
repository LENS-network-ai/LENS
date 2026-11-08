"""Cross-validation splits that keep every patient's slides in one fold."""

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold


def patient_id(slide_id):
    """Patient identifier from a slide id.

    TCGA barcodes (TCGA-3C-AALI-01Z-...): first three segments.
    CPTAC ids (C3L-02660-25): first two segments, i.e. the case id; the last
    segment numbers the slide within the case.
    """
    parts = slide_id.split('-')
    return '-'.join(parts[:3] if slide_id.startswith('TCGA') else parts[:2])


def make_folds(ids, labels, n_folds, seed, group_by_patient=True):
    """Return a list of (train_idx, val_idx) index arrays.

    ids: slide ids or list-file lines ("slide_id<TAB>label"); only the first
        whitespace-separated token is used.
    group_by_patient: use StratifiedGroupKFold on the patient id so no patient
        appears in both train and validation. False reproduces the legacy slide-level
        StratifiedKFold split.
    """
    labels = np.asarray(labels)
    if not group_by_patient:
        return list(StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
                    .split(np.zeros(len(labels)), labels))
    groups = [patient_id(line.split()[0]) for line in ids]
    return list(StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
                .split(np.zeros(len(labels)), labels, groups))
