"""
Tests for utils/splits.py: patient ids and patient-grouped folds.
    pytest tests/test_splits.py
"""

import numpy as np

from utils.splits import make_folds, patient_id


def test_patient_id_cptac_and_tcga():
    assert patient_id('C3L-02660-25') == 'C3L-02660'
    assert patient_id('C3L-02660-21') == 'C3L-02660'
    assert patient_id('TCGA-3C-AALI-01Z-00-DX1.F6E9') == 'TCGA-3C-AALI'


def _toy_ids(n_patients=60, slides_per_patient=4):
    ids, labels = [], []
    for p in range(n_patients):
        for s in range(slides_per_patient):
            ids.append(f"C3L-{p:05d}-{s:02d}\tx")
            labels.append(p % 3)
    return ids, labels


def test_grouped_folds_never_split_a_patient():
    ids, labels = _toy_ids()
    folds = make_folds(ids, labels, n_folds=5, seed=42, group_by_patient=True)
    assert len(folds) == 5
    for train_idx, val_idx in folds:
        train_patients = {patient_id(ids[i].split()[0]) for i in train_idx}
        val_patients = {patient_id(ids[i].split()[0]) for i in val_idx}
        assert not train_patients & val_patients
        assert set(train_idx).isdisjoint(val_idx)


def test_every_slide_is_validated_exactly_once():
    ids, labels = _toy_ids()
    folds = make_folds(ids, labels, n_folds=5, seed=7)
    val_all = np.concatenate([v for _, v in folds])
    assert sorted(val_all) == list(range(len(ids)))


def test_legacy_split_can_leak_patients():
    # The slide-level split is what the grouped one fixes: with 4 slides per
    # patient it puts a patient on both sides of a fold.
    ids, labels = _toy_ids()
    folds = make_folds(ids, labels, n_folds=5, seed=42, group_by_patient=False)
    leaked = False
    for train_idx, val_idx in folds:
        train_patients = {patient_id(ids[i].split()[0]) for i in train_idx}
        val_patients = {patient_id(ids[i].split()[0]) for i in val_idx}
        leaked = leaked or bool(train_patients & val_patients)
    assert leaked
