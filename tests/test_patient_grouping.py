"""
Regression test for get_patient_id (training/train_survival.py), the
TCGA-barcode parsing used to group slides by patient for
StratifiedGroupKFold -- guards against the patient-level CV leakage bug this
fixed (some patients in the TCGA-BRCA cohort contribute >1 slide, e.g. both
a DX1 and DX2 diagnostic slide; a plain slide-level split can and did put
one of a patient's slides in train and the other in val for the same fold).

Requires the project's own environment -- run with:
    pytest tests/test_patient_grouping.py
"""

from training.train_survival import get_patient_id


def test_single_slide_patient():
    assert get_patient_id('TCGA-3C-AALI-01Z-00-DX1.F6E9A5DF-1234') == 'TCGA-3C-AALI'


def test_two_slides_same_patient_map_to_same_id():
    # The exact scenario the patient-grouping fix targets: DX1 and DX2 are
    # different slides but the same patient, and must land in the same
    # group/fold.
    dx1 = get_patient_id('TCGA-3C-AALI-01Z-00-DX1.F6E9A5DF-AAAA-BBBB')
    dx2 = get_patient_id('TCGA-3C-AALI-01Z-00-DX2.A1B2C3D4-CCCC-DDDD')
    assert dx1 == dx2 == 'TCGA-3C-AALI'


def test_different_patients_map_to_different_ids():
    a = get_patient_id('TCGA-3C-AALI-01Z-00-DX1.F6E9A5DF-1234')
    b = get_patient_id('TCGA-A2-A0YM-01Z-00-DX1.1A2B3C4D-5678')
    assert a != b
