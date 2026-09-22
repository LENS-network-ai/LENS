# Adapted from score written by wkentaro
# https://github.com/wkentaro/pytorch-fcn/blob/master/torchfcn/utils.py

import numpy as np


def concordance_index(risk_scores, event_times, censorships):
    """
    Harrell's concordance index (c-index) for survival prediction.

    A pair (i, j) is "comparable" when the one with the shorter observed
    time actually had the event (i.e. its time isn't censored). Among
    comparable pairs, the fraction where the higher risk score corresponds
    to the shorter survival time is the c-index.

    Args:
        risk_scores  : [N] higher = higher predicted risk (e.g. sum of hazards,
                       or -predicted survival time)
        event_times  : [N] observed time (survival_months, or the discretized bin)
        censorships  : [N] 1 if censored, 0 if the event was observed

    Returns:
        float in [0, 1], or 0.5 if there are no comparable pairs.
    """
    risk_scores = np.asarray(risk_scores).reshape(-1)
    event_times = np.asarray(event_times).reshape(-1)
    censorships = np.asarray(censorships).reshape(-1)

    n = len(risk_scores)
    if n < 2:
        return 0.5

    # pair (i, j) is comparable iff i had an observed event and i's time < j's time
    had_event = (censorships == 0)[:, None]                       # [N, 1]
    shorter_time = event_times[:, None] < event_times[None, :]    # [N, N]
    comparable = had_event & shorter_time

    higher_risk = risk_scores[:, None] > risk_scores[None, :]
    tied_risk = risk_scores[:, None] == risk_scores[None, :]

    num_comparable = comparable.sum()
    if num_comparable == 0:
        return 0.5

    num_concordant = (comparable & higher_risk).sum() + 0.5 * (comparable & tied_risk).sum()
    return float(num_concordant / num_comparable)


class ConfusionMatrix(object):

    def __init__(self, n_classes):
        self.n_classes = n_classes
        # axis = 0: prediction
        # axis = 1: target
        self.confusion_matrix = np.zeros((n_classes, n_classes))

    def _fast_hist(self, label_true, label_pred, n_class):
        hist = np.zeros((n_class, n_class))
        hist[label_pred, label_true] += 1

        return hist

    def update(self, label_trues, label_preds):
        for lt, lp in zip(label_trues, label_preds):
            tmp = self._fast_hist(lt.item(), lp.item(), self.n_classes)    #lt.item(), lp.item()
            self.confusion_matrix += tmp

    def get_scores(self):
        """Returns accuracy score evaluation result.
            - overall accuracy
            - mean accuracy
            - mean IU
            - fwavacc
        """
        hist = self.confusion_matrix
        # accuracy is recall/sensitivity for each class, predicted TP / all real positives
        # axis in sum: perform summation along

        if sum(hist.sum(axis=1)) != 0:
            acc = sum(np.diag(hist)) / sum(hist.sum(axis=1))
        else:
            acc = 0.0
        
        return acc
    
    def plotcm(self):
        print(self.confusion_matrix)

    def reset(self):
        self.confusion_matrix = np.zeros((self.n_classes, self.n_classes))

