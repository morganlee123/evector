# CREDIT https://speechbrain.readthedocs.io/en/latest/_modules/speechbrain/utils/metric_stats.html

#
# Since there are a LOT of impostor scores, I modified this code from speechbrain to randomly
# sample / throw away ~95% of cand. threshold values (which it assumes every score is a threshold X 2 because of interm values)
# so the matrix multiplication doesn't take up 1000GB of ram...
# My assumption for this is that my dataset is so large, it's normally distributed, etc.
#
# - Morgan
#

import torch

def minDCF(
    positive_scores, negative_scores, c_miss=1.0, c_fa=1.0, p_target=0.01
):
    """Computes the minDCF metric normally used to evaluate speaker verification
    systems. The min_DCF is the minimum of the following C_det function computed
    within the defined threshold range:

    C_det =  c_miss * p_miss * p_target + c_fa * p_fa * (1 -p_target)

    where p_miss is the missing probability and p_fa is the probability of having
    a false alarm.

    Arguments
    ---------
    positive_scores : torch.tensor
        The scores from entries of the same class.
    negative_scores : torch.tensor
        The scores from entries of different classes.
    c_miss : float
         Cost assigned to a missing error (default 1.0).
    c_fa : float
        Cost assigned to a false alarm (default 1.0).
    p_target: float
        Prior probability of having a target (default 0.01).


    Example
    -------
    >>> positive_scores = torch.tensor([0.6, 0.7, 0.8, 0.5])
    >>> negative_scores = torch.tensor([0.4, 0.3, 0.2, 0.1])
    >>> val_minDCF, threshold = minDCF(positive_scores, negative_scores)
    >>> val_minDCF
    0.0
    """

    # Computing candidate thresholds
    #print('computed cand thresholds')
    thresholds, _ = torch.sort(torch.cat([positive_scores, negative_scores]))
    #print(len(thresholds))
    
    uniq_thresholds = torch.unique(thresholds)
    import random
    filter = [random.random() > 0.95 for _ in range(len(uniq_thresholds))]
    thresholds = uniq_thresholds[filter]
    #print(len(thresholds))

    #print('computed im thresholds')
    # Adding intermediate thresholds
    interm_thresholds = (thresholds[0:-1] + thresholds[1:]) / 2
    thresholds, _ = torch.sort(torch.cat([thresholds, interm_thresholds]))

    #print('computed FRR')
    # Computing False Rejection Rate (miss detection)
    positive_scores = torch.cat(
        len(thresholds) * [positive_scores.unsqueeze(0)]
    )
    pos_scores_threshold = positive_scores.transpose(0, 1) <= thresholds
    p_miss = (pos_scores_threshold.sum(0)).float() / positive_scores.shape[1]
    del positive_scores
    del pos_scores_threshold

    #print('computed FAR')
    # Computing False Acceptance Rate (false alarm) # TODO: Modify this to take up time rather than space. Currently doing some matrix stuff and it's taking 200 GB of RAM
    negative_scores = torch.cat(
        len(thresholds) * [negative_scores.unsqueeze(0)]
    )
    neg_scores_threshold = negative_scores.transpose(0, 1) > thresholds
    p_fa = (neg_scores_threshold.sum(0)).float() / negative_scores.shape[1]
    del negative_scores
    del neg_scores_threshold

    #print('costing')
    c_det = c_miss * p_miss * p_target + c_fa * p_fa * (1 - p_target)
    c_min, min_index = torch.min(c_det, dim=0)

    return float(c_min), float(thresholds[min_index])
