import numpy as np

def multi_dim_normalize(x,**kwargs):
    shape = x.shape
    return ((x.flatten() - x.min()) / x.max()).reshape(shape)

def dice_coef2(y_true, y_pred):
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    union = np.sum(y_true_f) + np.sum(y_pred_f)
    if union==0: return 1
    intersection = np.sum(y_true_f * y_pred_f)
    return 2. * intersection / union