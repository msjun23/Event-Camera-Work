import torch

DISPARITY_MULTIPLIER = 7.0
FOCAL_LENGTH_X_BASELINE = {
    'indoor_flying': 19.941772,
}

def disparity_to_depth(disparity_map):
    depth_map = FOCAL_LENGTH_X_BASELINE['indoor_flying'] / (disparity_map + 1e-7)
    return depth_map

def error_per_image(_pred, _gt, _mask):
    assert _pred.shape == _gt.shape == _mask.shape
    _pred, _gt = _pred[_mask], _gt[_mask]
    _error = torch.abs(_pred - _gt)
    return _error.mean()

def mean_depth_error(pred, gt, mask=None, on_epoch=True):
    # pred, gt, mask: (N, H, W), Channel of disparity map is one
    assert pred.shape == gt.shape == mask.shape
    
    error = 0.
    for _pred, _gt, _mask in zip(pred, gt, mask):
        # Disparity to depth
        _pred = disparity_to_depth(_pred)
        _gt = disparity_to_depth(_gt)
        error += error_per_image(_pred, _gt, _mask)
        
    if on_epoch:
        # Returns final average error for all batches
        error = error.mean()
    else:   # on step
        # Returns the sum of errors for the current batch
        # Final average error should be calculated outside this function
        error = error.sum()
    return error

def mean_disparity_error(pred, gt, mask=None, on_epoch=True):
    # This is same to mean_average_error
    # pred, gt, mask: (N, H, W), Channel of disparity map is one
    assert pred.shape == gt.shape == mask.shape
    
    error = 0.
    for _pred, _gt, _mask in zip(pred, gt, mask):
        error += error_per_image(_pred, _gt, _mask)
    
    if on_epoch:
        # Returns final average error for all batches
        error = error.mean()
    else:   # on step
        # Returns the sum of errors for the current batch
        # Final average error should be calculated outside this function
        error = error.sum()
    return error

def n_pixel_accuracy(pred, gt, mask=None, n=1, on_epoch=True):
    # This is same to mean_average_error
    # pred, gt, mask: (N, H, W), Channel of disparity map is one
    assert pred.shape == gt.shape == mask.shape
    
    n_pa = 0.
    for _pred, _gt, _mask in zip(pred, gt, mask):
        _pred, _gt = _pred[_mask], _gt[_mask]
        _error = torch.abs(_pred - _gt)
        _error_mask = _error <= n
        _error_mask = _error_mask.to(torch.float)
        n_pa += _error_mask.mean()       # n pixel accuracy for an image
        
    if on_epoch:
        # Returns final average error for all batches
        n_pa = n_pa.mean()
    else:   # on step
        # Returns the sum of errors for the current batch
        # Final average error should be calculated outside this function
        n_pa = n_pa.sum()
    return n_pa