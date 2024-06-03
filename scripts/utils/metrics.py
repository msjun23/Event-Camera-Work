import torch

DISPARITY_MULTIPLIER = 7.0
FOCAL_LENGTH_X_BASELINE = {
    'indoor_flying': 19.941772,
}

def disparity_to_depth(disparity_map):
    depth_map = FOCAL_LENGTH_X_BASELINE['indoor_flying'] / (disparity_map + 1e-7)
    return depth_map

def mean_depth_error(pred, gt):
    # pred, gt, mask: (H, W)
    assert pred.shape == gt.shape    
    mask = pred > 0.
    pred, gt = pred[mask], gt[mask]
    assert pred.shape == gt.shape
    
    pred = disparity_to_depth(pred)
    gt = disparity_to_depth(gt)
    error = torch.abs(pred - gt)
    return error.mean() * 100       # report in [cm]

def mean_disparity_error(pred, gt):
    # This is same to mean_average_error
    # pred, gt, mask: (H, W)
    assert pred.shape == gt.shape
    
    error = torch.abs(pred - gt)
    return error.mean()             # report in [pix]

def n_pixel_accuracy(pred, gt, n=1):
    # pred, gt, mask: (H, W)
    assert pred.shape == gt.shape
    
    error = torch.abs(pred - gt)
    error_mask = (error <= n).to(torch.float)
    return error_mask.mean() * 100  # report in [%]

def mean_average_error(pred, gt):
    # This is same to mean_disparity_error
    # pred, gt, mask: (H, W)
    assert pred.shape == gt.shape
    
    error = torch.abs(pred - gt)
    return error.mean()             # report in [pix]

def n_pixel_error(pred, gt, n=1):
    # pred, gt, mask: (H, W)
    assert pred.shape == gt.shape
    
    error = torch.abs(pred - gt)
    error_mask = (error > n).to(torch.float)
    return error_mask.mean() * 100
    
def root_mean_square_error(pred, gt):
    # pred, gt, mask: (H, W)
    assert pred.shape == gt.shape
    error = ((pred - gt)**2).mean().sqrt()
    return error