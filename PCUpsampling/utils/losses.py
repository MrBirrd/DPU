import torch
import torch.nn.functional as F


def get_scaling(snr):
    snr_scaled = torch.log(1 + snr)
    return snr_scaled


def projection_loss(cloud_gt, cloud_pred, grid_size=0.01, range_x=[-1, 1], range_y=[-1, 1], reduction="none"):
    """
    Computes the projection loss between the ground truth and the predicted pointcloud.
    Pointclouds should be of shape (batch_size, num_points, 3) and normalized to [-1, 1]
    """
    # generate hists
    gt_hist = soft_histogram(cloud_gt, grid_size=grid_size, range_x=range_x, range_y=range_y)
    pred_hist = soft_histogram(cloud_pred, grid_size=grid_size, range_x=range_x, range_y=range_y)

    loss = F.mse_loss(pred_hist, gt_hist, reduction=reduction)
    return loss


def project_to_plane(points):
    return points[:, :2]  # Assuming points is (Batch_size x N x 3)


def soft_histogram(points, grid_size, range_x, range_y):
    # Create the grid
    grid_x = torch.linspace(range_x[0], range_x[1], steps=int((range_x[1] - range_x[0]) / grid_size)).to(points.device)
    grid_y = torch.linspace(range_y[0], range_y[1], steps=int((range_y[1] - range_y[0]) / grid_size)).to(points.device)

    # Soft assignment to the x and y axis
    points_x = points[:, :, 0:1]  # (Batch_size x N x 1)
    points_y = points[:, :, 1:2]  # (Batch_size x N x 1)

    weights_x = torch.exp(-((points_x - grid_x) ** 2) / (2 * 0.01**2))  # Gaussian kernel
    weights_y = torch.exp(-((points_y - grid_y) ** 2) / (2 * 0.01**2))

    # 2D histogram (Batch_size x len(grid_x) x len(grid_y))
    histogram = torch.einsum("bnm,bnk->bmk", weights_x, weights_y)

    return histogram
