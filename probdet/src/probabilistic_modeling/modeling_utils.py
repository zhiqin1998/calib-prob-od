import torch
from torch.nn import functional as F


def cross_entropy_soft_targets(input, target, reduction='none'):
    log_input = torch.nn.functional.log_softmax(input, dim=-1)
    loss = -torch.sum(target * log_input, dim=-1)

    if reduction == 'none':
        return loss
    elif reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    else:
        raise NotImplementedError('Unsupported reduction mode.')

def sigmoid_focal_loss_soft(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = -1,
    gamma: float = 2,
    reduction: str = "none",
) -> torch.Tensor:
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
        reduction: 'none' | 'mean' | 'sum'
                 'none': No reduction will be applied to the output.
                 'mean': The output will be averaged.
                 'sum': The output will be summed.
    Returns:
        Loss tensor with the reduction option applied.
    """
    inputs = inputs.float()
    targets = targets.float()
    p = torch.sigmoid(inputs)
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    # p_t = p * targets + (1 - p) * (1 - targets)
    loss = ce_loss * (torch.abs(targets - p) ** gamma)

    if alpha >= 0:
        a1, a0 = alpha, 1 - alpha
        # alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        alpha_t = a0 + targets * (a1 - a0)
        loss = alpha_t * loss

    if reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()

    return loss


sigmoid_focal_loss_soft_jit: "torch.jit.ScriptModule" = torch.jit.script(sigmoid_focal_loss_soft)
# sigmoid_focal_loss_soft_jit = sigmoid_focal_loss_soft
def covariance_output_to_cholesky(pred_bbox_cov):
    """
    Transforms output to covariance cholesky decomposition.
    Args:
        pred_bbox_cov (kx4 or kx10): Output covariance matrix elements.

    Returns:
        predicted_cov_cholesky (kx4x4): cholesky factor matrix
    """
    # Embed diagonal variance
    diag_vars = torch.sqrt(torch.exp(pred_bbox_cov[:, 0:4]))
    predicted_cov_cholesky = torch.diag_embed(diag_vars)

    if pred_bbox_cov.shape[1] > 4:
        tril_indices = torch.tril_indices(row=4, col=4, offset=-1)
        predicted_cov_cholesky[:, tril_indices[0],
                               tril_indices[1]] = pred_bbox_cov[:, 4:]

    return predicted_cov_cholesky


def clamp_log_variance(pred_bbox_cov, clamp_min=-7.0, clamp_max=7.0):
    """
    Tiny function that clamps variance for consistency across all methods.
    """
    pred_bbox_var_component = torch.clamp(
        pred_bbox_cov[:, 0:4], clamp_min, clamp_max)
    return torch.cat((pred_bbox_var_component, pred_bbox_cov[:, 4:]), dim=1)


def get_probabilistic_loss_weight(current_step, annealing_step):
    """
    Tiny function to get adaptive probabilistic loss weight for consistency across all methods.
    """
    probabilistic_loss_weight = min(1.0, current_step / annealing_step)
    probabilistic_loss_weight = (
        100 ** probabilistic_loss_weight - 1.0) / (100.0 - 1.0)

    return probabilistic_loss_weight
