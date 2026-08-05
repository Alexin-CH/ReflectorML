import torch


def anneal_blur_sigma(step, epochs, blur_sigma, blur_final=0.0):
    """Linearly anneal the Gaussian blur width from `blur_sigma` to `blur_final`.

    Early training sees heavily smoothed densities (smooth f/g -> well-posed MA),
    which is progressively sharpened toward the true piecewise-constant densities
    as training converges.
    """
    if epochs <= 1:
        return blur_final
    frac = min(max(step / (epochs - 1), 0.0), 1.0)
    return blur_final + (blur_sigma - blur_final) * (1.0 - frac)


def loss_grad_norm(loss_t, model):
    """Mean |grad| of a loss term wrt model params."""
    grads = torch.autograd.grad(
        loss_t, model.parameters(),
        retain_graph=True, allow_unused=True
    )
    total = 0.0
    count = 0.0
    for g in grads:
        if g is None:
            continue
        total = total + g.abs().sum()
        count = count + g.numel()
    return total / count


def anneal_weights(model, ma_loss, cv_loss, bc_loss, transport_loss,
                   loss_weights, alpha):
    """Wang et al. 2001.04536, Algorithm 1.

    Using the MA (physics) term as the reference, rescale the CV/BC/data
    weighting lambdas so that the mean back-propagated gradient magnitude of each
    term is matched to that of the MA term. loss_weights is updated IN PLACE
    (mutated) so that closures capture the final values via the same list object.
    """
    ref = loss_grad_norm(ma_loss, model) + 1e-12
    for term, idx in ((cv_loss, 2), (bc_loss, 1), (transport_loss, 3)):
        if loss_weights[idx] == 0:
            continue
        lam = ref / (loss_grad_norm(term, model) + 1e-12)
        loss_weights[idx] = (1 - alpha) * loss_weights[idx] + alpha * float(lam)