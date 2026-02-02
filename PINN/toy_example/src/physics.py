import torch


def burgers_residual(model, xyt, nu):
    """This function computes and returns the residual of the 2D Burgers' equation."""

    xyt.requires_grad_(True)
    u = model(xyt)

    grads = torch.autograd.grad(
        u, xyt, torch.ones_like(u), create_graph=True
    )[0]

    u_x = grads[:, 0:1]
    u_y = grads[:, 1:2]
    u_t = grads[:, 2:3]

    u_xx = torch.autograd.grad(u_x, xyt, torch.ones_like(u_x), create_graph=True)[0][:, 0:1]
    u_yy = torch.autograd.grad(u_y, xyt, torch.ones_like(u_y), create_graph=True)[0][:, 1:2]

    return u_t + u * u_x + u * u_y - nu * (u_xx + u_yy)