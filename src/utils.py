import torch

# project a vector to a L_inf ball
# with radius r
def L_inf_proj(v, r):
    vec = v.clone()
    vec[vec < -r] = -r
    vec[vec > r] = r
    return vec

# an iterative method based on Rayleigh 
# quotient to estimate the largest eigenvalue
# and the associated eigenvector
def power_method(mat, Iter=50):
    n = len(mat)
    x = torch.rand(n) 
    x /= torch.norm(x, 2)

    flag = 1e-6
    # while True:
    for i in range(Iter):
        x_new = mat @ x
        x_new /= torch.norm(x_new, 2)
        Diff = torch.norm(x_new - x, 2)
        if Diff <= flag:
            break
        x = x_new
    return x_new


def matrix_proj_spectralNorm(mat, epsilon):
    U, Sigma, V = torch.svd(mat)
    Sigma_proj = L_inf_proj(Sigma, epsilon)
    mat_proj = U @ torch.diag(Sigma_proj) @ V.T
    mat_proj = mat_proj - torch.diag(torch.diag(mat_proj))
    mat_proj[mat_proj < 0] = 0
    mat_proj = (1/2) * (mat_proj + torch.transpose(mat_proj, 0, 1))
    return mat_proj



def matrix_vectorize_proj_frobinusNorm(mat, epsilon):
    dim = mat.size()
    mat_vec = mat.view(-1, 1)
    mat_proj = mat_vec * epsilon / max(epsilon, mat_vec.norm('fro'))
    mat_proj = mat_proj - torch.diag(torch.diag(mat_proj))
    mat_proj[mat_proj < 0] = 0
    mat_proj = (1/2) * (mat_proj + torch.transpose(mat_proj, 0, 1))
    return mat_proj.view(dim)


def get_submatrix(mat, row_idx, col_idx):
    return torch.index_select(torch.index_select(mat, 0, row_idx), 1, col_idx)

    