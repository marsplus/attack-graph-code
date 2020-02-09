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
def power_method(mat):
	n = len(mat)
	x = torch.rand(n) 
	x /= x.sum()

	flag = 1e-6
	# while True:
	for i in range(100):
		x_new = mat @ x
		x_new /= torch.norm(x_new, 2)
		Diff = torch.norm(x_new - x, 2)
		if Diff <= flag:
			break
		x = x_new
	return x_new

