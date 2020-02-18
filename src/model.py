import torch
import numpy as np
import torch.nn as nn
import networkx as nx
from utils import power_method


class Threat_Model(nn.Module):
    def __init__(self, S, S_prime, Alpha, budget_change_ratio, learning_rate, G):
        super(Threat_Model, self).__init__()
        self.numNodes = len(G)
        self.avgDeg = np.mean([G.degree(i) for i in range(self.numNodes)])
        self.maxDeg = np.max(list(dict(G.degree).values()))

        self.S = S
        self.S_prime = S_prime
        
        self.alpha_1, self.alpha_2, self.alpha_3 = Alpha
        self.learning_rate = learning_rate

        # tracks the amount of budget used
        self.used_budget = torch.zeros(1)
        
        # those eigenvalues in the objective function
        self.lambda1_S_prime = 0
        self.lambda1_S = 0
        self.centrality = 0
        self.lambda1 = 0
        self.Loss = 0
        
        # the pristine adjacency matrix
        adj = nx.adjacency_matrix(G).todense()
        self.original_adj = torch.tensor(adj, dtype=torch.float32)
        
        # eigenvals and eigenvectors associated with the largest eig-value of adj
        v_original = power_method(self.original_adj)
        self.lambda1_original = (v_original.view(1, -1) @ self.original_adj @ v_original.view(-1, 1)).squeeze()

        # degree matrix
        D = torch.diag(self.original_adj @ torch.ones(self.numNodes).view(-1, 1).squeeze())
        # Laplacian matrix
        L = D - self.original_adj

        # characteristic vector for sets S and S_prime
        x_s = torch.zeros(self.numNodes)
        x_s[self.S] = 1
        x_s_prime = torch.zeros(self.numNodes)
        x_s_prime[self.S_prime] = 1

        # select the sub adjacency matrix corresponding to S and S_prime
        adj_S = torch.index_select(torch.index_select(self.original_adj, 0, self.S), 1, self.S)
        v_est_S = power_method(adj_S)
        self.lambda1_S_original = v_est_S.view(1, -1) @ adj_S @ v_est_S.view(-1, 1)
    
        # centrality measure
        vol_S = x_s.view(1, -1) @ D @ x_s.view(-1, 1)
        vol_S_prime = x_s_prime.view(1, -1) @ D @ x_s_prime.view(-1, 1)
        normalization_const = 1 / vol_S + 1 / vol_S_prime 
        cut_size = x_s.view(1, -1) @ L @ x_s.view(-1, 1)
        self.centrality_original = cut_size * normalization_const

        # |lambda1(\tilde{A})-lambda1(A)| <= lambda1(A) * budget_change_ratio
        self.budget = self.lambda1_original * budget_change_ratio
       
        # the thing we wanna optimize
        # requires_grad_(True): tells PyTorch to starting tracking the gradients of this parameter
        self.adj_tensor = torch.tensor(adj, dtype=torch.float32).requires_grad_(True)
        # tell PyTorch that self.adj_tensor is a parameter of Threat_Model, which we will optimize over
        self.adj_tensor = nn.Parameter(self.adj_tensor)
        
        # masking the gradients backpropagated to adj_tensor
        # make sure the perturbation added to the adjacency matrix is symmetric
        self.adj_tensor.register_hook(lambda x: (1/2) * (x + torch.transpose(x, 0, 1)) * self.original_adj)


    def forward(self):
        """
            Compute loss given current (perturbed) adjacency matrix
        """
        # degree matrix
        D = torch.diag(self.adj_tensor @ torch.ones(self.numNodes).view(-1, 1).squeeze())
        # Laplacian matrix
        L = D - self.adj_tensor

        # characteristic vector for sets S and S_prime
        x_s = torch.zeros(self.numNodes)
        x_s[self.S] = 1
        x_s_prime = torch.zeros(self.numNodes)
        x_s_prime[self.S_prime] = 1

        # select the sub adjacency matrix corresponding to S and S_prime
        adj_tensor_S = torch.index_select(torch.index_select(self.adj_tensor, 0, self.S), 1, self.S)
        adj_tensor_S_prime = torch.index_select(torch.index_select(self.adj_tensor, 0, self.S_prime), 1, self.S_prime)
    
        # all sorts of largest eigenvalues 
        v_est = power_method(self.adj_tensor.data)
        v_est_S = power_method(adj_tensor_S.data)
        v_est_S_prime = power_method(adj_tensor_S_prime.data)
        self.lambda1 = v_est.view(1, -1) @ self.adj_tensor @ v_est.view(-1, 1)
        self.lambda1_S = v_est_S.view(1, -1) @ adj_tensor_S @ v_est_S.view(-1, 1)
        self.lambda1_S_prime = v_est_S_prime.view(1, -1) @ adj_tensor_S_prime @ v_est_S_prime.view(-1, 1)
    
        # centrality measure
        vol_S = x_s.view(1, -1) @ D @ x_s.view(-1, 1)
        vol_S_prime = x_s_prime.view(1, -1) @ D @ x_s_prime.view(-1, 1)
        normalization_const = 1 / vol_S + 1 / vol_S_prime 
        cut_size = x_s.view(1, -1) @ L @ x_s.view(-1, 1)
        self.centrality = cut_size * normalization_const
        
        # loss function 
        U1 = self.alpha_1 * (self.lambda1_S - self.lambda1_S_original) / self.lambda1_S_original / self.avgDeg
        U3 = self.alpha_3 * (self.centrality - self.centrality_original) / self.centrality_original
        #print("U1: {:.4f}    U2: {:.4f}    U3: {:.4f}".format(U1.detach().squeeze().numpy(), 
        #                                                     U2.detach().squeeze().numpy(),
        #                                                     U3.detach().squeeze().numpy()))
        self.Loss = -1 * (U1 + U3)
        return self.Loss


    def subgraph_centrality(self):
        eigVals, eigVecs = torch.symeig(self.adj_tensor, eigenvectors=True)
        eigVals_exp = torch.diag(torch.exp(eigVals))
        subgraph_cent = torch.diag(torch.mm(eigVecs, torch.mm(eigVals_exp, torch.transpose(eigVecs, 0, 1))))
        C = subgraph_cent[self.S].sum()
        return C

    def get_Laplacian(self):
        # degree matrix
        D = torch.diag(self.adj_tensor @ torch.ones(self.numNodes).view(-1, 1).squeeze())
        # Laplacian matrix
        L = D - self.adj_tensor
        return L.detach().clone()


    def get_budget(self):
        return self.budget
    

    # budget consumed in each step
    def get_step_budget(self):
        if self.adj_tensor.grad != None:
            # perturbation = gradient x learning rate
            pert = self.adj_tensor.grad * self.learning_rate
            # budget used in this step is the operator norm of pert
            # step_budget = torch.max(torch.abs(torch.symeig(pert)[0]))
            v = power_method(pert)
            step_budget = v.view(1, -1) @ pert @ v.view(-1, 1)
            return step_budget


    # update how much budget used
    def update_used_budget(self, used_b):
        self.used_budget += used_b.squeeze()


    # return the amount of budget consumed
    def get_used_budget(self):
        return self.used_budget


    def getRet(self):
        return self.lambda1_S, self.lambda1, self.centrality

    
    def get_attacked_adj(self):
        return self.adj_tensor.detach().clone()


    def get_utility(self):
        return -1 * self.Loss


    # check budget constraint (for debug purpose)
    def check_constraint(self):
        #lambda1 = torch.max(torch.symeig(self.adj_tensor, eigenvectors=True)[0])
        v = power_method(self.adj_tensor)
        lambda1 = v.view(1, -1) @ self.adj_tensor @ v.view(-1, 1)
        return torch.abs(self.lambda1 - self.lambda1_original) <= self.budget


