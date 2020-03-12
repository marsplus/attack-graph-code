import torch
import numpy as np
import torch.nn as nn
import networkx as nx
from utils import *


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
        adj_S         = get_submatrix(self.original_adj, self.S, self.S)
        adj_S_prime   = get_submatrix(self.original_adj, self.S_prime, self.S_prime)
        v_est_S       = power_method(adj_S)
        v_est_S_prime = power_method(adj_S_prime)
        self.lambda1_S_original = v_est_S.view(1, -1) @ adj_S @ v_est_S.view(-1, 1)
        self.lambda1_S_prime_original = v_est_S_prime.view(1, -1) @ adj_S_prime @ v_est_S_prime.view(-1, 1)
    
        ## centrality measure
        vol_S = x_s.view(1, -1) @ D @ x_s.view(-1, 1)
        vol_S_prime = x_s_prime.view(1, -1) @ D @ x_s_prime.view(-1, 1)
        normalization_const = 1 / vol_S + 1 / vol_S_prime 
        cut_size = x_s.view(1, -1) @ L @ x_s.view(-1, 1)
        self.centrality_original = cut_size * normalization_const

        ## negative impact
        self.impact_S_original = v_original[self.S].sum()
        # self.impact_S_original = self.lambda1_S_prime_original


        # |lambda1(\tilde{A})-lambda1(A)| <= lambda1(A) * budget_change_ratio
        self.budget = self.lambda1_original * budget_change_ratio
       
        ## requires_grad_(True): tells PyTorch to starting tracking the gradients of this parameter
        self.adj_tensor = torch.tensor(adj, dtype=torch.float32).requires_grad_(True)
        self.adj_tensor = nn.Parameter(self.adj_tensor)
        
        # masking the gradients backpropagated to adj_tensor
        def _mask_(x):
            x_copy = x.clone()
            x_copy = (1/2) *( x_copy + torch.transpose(x_copy, 0, 1))
            #x_copy[x_copy > 0] = 1
            return x_copy
        self.adj_tensor.register_hook(lambda x: _mask_(x))
        # self.adj_tensor.register_hook(lambda x: (1/2) * (x + torch.transpose(x, 0, 1)))

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
        adj_tensor_S = get_submatrix(self.adj_tensor, self.S, self.S)
        adj_tensor_S_prime = get_submatrix(self.adj_tensor, self.S_prime, self.S_prime)
    
        # all sorts of largest eigenvalues 
        v_est          = power_method(self.adj_tensor.data)
        v_est_S        = power_method(adj_tensor_S.data)
        v_est_S_prime  = power_method(adj_tensor_S_prime.data)
        self.lambda1_S = v_est_S.view(1, -1) @ adj_tensor_S @ v_est_S.view(-1, 1)
        self.lambda1_S_prime = v_est_S_prime.view(1, -1) @ adj_tensor_S_prime @ v_est_S_prime.view(-1, 1)

    
        ## centrality measure
        vol_S = x_s.view(1, -1) @ D @ x_s.view(-1, 1)
        vol_S_prime = x_s_prime.view(1, -1) @ D @ x_s_prime.view(-1, 1)
        normalization_const = 1 / vol_S + 1 / vol_S_prime 
        cut_size = x_s.view(1, -1) @ L @ x_s.view(-1, 1)
        self.centrality = cut_size * normalization_const


        ## negative impact
        self.impact_S = v_est[self.S].sum()
        # self.impact_S = self.lambda1_S_prime

        
        # utility function 
        U1 =  self.alpha_1 * (self.lambda1_S - self.lambda1_S_original) / self.lambda1_S_original 
        U2 =  self.alpha_2 * (self.impact_S - self.impact_S_original) / self.impact_S_original
        U3 =  self.alpha_3 * (self.centrality - self.centrality_original) / self.centrality_original
        #print("U1: {:.4f}    U2: {:.4f}".format(U1.detach().squeeze().numpy(), 
        #                                                     U2.detach().squeeze().numpy()))
                                                        
        self.Loss = -1 * (U1 + U2 + U3)
        return self.Loss


    def subgraph_centrality(self):
        eigVals, eigVecs = torch.symeig(self.adj_tensor, eigenvectors=True)
        eigVals_exp = torch.diag(torch.exp(eigVals))
        subgraph_cent = torch.diag(torch.mm(eigVecs, torch.mm(eigVals_exp, torch.transpose(eigVecs, 0, 1))))
        C = subgraph_cent[self.S].sum()
        return C


    def get_Laplacian(self):
        D = torch.diag(self.adj_tensor @ torch.ones(self.numNodes).view(-1, 1).squeeze())
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
            v1 = power_method(pert, 100)
            u1 = power_method(-pert, 100)
            step_budget = max(v1.view(1, -1) @ pert    @ v1.view(-1, 1), \
                              u1.view(1, -1) @ (-pert) @ u1.view(-1, 1) )
            return step_budget


    # update how much budget used
    def update_used_budget(self, used_b):
        self.used_budget += used_b.squeeze()


    # return the amount of budget consumed
    def get_used_budget(self):
        return self.used_budget


    def get_result(self):
        return self.lambda1_S, self.impact_S, self.centrality

    
    def get_attacked_adj(self):
        return self.adj_tensor.detach().clone()


    def get_utility(self):
        return -1 * self.Loss


    # check budget constraint 
    def check_constraint(self, extTensor=[]):
        if extTensor:
            pert = extTensor[0] - self.original_adj
        else:
            pert = self.adj_tensor - self.original_adj

        v1 = power_method(pert,  100)
        u1 = power_method(-pert, 100)
        spec_norm = max(v1.view(1, -1) @ pert    @ v1.view(-1, 1), \
                        u1.view(1, -1) @ (-pert) @ u1.view(-1, 1) )
        # print(spec_norm, self.budget)
        eigVal_constraint = (spec_norm <= self.budget)

        isSymmetric = torch.all(self.adj_tensor == torch.transpose(self.adj_tensor, 0, 1))

        isNonnegative = torch.all(self.adj_tensor >= 0)

        return (eigVal_constraint and isSymmetric and isNonnegative)
        


    # return the change (%) of the average degree
    # idx: focus on a subgraph indexed by idx
    def diff_avgDeg(self, idx=None):
        # the whole graph
        if idx == None:
            idx = torch.LongTensor(range(self.numNodes))
        mat_original = get_submatrix(self.original_adj, idx, idx)
        mat_attacked = get_submatrix(self.adj_tensor, idx, idx)

        avg_deg_original = mat_original.sum()   / len(idx)
        avg_deg_attacked = mat_attacked.sum()   / len(idx)

        ret = (avg_deg_attacked - avg_deg_original) / avg_deg_original
        return ret.detach().numpy()


    # measure the difference (%) of the frobinus norm of the adjacency matrix 
    # before and after the attack
    def diff_adjNorm(self):
        original_norm = torch.norm(self.original_adj)
        attacked_norm = torch.norm(self.adj_tensor)
        ret =  (attacked_norm - original_norm) / original_norm
        return ret.detach().numpy()





