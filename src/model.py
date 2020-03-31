import torch
import numpy as np
import torch.nn as nn
import networkx as nx
from utils import *


class Threat_Model(nn.Module):
    def __init__(self, S, S_prime, Alpha, budget_change_ratio, learning_rate, G, weighted_graph):
        super(Threat_Model, self).__init__()
        self.weighted_graph = weighted_graph 

        self.numNodes = len(G)
        self.avgDeg = np.mean([G.degree(i) for i in range(self.numNodes)])
        self.maxDeg = np.max(list(dict(G.degree).values()))

        self.S = S
        self.S_prime = S_prime
        
        self.alpha_1, self.alpha_2, self.alpha_3 = Alpha
        self.learning_rate = learning_rate

        ## tracks the amount of budget used
        self.used_budget = torch.zeros(1)
        
        self.lambda1_S_prime = 0
        self.lambda1_S = 0
        self.centrality = 0
        self.lambda1 = 0
        self.Loss = 0
        
        adj = nx.adjacency_matrix(G).todense()
        self.original_adj = torch.tensor(adj, dtype=torch.float32)
        
        ## eigenvals and eigenvectors associated with the largest eig-value of adj
        v_original = power_method(self.original_adj)
        self.lambda1_original = v_original @ self.original_adj @ v_original

        # degree and Laplacian matrices
        D = torch.diag(self.original_adj @ torch.ones(self.numNodes))
        L = D - self.original_adj

        # characteristic vector for sets S and S_prime
        x_s = torch.zeros(self.numNodes)
        x_s[self.S] = 1
        x_s_prime = torch.zeros(self.numNodes)
        x_s_prime[self.S_prime] = 1

        # select the sub-adjacency matrix corresponding to S and S_prime
        adj_S   = get_submatrix(self.original_adj, self.S, self.S)
        adj_SP  = get_submatrix(self.original_adj, self.S_prime, self.S_prime)

        self.avgDeg_S  = adj_S.sum() / len(self.S)
        self.avgDeg_SP = adj_SP.sum() / len(self.S_prime)
        
        v_est_S = power_method(adj_S)
        self.lambda1_S_original = v_est_S @ adj_S @ v_est_S
        
        v_est_SP = power_method(adj_SP)
        self.lambda1_SP_original = v_est_SP @ adj_SP @ v_est_SP
   
        ## centrality measure
        vol_S = x_s @ D @ x_s
        vol_S_prime = x_s_prime @ D @ x_s_prime
        normalization_const = 1 / vol_S + 1 / vol_S_prime 
        cut_size = x_s @ L @ x_s
        self.centrality_original = cut_size * normalization_const


        ## impact on S
        self.impact_S_original = v_original[self.S].sum()
        #self.impact_S_original = -self.lambda1_SP_original 

        # |lambda1(\tilde{A})-lambda1(A)| <= lambda1(A) * budget_change_ratio
        self.budget = self.lambda1_original * budget_change_ratio

        ## requires_grad_(True): tells PyTorch to track the gradients of this parameter
        self.adj_tensor = torch.tensor(adj, dtype=torch.float32).requires_grad_(True)
        self.adj_tensor = nn.Parameter(self.adj_tensor)
        
        ## ensure at each iteration the structure of the adjacency matrix 
        ## is perserved
        def _mask_(x):
            x_copy = x.clone()
            x_copy = (1/2) *( x_copy + torch.transpose(x_copy, 0, 1))         # symmetric
            x_copy -= torch.diag(torch.diag(x_copy))                          # diagonal entries are zero
            if self.weighted_graph:
                x_copy *= self.original_adj                                   # only modify existing edges 
            return x_copy
        self.adj_tensor.register_hook(lambda x: _mask_(x))



    def forward(self):
        """
            Compute loss given current (perturbed) adjacency matrix
        """
        #D = torch.diag(self.adj_tensor @ torch.ones(self.numNodes).view(-1, 1).squeeze())
        D = torch.diag(self.adj_tensor @ torch.ones(self.numNodes))
        L = D - self.adj_tensor

        # characteristic vector for sets S and S_prime
        x_s = torch.zeros(self.numNodes)
        x_s[self.S] = 1
        x_s_prime = torch.zeros(self.numNodes)
        x_s_prime[self.S_prime] = 1

        # select the sub adjacency matrix corresponding to S and S_prime
        adj_tensor_S   = get_submatrix(self.adj_tensor, self.S, self.S)
        adj_tensor_SP  = get_submatrix(self.adj_tensor, self.S_prime, self.S_prime)
    
        # all sorts of largest eigenvalues 
        v_est = power_method(self.adj_tensor.data)
        
        #eigVals_S, eigVecs_S     = torch.symeig(adj_tensor_S, eigenvectors=True)
        v_est_S        = power_method(adj_tensor_S.data)
        self.lambda1_S = v_est_S @ adj_tensor_S @ v_est_S 

        #eigVals_SP, eigVecs_SP   = torch.symeig(adj_tensor_S_prime, eigenvectors=True)
        v_est_SP        = power_method(adj_tensor_SP.data)
        self.lambda1_SP = v_est_SP @ adj_tensor_SP @ v_est_SP 

        ## centrality measure
        vol_S = x_s @ D @ x_s
        vol_S_prime = x_s_prime @ D @ x_s_prime
        normalization_const = 1 / vol_S + 1 / vol_S_prime 
        cut_size = x_s @ L @ x_s
        self.centrality = cut_size * normalization_const


        ## negative impact
        self.impact_S = v_est[self.S].sum()
        #self.impact_S = -self.lambda1_SP

        
        # utility function 
        U1 =  self.alpha_1 * self.lambda1_S / self.avgDeg_S
        U2 =  self.alpha_2 * self.impact_S 
        U3 =  self.alpha_3 * self.centrality
        #print("U1: {:.4f}    U2: {:.4f}    U3: {:.4f}".format(U1.detach().squeeze().numpy(), 
        #                                                      U2.detach().squeeze().numpy(),
        #                                                      U3.detach().squeeze().numpy()))
                                                        
        self.Loss = -1 * (U1 + U2 + U3)
        return self.Loss


    def get_Laplacian(self):
        D = torch.diag(self.adj_tensor @ torch.ones(self.numNodes).view(-1, 1).squeeze())
        L = D - self.adj_tensor
        return L.detach().clone()


    def get_budget(self):
        return self.budget
    
    
    ## budget consumed in each step
    def get_step_budget(self):
        if self.adj_tensor.grad != None:
            ## perturbation = gradient * learning rate
            pert = self.adj_tensor.grad * self.learning_rate
            step_budget = estimate_sym_specNorm(pert)
            return step_budget


    # update how much budget used
    def update_used_budget(self, used_b):
        self.used_budget += used_b.squeeze()


    # return the amount of budget consumed
    def get_used_budget(self):
        return self.used_budget

    def get_result(self):
        return ( self.lambda1_S, self.impact_S, self.centrality )
    
    def get_attacked_adj(self):
        return self.adj_tensor.detach().clone()

    def get_utility(self):
        return -1 * self.Loss

    ## check budget constraint 
    def check_constraint(self, extTensor=[]):
        if extTensor:
            pert = extTensor[0] - self.original_adj
        else:
            pert = self.adj_tensor - self.original_adj
        
        spec_norm = estimate_sym_specNorm(pert)
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





