import torch
import torch.nn as nn
import networkx as nx


class Threat_Model(nn.Module):
    def __init__(self, S, S_prime, Alpha, budget, learning_rate, G):
        super(Threat_Model, self).__init__()
        self.numNodes = len(G)

        self.S = S
        self.S_prime = S_prime
        
        self.alpha_1, self.alpha_2, self.alpha_3 = Alpha
        self.budget = budget
        self.learning_rate = learning_rate

        # tracks the amount of budget used
        self.used_budget = torch.zeros(1)
        
        # those eigenvalues in the objective function
        self.lambda1_S_prime = 0
        self.lambda1_S = 0
        self.normalizedCut = 0
        self.lambda1 = 0
        
        # the pristine adjacency matrix
        adj = nx.adjacency_matrix(G).todense()
        self.mask = torch.tensor(adj, dtype=torch.float32)
        
        # eigenvals and eigenvectors associated with the largest eig-value of adj (since mask = adj)
        # the eigenvalues returned by torch.symeig() are in ascending order
        # so the last one is the largest eigenvalue we want 
        eigVals, eigVecs = torch.symeig(self.mask, eigenvectors=True)
        self.eig_v = eigVecs[:, -1]
        self.lambda1_original = eigVals[-1]
       
        # the thing we wanna optimize
        # requires_grad_(True): tells PyTorch to starting tracking the gradients of this parameter
        self.adj_tensor = torch.tensor(adj, dtype=torch.float32).requires_grad_(True)
        # tell PyTorch that self.adj_tensor is a parameter of Threat_Model, which we will optimize over
        self.adj_tensor = nn.Parameter(self.adj_tensor)
        
        # masking the gradients backpropagated to adj_tensor
        self.adj_tensor.register_hook(lambda x: x * self.mask)
        
    def forward(self):
        """
            Compute loss given current (perturbed) adjacency matrix
        """
        # degree matrix
        D = torch.diag(torch.mm(self.adj_tensor, torch.ones(self.numNodes).view(-1, 1)).squeeze())
        # Laplacian matrix
        L = D - self.adj_tensor

        # characteristic vector for the set S
        x_s = torch.zeros(self.numNodes)
        x_s[self.S] = 1
    
        # select the sub adjacency matrix corresponding to S and S_prime
        adj_tensor_S = torch.index_select(torch.index_select(self.adj_tensor, 0, self.S), 1, self.S)
        adj_tensor_S_prime = torch.index_select(torch.index_select(self.adj_tensor, 0, self.S_prime), 1, self.S_prime)
    
        # all sorts of largest eigenvalues 
        # Note: need to set eigenvectors=True when calling torch.symeig, which is 
        # required in order to do automatic differentiation
        # eigValues, eigVecs = torch.symeig()
        self.lambda1 = torch.max(torch.symeig(self.adj_tensor, eigenvectors=True)[0])
        self.lambda1_S = torch.max(torch.symeig(adj_tensor_S, eigenvectors=True)[0])
        self.lambda1_S_prime = torch.max(torch.symeig(adj_tensor_S_prime, eigenvectors=True)[0])
        
        # centrality measure
        # torch.mm(): do matrix multiplication
        # x_s.view(1, -1): convert x_s to a row vector
        normalization_const = (0.5 * adj_tensor_S.sum())
        self.normalizedCut = torch.mm(x_s.view(1, -1), torch.mm(L, x_s.view(-1, 1))) / normalization_const
        
        # loss function (the negative of U_a)
        # since we defined self.adj_tensor as a parameter, PyTorch automatically tracks 
        # the functional relation between Loss and self.adj_tensor, and correspondingly computes derivatives
        # of Loss w.r.t self.adj_tensor 
        Loss = -1 * (self.alpha_1 * self.lambda1_S - \
                     self.alpha_2 * self.lambda1_S_prime + self.alpha_3 * self.normalizedCut)
        
        return Loss

    def get_budget(self):
        return self.budget
    
    # budget consumed in each step
    def get_step_budget(self):
        if self.adj_tensor.grad != None:
            # perturbation = gradient x learning rate
            pert = self.adj_tensor.grad * self.learning_rate
            # budget used in this step is the operator norm of pert
            step_budget = torch.max(torch.abs(torch.symeig(pert)[0]))
            return step_budget

    # update how much budget used
    def update_used_budget(self, used_b):
        self.used_budget += used_b

    # return the amount of budget consumed
    def get_used_budget(self):
        return self.used_budget.clone()

    def getRet(self):
        return self.lambda1_S, self.lambda1, self.normalizedCut
    
    def get_attacked_adj(self):
        return self.adj_tensor.clone()

    # check budget constraint (for debug purpose)
    def check_constraint(self):
        return torch.abs(self.lambda1 - self.lambda1_original) <= self.budget
    
