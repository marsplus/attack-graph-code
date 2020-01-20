import torch
import torch.nn as nn
import networkx as nx


class Threat_Model(nn.Module):
    def __init__(self, S, S_prime, Alpha, budget, learning_rate, G):
        super(Threat_Model, self).__init__()
        self.numNodes = len(G)

        self.S = S
        # S_prime = V \ S
        self.S_prime = S_prime
        self.alpha_1, self.alpha_2, self.alpha_3 = Alpha
        self.budget = budget
        self.learning_rate = learning_rate

        # haven't took budget constraint into account
        self.used_budget = 0
        
        self.lambda1_S_prime = 0
        self.lambda1_S = 0
        self.normalizedCut = 0
        self.lambda1 = 0
        
        adj = nx.adjacency_matrix(G).todense()
        self.mask = torch.tensor(adj, dtype=torch.float32)
        eigVals, eigVecs = torch.symeig(self.mask, eigenvectors=True)
        # eig-vector associated with the largest eig-value of A (since mask = A)
        self.eig_v = eigVecs[:, -1]
        self.original_lambda1 = eigVals[-1]
        
        self.adj_tensor = torch.tensor(adj, dtype=torch.float32).requires_grad_(True)
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
    
        # select the sub-matrix corresponding to S
        adj_tensor_S = torch.index_select(torch.index_select(self.adj_tensor, 0, self.S), 1, self.S)
        adj_tensor_S_prime = torch.index_select(torch.index_select(self.adj_tensor, 0, self.S_prime), 1, self.S_prime)
    
        # all sorts of largest eigenvalues 
        self.lambda1 = torch.max(torch.symeig(self.adj_tensor, eigenvectors=True)[0])
        self.lambda1_S = torch.max(torch.symeig(adj_tensor_S, eigenvectors=True)[0])
        self.lambda1_S_prime = torch.max(torch.symeig(adj_tensor_S_prime, eigenvectors=True)[0])
        
        # centrality measure
        self.normalizedCut = torch.mm(x_s.view(1, -1), torch.mm(L, x_s.view(-1, 1))) / (0.5 * adj_tensor_S.sum())
        Loss = -1 * (self.alpha_1 * self.lambda1_S - \
                     self.alpha_2 * self.lambda1_S_prime + self.alpha_3 * self.normalizedCut)
        return Loss
    
    
    # check if the attacker still has budget (haven't took budget constraint into account)
    def update_used_budget(self):
        # make sure the attacker has made some attack
        if self.adj_tensor.grad != None:
            current_used_budget = \
                torch.abs(torch.mm(self.eig_v.view(1, -1), torch.mm(self.aggre_change, self.eig_v.view(-1, 1))))
            self.used_budget += current_used_budget 

    
    # return the results that we are interested in
    def getRet(self):
        return self.lambda1_S, self.lambda1, self.normalizedCut
    
    
    def get_used_budget(self):
        return torch.Tensor(self.used_budget).clone()
