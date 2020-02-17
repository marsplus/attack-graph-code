import sys
import numpy as np
import cvxpy as cvx
import numpy.linalg as LIN


# n = 3
# A = np.random.rand(n, n)
# A = (A + A.T) / 2
# A = A - np.diag(np.diag(A))
# epsilon = LIN.norm(A, 'f') + 0.5
# l_A = LIN.eig(A)[0].max()

# B = np.random.rand(n, n) * 100
# B = (B + B) / 2

# X_opt = cvx.Variable((n, n), symmetric=True)
# obj = cvx.norm(X_opt - B, 'fro')
# cons = [cvx.lambda_max(X_opt) - l_A <= epsilon]
# prob = cvx.Problem(cvx.Minimize(obj), cons)
# prob.solve()
# X_opt = X_opt.value
# print(LIN.eig(X_opt)[0].max() - l_A, epsilon)

Lambda = float(sys.argv[1])

A = np.array([[0, 1, 1], [1, 0, 0], [1, 0, 0]])
L_A = LIN.eig(A)[0].max()

B = np.array([[0, 1.15, 1.6], [1.15, 0, 0], [1.6, 0, 0]])
L_B = LIN.eig(B)[0].max()

C = np.array([[0, 0.6, 0.6], [0.6, 0, 0], [0.6, 0, 0]])
L_C = LIN.eig(C)[0].max()

D = Lambda * B + (1 - Lambda) * C
L_D = LIN.eig(D)[0].max()


E = (Lambda - 0.1) * B + (1 - Lambda + 0.1) * C
L_E = LIN.eig(E)[0].max()

print(L_A, L_D, L_E)

#print("|L_A - L_C|: {:.4f}".format(np.abs(L_A - L_C)))
#print("|L_A - L_B|: {:.4f}".format(np.abs(L_A - L_B)))
#print("|L_A - L_D|: {:.4f}".format(np.abs(L_A - L_D)))
#



