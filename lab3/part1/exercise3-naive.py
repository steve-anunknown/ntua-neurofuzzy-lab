import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    conds = []
    dims = list(range(1, 300))
    for dim in dims:
        A = np.random.rand(dim, dim)
        Q = A @ A.T
        eigenvalues = np.linalg.eig(Q).eigenvalues
        conds.append(max(eigenvalues)/min(eigenvalues))
    plt.figure()
    plt.plot(dims, conds)
    plt.title("Condition number vs dimensionality")
    plt.xlabel("Dimensions")
    plt.ylabel("Condition number")
    plt.grid()
    plt.savefig("condition_number_vs_dims.png")
    plt.show()


