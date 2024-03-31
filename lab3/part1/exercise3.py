import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    dims = np.arange(2, 100)
    cond_numbers_harmonic = []
    cond_numbers_arithmetic = []
    cond_numbers_geometric = []
    cond_numbers_median = []
    
    reps = 400
    
    for dim in dims:
        cond_list = []
        for i in range(reps):
            # A random matrix
            A = np.random.randn(dim, dim)
    
            # A * A^T
            Q = A @ A.T
    
            # condition number of Q
            cond = np.linalg.cond(Q)
            cond_list.append(cond)
    
        # use harmonic mean
        mean_cond = reps / np.sum(1 / np.array(cond_list))
        cond_numbers_harmonic.append(mean_cond)
        # use arithmetic mean
        mean_cond = np.mean(cond_list)
        cond_numbers_arithmetic.append(mean_cond)
        # use geometric mean
        mean_cond = np.exp(np.mean(np.log(cond_list)))
        cond_numbers_geometric.append(mean_cond)
        # use median
        mean_cond = np.median(cond_list)
        cond_numbers_median.append(mean_cond)
    
    
    plt.figure()
    plt.plot(dims, cond_numbers_arithmetic, 'o-')
    plt.plot(dims, cond_numbers_geometric, 'o-')
    plt.plot(dims, cond_numbers_harmonic, 'o-')
    plt.plot(dims, cond_numbers_median, 'o-')
    plt.legend(['Arithmetic mean', 'Geometric mean', 'Harmonic mean', 'Median'])
    #plt.legend(['Geometric mean', 'Harmonic mean', 'Median'])
    plt.xlabel('Dimension')
    plt.ylabel('Condition number')
    plt.title('Condition number of Q')
    plt.yscale('log')
    #plt.xscale('log')
    plt.savefig("condition_number_vs_dims.png")
    plt.show()
