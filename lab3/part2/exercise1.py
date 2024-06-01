import matplotlib.pyplot as plt

def relu(x):
    return max(x, 0)

def target(x):
    return abs(abs(x + 1) - 1)

def neural_network(x):
    h1 = relu(-2*x - 4)
    h2 = relu(-2*x - 2)
    h3 = relu(-x)
    h4 = relu(x)
    return relu(h1 - h2 + h3 + h4)

if __name__ == '__main__':
    plt.figure()
    epsilon = 0.1
    plt.plot([x for x in range(-5, 5)], 
             [target(x) for x in range(-5, 5)])
    plt.plot([x for x in range(-5, 5)], 
             [neural_network(x) + epsilon for x in range(-5, 5)])
    plt.legend(["target", "neural network"])
    plt.axhline(y=0, color='k')
    plt.axvline(x=0, color='k')
    plt.title("target curve")
    plt.xlabel("input")
    plt.ylabel("output")
    plt.grid()
    plt.savefig("target-curve.png")
    plt.clf()
