import matplotlib.pyplot as plt

def relu(x):
    return max(x, 0)

def target(x):
    return min(abs(x+2), abs(x))

if __name__ == '__main__':
    plt.figure()
    plt.plot([x for x in range(-5, 5)], [target(x) for x in range(-5, 5)])
    plt.axhline(y=0, color='k')
    plt.axvline(x=0, color='k')
    plt.title("target curve")
    plt.xlabel("input")
    plt.ylabel("output")
    plt.grid()
    plt.savefig("target-curve.png")
    plt.clf()
