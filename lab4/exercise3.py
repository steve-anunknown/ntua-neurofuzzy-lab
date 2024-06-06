import matplotlib.pyplot as plt

N = 200
C = 100


def c(k):
    return N - k


def p(k):
    return 0.05


res = [0] * (N+1)

res[N] = C

for k in range(N-1, -1, -1):
    res[k] = p(k) * c(k) + (1-p(k)) * res[k+1]


# Print the seat number with the minimum expected cost
print('Seat:', res.index(min(res))+1, ' Ecpected Cost:', min(res))

plt.figure()
plt.plot(range(N+1), res)
plt.xlabel('Expected Cost')
plt.ylabel('Parking Seats')
plt.title('Parking Seats vs Expected Cost')
plt.grid()
plt.savefig('exercise3.png')

