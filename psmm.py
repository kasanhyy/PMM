import numpy as np
from scipy import optimize
from sklearn.datasets._samples_generator import make_blobs
import matplotlib.pyplot as plt


class subregion:
    def __init__(self, a, b):
        self.smallest = a  # A d-tuple indicating the smallest point in the region
        self.largest = b  # A d-tuple indicating the largest point in the region
        self.count = 0
        self.noisy = 0

    def split(self):
        a = self.smallest
        b = self.largest
        split_pos = np.argmax(b - a)
        a_new = np.copy(a)
        b_new = np.copy(b)
        a_new[split_pos] = (a[split_pos] + b[split_pos]) / 2
        b_new[split_pos] = (a[split_pos] + b[split_pos]) / 2
        return [subregion(a, b_new), subregion(a_new, b)]

    def countTrueData(self, data):
        a = self.smallest
        b = self.largest
        indicator = (a <= data).all(axis=1) & (data < b).all(axis=1)
        self.count = np.count_nonzero(indicator)

    def addNoise(self, eps=1):
        noise = np.random.laplace(scale=1 / eps)
        # if noise >= 0:
        #     noise = np.floor(noise)
        # else:
        #     noise = np.ceil(noise)
        self.noisy = noise + self.count

    def centerPoint(self):
        return (self.smallest + self.largest) / 2


def distanceMatrix(data):
    m = len(data)
    dist = np.zeros((m, m))
    for i in range(m):
        for j in range(i + 1, m):
            dist[i, j] = max(np.abs(data[i] - data[j]))
            dist[j, i] = dist[i, j]
    return dist


def psmm_measure(true_data, eps=1):
    n, d = true_data.shape
    m = np.ceil(eps * n)

    regions = [subregion(np.zeros(d), np.ones(d))]
    while len(regions) < m:
        temp = regions.pop(0)
        regions += temp.split()
    m = len(regions)

    for region in regions:
        region.countTrueData(true_data)
        region.addNoise(eps)

    # Start linear programming
    Y = [region.centerPoint() for region in regions]
    dist = distanceMatrix(Y)
    c = np.zeros(2 * m * m)
    A = np.zeros((m, 2 * m * m))
    b = np.zeros(m)
    Aeq = np.zeros((1, 2 * m * m))
    beq = 1
    for i in range(m):
        temp = dist[i, :]
        temp = np.delete(temp, i)

        c[i * (m - 1):(i + 1) * (m - 1)] = temp
        c[(m + i) * (m - 1):(m + i + 1) * (m - 1)] = temp

        A[i, i * (m - 1):(i + 1) * (m - 1)] = 1  # u_ij
        A[i, (m + i) * (m - 1):(m + i + 1) * (m - 1)] = -1  # u_ij'
        A[i, 2 * m * (m - 1) + i] = 1  # v_i
        A[i, 2 * m * (m - 1) + m + i] = 1  # tau_i
        b[i] = regions[i].noisy / n
    c[2 * m * (m - 1): 2 * m * (m - 1) + m] = 2
    Aeq[0, -m:] = 1
    A = -A
    b = -b

    res = optimize.linprog(c, A, b, Aeq, beq)
    nu = res.x[-m:]
    return regions, nu


def sampling(mu, regions, n):
    d = len(regions[0].smallest)
    numbers = np.round(mu * n)
    p = 0
    syn = np.zeros((round(sum(numbers)), d))
    for i, r in enumerate(regions):
        a = r.smallest
        b = r.largest
        ni = round(numbers[i])
        syn[p: p + ni] = np.random.rand(ni, d) * (b - a) + a
        p += ni
    return syn


def psmm_data(true_data, eps=1):
    n,d = true_data.shape
    regions, nu = psmm_measure(true_data, eps)
    syn = sampling(nu, regions, n)
    return syn