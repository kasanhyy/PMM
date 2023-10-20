import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets._samples_generator import make_blobs, make_regression
from pmm import pmm_data
from psmm import psmm_data
from mpl_toolkits.mplot3d import Axes3D
import ot.plot
from scipy.stats import linregress


# Generate raw data.
# Change this function to change distribution
def generateRawData(n, d, datatype="blobs"):
    if datatype == "blobs":
        data, label = make_blobs(n_samples=n, n_features=d, centers=[[0.4] * d, [0.7] * d],
                                 cluster_std=[0.1, 0.05])
    elif datatype == "uniform":
        data = np.random.rand(n, d)

    return data


def computeW1(data1, data2, plot_data_pts=False, plot_ot=False):
    n1 = len(data1)
    n2 = len(data2)
    d = len(data1[0])

    if plot_data_pts:
        plt.figure(1)
        plt.subplot(1, 2, 1)
        plt.scatter(data1[:, 0], data1[:, 1], marker='o', s=3)
        plt.xlim((0, 1))
        plt.ylim((0, 1))
        plt.title("True data")

        plt.subplot(1, 2, 2)
        plt.scatter(data2[:, 0], data2[:, 1], marker='o', s=3)
        plt.xlim((0, 1))
        plt.ylim((0, 1))
        plt.title("Synthetic data")
        plt.show()

    # This is the distance matrix of the data
    dist = ot.dist(data1, data2, metric="chebyshev")
    # Consider uniform distribution of data, i.e. empirical distribution
    uniform = np.ones((n1,)) / n1
    uniform_syn = np.ones((n2,)) / n2
    # sol is the solution to OT
    sol = ot.emd(uniform, uniform_syn, dist)

    if plot_ot:
        # plt.figure(2)
        # plt.imshow(sol, interpolation='nearest')
        # plt.title('OT matrix sol')

        plt.figure(2)
        ot.plot.plot2D_samples_mat(data1, data2, sol, c=[.5, .5, 1])
        plt.plot(data1[:, 0], data1[:, 1], '+b', label='True data')
        plt.plot(data2[:, 0], data2[:, 1], 'xr', label='Synthetic data')
        plt.legend(loc=0)
        plt.xlim((0,1))
        plt.ylim((0,1))
        plt.title('OT matrix with samples')
        plt.show()

    w1dist = np.linalg.norm(sol * dist, ord="fro")
    print("The Wasserstein distance is %.3e" % w1dist, "with n = %d" % n1, end=' ')
    return w1dist


# draw W1-eps graph if eps_list!=None, otherwise draw W1-n
# datatype is the way to generate true data:
def plotW1(d, n_list=np.array([250, 500, 1000, 2000, 4000]), eps_default=1, eps_list=None, n_default=2500,
           datatype="blobs", method=pmm_data, plot_data_pts=False, plot_ot=False):
    accuracy = []

    if eps_list is None:
        for n in n_list:
            data = generateRawData(n, d, datatype)
            syn_data = method(data, eps_default)  # PMM!
            accuracy.append(computeW1(data, syn_data, plot_data_pts=plot_data_pts, plot_ot=plot_ot))
            print("epsilon = %.1f" % eps_default)
        x = np.log(n_list)
        plt.xlabel("log(n)")
        plt.title("W1-dist vs number of data")
    else:
        n = n_default
        for eps in eps_list:
            data = generateRawData(n, d, datatype)
            syn_data = method(data, eps)  # PMM!
            accuracy.append(computeW1(data, syn_data, plot_data_pts=plot_data_pts, plot_ot=plot_ot))
            print("epsilon = %.1f" % eps)
        x = np.log(eps_list)
        plt.xlabel("log(epsilon)")
        plt.title("W1-dist vs privacy parameter")

    y = np.log(accuracy)
    plt.scatter(x, y, c='b')
    slope, intercept, r_value, p_value, std_err = linregress(x, y)
    x_line = np.arange(x[0] - 0.5, x[-1] + 0.5, 0.1)
    y_line = slope * x_line + intercept
    plt.plot(x_line, y_line, c='r')
    plt.ylabel("log(W1)")
    plt.show()
    print("slope =", slope)
    print("intercept = ", intercept)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    dim = 4
    method = pmm_data
    n_list = np.array([250, 500, 1000, 2000, 4000, 8000])
    eps_list = np.array([1, 2, 4, 8, 16])
    plotW1(dim, eps_list=eps_list, n_default=400, method=method, datatype="blobs")
    # plotW1(dim, n_list=n_list, method=method, datatype="uniform")
