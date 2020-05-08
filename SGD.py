import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def structure_data():
    data = np.loadtxt('iris_data.csv', delimiter=',', dtype={'names': ('sl', 'sw', 'pl', 'cl'), 'formats': (np.float,
                                                                                                            np.float,
                                                                                                            np.float,
                                                                                                            "S15")})
    # create an empty array cl
    cl = []
    # go through the lines of the last column (strings) in the data file and replace the strings with the labels 1 and 0
    for lbl in data['cl']:
        lbl = lbl.decode("utf-8")
        if lbl == 'Iris-setosa':
            lbl = 1
        elif lbl == 'Iris-versicolor':
            lbl = 0
    # store the labels in cl
        cl.append(lbl)
    cl = np.asarray(cl)
    cl = cl.reshape(100, 1)
    # name + reshape
    sl = data['sl']
    sl = sl.reshape(100, 1)
    sw = data['sw']
    sw = sw.reshape(100, 1)
    pl = data['pl']
    pl = pl.reshape(100, 1)
    # include the bias
    u = np.ones([100, 1])
    # create a new matrix out of the columns in st_data
    st_data = np.hstack((u, sl, sw, pl, cl))
    x = st_data[:, 0:4]
    y = st_data[:, 4]

    return x, y


def plot(x):

    sl = x[:, 1]
    sw = x[:, 2]
    pl = x[:, 3]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    s = ('x', 0, 49)
    v = ('^', 50, 99)
    for m, i, j in [s, v]:
        xs = sl[i:j]
        ys = pl[i:j]
        zs = sw[i:j]
        ax.scatter(xs, ys, zs, marker=m)

    ax.set_xlabel('sepal length (cm)')
    ax.set_ylabel('petal length (cm)')
    ax.set_zlabel('sepal width (cm)')
    ax.legend(['Iris-setosa', 'Iris-versicolor'])

    scatter_plot = plt.show()
    return scatter_plot


[x, y] = structure_data()

#TASK 1: Produce a 3-d scatterplot of the data (each dimension corresponding to a feature), with the data points colored differently according to their class, and put this plot in your write-up.
# TASK 1 RESULTS:
plot(x)


# define the logistic regression hypothesis
def hypothesize(x, theta):
    z = np.array(theta[0]*x[:, 0]+theta[1]*x[:, 1]+theta[2]*x[:, 2]+theta[3]*x[:, 3])
    h = 1/(1+np.exp(-z))
    return h


def loss(theta, x, y):
    h = hypothesize(x, theta)
    lo = sum(-y*np.log(h)-(1-y)*np.log(1-h))
    return lo


def gradient(theta, x, y):
    h = hypothesize(x, theta)
    g = np.zeros((4, 1))
    g[0] = sum((h-y)*x[:, 0])
    g[1] = sum((h-y)*x[:, 1])
    g[2] = sum((h-y)*x[:, 2])
    g[3] = sum((h-y)*x[:, 3])
    return g


def sgd(reps, eta_init, k):
    # initialize theta
    theta = np.zeros((4, 1))
    # initialize a list for the values of the loss
    losses = []
    # initialize a list for the values of theta
    th = []
    [x, y] = structure_data()

    for t in range(1, reps + 1):
        # randomly draw k samples
        np.random.shuffle([x, y])
        x = x[0:k]
        y = y[0:k]
        # update step rate
        eta = eta_init / np.sqrt(t)
        # update theta
        g = gradient(theta, x, y)
        theta[0] = theta[0] - eta * (1 / k) * g[0]
        theta[1] = theta[1] - eta * (1 / k) * g[1]
        theta[2] = theta[2] - eta * (1 / k) * g[2]
        theta[3] = theta[3] - eta * (1 / k) * g[3]
        # save the loss values
        lo = loss(theta, x, y)
        losses.append(lo)
        # save theta values
        th.append(theta)
    # get theta that minimizes the loss
    arg = np.argmin(losses)
    # retrieve theta that minimizes the loss
    theta_hat = np.array(th[arg])

    return theta_hat


# Task 2 : Implement stochastic gradient descent for logistic regression. Make sure can easily adjust the initial learning rate η, and number of iterations
# TASK 2 RESULTS:
print('n° of iter.:', 100, 'step size:', 0.01, 'batch size:', 20, 'theta_hat:\n', sgd(100, 0.01, 20))


def decision_boundary(theta_hat):
    sl = x[:, 1]
    sw = x[:, 2]
    pl = x[:, 3]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    s = ('x', 0, 49)
    v = ('^', 50, 99)
    for m, i, j in [s, v]:
        xs = sl[i:j]
        ys = pl[i:j]
        zs = sw[i:j]
        ax.scatter(xs, ys, zs, marker=m)

    x1 = [x[:, 1].min(), x[:, 1].max(), x[0, 1]]
    x2 = [x[:, 2].min(), x[:, 2].max(), x[0, 2]]
    x1, x2 = np.meshgrid(x1, x2)
    z = -(theta_hat[0] + theta_hat[1] * x1 + theta_hat[2] * x2) / theta_hat[3]
    # Plot the surface.

    surf = ax.plot_surface(x1, x2, z)

    ax.set_xlabel('sepal length (cm)')
    ax.set_ylabel('petal length (cm)')
    ax.set_zlabel('sepal width (cm)')
    ax.legend(['Iris-setosa', 'Iris-versicolor'])

    boundary_plot = plt.show()
    return boundary_plot


theta_hat = sgd(100000, 0.01, 100)
decision_boundary(theta_hat)
