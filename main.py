import argparse
import numpy as np  # noqa
import matplotlib.pyplot as plt  # noqa

# todo: plot covariance
# todo: compare before optimization to after that.


def infer():
    """
    カーネル関数のパラメータを最適化した上で, 推論する.
    """
    # data
    N = 100
    x = np.linspace(0, 100, N)
    y = 5*np.sin(np.pi/15*x)*np.exp(-x/50)
    M = 200
    x_new = np.linspace(0, 100, M)
    # initialize and optimize kernel
    mode = 'dnn'
    if mode == 'rbf':
        initial_param = [2, 2, 2]
        jump_limit = [[0.01, 100], [0.01, 100], [0.01, 100]]
        kernel = RbfKernel(initial_param, jump_limit)
    elif mode == 'dnn':
        initial_param = [1, 2]
        jump_limit = [[0.01, 100], [0.01, 100]]
        kernel = DnnKernel(initial_param, jump_limit)
    print('optimizing kernel paramters...')
    mcmc = MetroPolice()
    # todo: iter_sizeを100に
    kernel.optimize(x, y, mcmc, log_likelihood, iter_size=10)
    print('done.')
    print('optimized param:', kernel.param)
    # predict
    print('calculate inverse matrix...')
    mu = np.zeros((M))
    sigma2 = np.zeros((M))
    k00 = calc_kernel_matrix(x, kernel)
    k00_inv = np.linalg.inv(k00)
    sigma2_y = 0.1
    for m in range(M):
        k10 = calc_kernel_sequence(x, x_new[m], kernel)
        mu[m] = np.dot(np.dot(k10.T, k00_inv), y)
        sigma2[m] = sigma2_y + kernel(x_new[m], x_new[m]) - np.dot(np.dot(k10.T, k00_inv), k10)
    print('done.')
    plt.plot(x_new, mu, marker='o', color='blue')
    plt.plot(x, y, marker='o', color='green')
    plt.show()


def calc_kernel_matrix(X, kernel):
    N = len(X)
    K = np.zeros((N, N))
    for n1 in range(N):
        for n2 in range(n1, N):
            K[n1, n2] = kernel(X[n1], X[n2])
    K = K + K.T - np.diag(np.diagonal(K))
    return K


def calc_kernel_sequence(X, x, kernel):
    N = len(X)
    seq = np.asarray([kernel(X[n], x) for n in range(N)])
    return seq


def log_likelihood(x: np.ndarray, y: np.ndarray, kernel: 'function'):  # noqa: F821
    k00 = calc_kernel_matrix(x, kernel)
    k00_inv = np.linalg.inv(k00)
    return -(np.linalg.slogdet(k00)[1]+y.dot(k00_inv.dot(y)))


class MCMC(object):
    def __init__(self):
        self.params, self.probs = [], []

    def __call__(self):
        raise NotImplementedError


class Kernel(object):
    def __init__(self, param, jump_limit):
        self.param = param
        self.jump_limit = jump_limit

    def __call__(self):
        raise NotImplementedError

    def optimize(self, x: np.ndarray, y: np.ndarray, mcmc: MCMC, log_likelihood: 'log_likelihood', iter_size: int = 1000):
        mcmc(x, y, self, log_likelihood, iter_size)
        self.param = np.exp(mcmc.params[np.argmax(mcmc.probs)])


class MetroPolice(MCMC):
    def __init__(self):
        super(MetroPolice, self).__init__()

    def __call__(self, x: np.ndarray, y: np.ndarray, kernel: Kernel, log_likelihood: 'log_likelihood', iter_size=1000):
        """
        Metropolice-Hastings Algorithm

        1. Sample jump size from gaussisan distribution. If the sampling value exceeded jump size limit, repeat to sample until the value less than the limit.
        2. Add jump size to current parameters. the result values is called as 'new parameters'. If the following condition was satisfied, parameters position is updated.  # noqa: E501
           ```
           if new_likelihood > likelihood or (new_likelihood / likelihood) / numpy.random.rand():
               parameters = new_parametrers
           ```
        3. Repeat 1 - 2.
        """
        log_jump_limit = np.log(kernel.jump_limit)  # jump size limit of for each parameter in once iteration. the shape is (num_param, 2).
        std_dev = (log_jump_limit[:, 1] - log_jump_limit[:, 0]) / 10.  # standard deviation of gaussian distribution for sampling jump size.
        num_params = len(kernel.param)
        param = np.log(kernel.param)  # start position of optimizing target parameters.
        prob = log_likelihood(x, y, kernel)  # log likelihood on the current parameters (= parameters at start position).
        print('sampling with mcmc...')
        for i in range(iter_size):
            jump_size = np.random.normal(0, std_dev, num_params)
            over_the_limit: bool = np.any((param + jump_size < log_jump_limit[:, 0]) | (param + jump_size > log_jump_limit[:, 1]))
            while over_the_limit:
                jump_size[over_the_limit] = np.random.normal(0, std_dev, num_params)[over_the_limit]
                over_the_limit = np.any((param + jump_size < log_jump_limit[:, 0]) | (param + jump_size > log_jump_limit[:, 1]))
            new_param = param + jump_size
            kernel.param = np.exp(new_param)
            new_prob = log_likelihood(x, y, kernel)
            if(new_prob > prob or np.exp(new_prob - prob) > np.random.rand()):
                print(f'get new candidate parameters: {new_param}')
                param, prob = new_param, new_prob
                self.params.append(param)
                self.probs.append(prob)
            print(f'iter: {i}')
            if i > 0 and i % 5 == 0:
                print(f'iter: {i}/{iter_size}')
        print('done.')


class RbfKernel(Kernel):
    def __init__(self, param, jump_limit):
        super(RbfKernel, self).__init__(param, jump_limit)

    def __call__(self, x1, x2):
        a1, s, a2 = self.param
        return a1**2*np.exp(-0.5*((x1-x2)/s)**2) + a2**2*(x1 == x2)


class DnnKernel(Kernel):
    """ a kernel for gaussian processes as DNN """
    def __init__(self, param, jump_limit, num_layer=2):
        super(DnnKernel, self).__init__(param, jump_limit)
        self.num_layer = num_layer

    def __call__(self, x1, x2):
        def _kernel(l, x1, x2, w, b):
            def theta(l, x1, x2):
                return np.arccos(
                        _kernel(l, x1, x2, w, b) /
                        np.sqrt(_kernel(l, x1, x1, w, b) * _kernel(l, x2, x2, w, b))
                    )
            if l > 0:
                return b + (w/(2*np.pi)) * np.sqrt(_kernel(l-1, x1, x1, w, b) * _kernel(l-1, x2, x2, w, b)) * (np.sin(theta(l-1, x1, x2)) + (np.pi-theta(l-1, x1, x2)) * np.cos(theta(l-1, x1, x2)))  # noqa: E501
            else:
                return b + w*(x1*x2)
        sigma_w, sigma_b = self.param
        return _kernel(self.num_layer, x1, x2, sigma_w, sigma_b)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Initialize trainning parameters.')
    parser.add_argument('-mode', type=str, help='you can choose kernel mode from "dnn" or "rbf". Default is dnn kernel.')
    args = parser.parse_args()
    if args.mode:
        assert args.mode in ['dnn', 'rbf'], f'{args.mode} mode is not implemented. please choose from "dnn" or "rbf"'
    mode = args.mode if args.mode else 'dnn'
    infer()
