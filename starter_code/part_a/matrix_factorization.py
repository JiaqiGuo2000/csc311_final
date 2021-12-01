from starter_code.utils import *
from matplotlib import pyplot as plt
from scipy.linalg import sqrtm

import numpy as np


def svd_reconstruct(matrix, k):
    """ Given the matrix, perform singular value decomposition
    to reconstruct the matrix.

    :param matrix: 2D sparse matrix
    :param k: int
    :return: 2D matrix
    """
    # First, you need to fill in the missing values (NaN) to perform SVD.
    # Fill in the missing values using the average on the current item.
    # Note that there are many options to do fill in the
    # missing values (e.g. fill with 0).
    new_matrix = matrix.copy()
    mask = np.isnan(new_matrix)
    masked_matrix = np.ma.masked_array(new_matrix, mask)
    item_means = np.mean(masked_matrix, axis=0)
    new_matrix = masked_matrix.filled(item_means)

    # Next, compute the average and subtract it.
    item_means = np.mean(new_matrix, axis=0)
    mu = np.tile(item_means, (new_matrix.shape[0], 1))
    new_matrix = new_matrix - mu

    # Perform SVD.
    Q, s, Ut = np.linalg.svd(new_matrix, full_matrices=False)
    s = np.diag(s)

    # Choose top k eigenvalues.
    s = s[0:k, 0:k]
    Q = Q[:, 0:k]
    Ut = Ut[0:k, :]
    s_root = sqrtm(s)

    # Reconstruct the matrix.
    reconst_matrix = np.dot(np.dot(Q, s_root), np.dot(s_root, Ut))
    reconst_matrix = reconst_matrix + mu
    return np.array(reconst_matrix)


def squared_error_loss(data, u, z):
    """ Return the squared-error-loss given the data.
    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param u: 2D matrix
    :param z: 2D matrix
    :return: float
    """
    loss = 0
    for i, q in enumerate(data["question_id"]):
        loss += (data["is_correct"][i]
                 - np.sum(u[data["user_id"][i]] * z[q])) ** 2.
    return 0.5 * loss


def update_u_z(train_data, lr, u, z):
    """ Return the updated U and Z after applying
    stochastic gradient descent for matrix completion.

    :param train_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param u: 2D matrix
    :param z: 2D matrix
    :return: (u, z)
    """
    #####################################################################
    # wasTODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    # Randomly select a pair (user_id, question_id).
    i = \
        np.random.choice(len(train_data["question_id"]), 1)[0]

    c = train_data["is_correct"][i]
    n = train_data["user_id"][i]
    q = train_data["question_id"][i]

    u_copy = u.copy()
    z_copy = z.copy()

    u[n][0] -= lr * (c - u_copy[n][0] * z_copy[q][0]) * (- z_copy[q][0])
    z[q][0] -= lr * (c - u_copy[n][0] * z_copy[q][0]) * (- u_copy[n][0])

    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return u, z


def als(train_data, k, lr, num_iteration, val_data=None, plot=False):
    """ Performs ALS algorithm, here we use the iterative solution - SGD 
    rather than the direct solution.

    :param train_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param k: int
    :param lr: float
    :param num_iteration: int
    :param plot: bool
    :return: 2D reconstructed Matrix.
    """
    # Initialize u and z
    u = np.random.uniform(low=0, high=1 / np.sqrt(k),
                          size=(len(set(train_data["user_id"])), k))
    z = np.random.uniform(low=0, high=1 / np.sqrt(k),
                          size=(len(set(train_data["question_id"])), k))

    #####################################################################
    # wasTODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    train_losses = []
    val_losses = []
    for iteration in range(num_iteration):
        u, z = update_u_z(train_data, lr, u, z)
        train_loss = 0
        val_loss = 0
        if plot:
            print(iteration)
            for i in range(len(train_data["question_id"])):
                train_loss += 0.5 * ((train_data["is_correct"][i] - u[train_data["user_id"][i]][0] * z[train_data["question_id"][i]][0]) ** 2)
            for i in range(len(val_data["question_id"])):
                val_loss += 0.5 * ((val_data["is_correct"][i] - u[val_data["user_id"][i]][0] * z[val_data["question_id"][i]][0]) ** 2)
            train_losses.append(train_loss)
            val_losses.append(val_loss)

    mat = np.matmul(u, z.T)

    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return mat, train_losses, val_losses


def main():
    train_matrix = load_train_sparse("../data").toarray()
    train_data = load_train_csv("../data")
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")

    #####################################################################
    # wasTODO:                                                             #
    # (SVD) Try out at least 5 different k and select the best k        #
    # using the validation set.                                         #
    #####################################################################
    k_values = [3, 5, 7, 9, 11, 13, 15]
    for k in k_values:
        reconst_matrix = svd_reconstruct(train_matrix, k)
        val_acc = sparse_matrix_evaluate(val_data, reconst_matrix)
        print("k = {}, \t validation accuracy = {}".format(k, val_acc))

    reconst_matrix = svd_reconstruct(train_matrix, 9)
    val_acc = sparse_matrix_evaluate(val_data, reconst_matrix)
    test_acc = sparse_matrix_evaluate(test_data, reconst_matrix)
    print("final k = 9, \t validation accuracy = {}, \t test accuracy = {}".format(val_acc, test_acc))

    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################

    #####################################################################
    # wasTODO:                                                             #
    # (ALS) Try out at least 5 different k and select the best k        #
    # using the validation set.                                         #
    #####################################################################

    k_values = [1, 2, 3, 4, 5]
    lr = 0.05
    iterations = 100000
    final_matrix = None
    final_train_losses = []
    final_val_losses = []
    for k in k_values:
        if k == 1:
            plot = True
        else:
            plot = False
        reconst_matrix, train_losses, val_losses = als(train_data, k, lr, iterations, val_data, plot)
        if k == 1:
            final_matrix = reconst_matrix
            final_train_losses = train_losses
            final_val_losses = val_losses
        val_acc = sparse_matrix_evaluate(val_data, reconst_matrix)
        print("k = {}, \t validation accuracy = {}".format(k, val_acc))

    val_acc = sparse_matrix_evaluate(val_data, final_matrix)
    test_acc = sparse_matrix_evaluate(test_data, final_matrix)
    print("final k = 1, \t validation accuracy = {}, \t test accuracy = {}".format(val_acc, test_acc))

    plt.ylabel("loss")
    plt.xlabel("iterations")
    plt.plot(range(iterations), final_val_losses, label="validation error")
    plt.plot(range(iterations), final_train_losses, label="training error")
    plt.title("square error to iteration")
    plt.legend()
    plt.show()

    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    main()
