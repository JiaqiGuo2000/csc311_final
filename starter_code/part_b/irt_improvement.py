from starter_code.utils import *

import numpy as np
import matplotlib.pyplot as plt


def sigmoid(x):
    """ Apply sigmoid function.
    """
    return np.exp(x) / (1 + np.exp(x))


def neg_log_likelihood(data, theta, beta, a_list, b_list):
    """ Compute the negative log-likelihood.

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param theta: Vector
    :param beta: Vector
    :return: float
    """
    #####################################################################
    # wasTODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    log_lklihood = 0.
    for i in range(len(data["user_id"])):
        frac = np.exp(theta[data["user_id"][i]] -
                      beta[data["question_id"][i]])
        a = a_list[data["question_id"][i]]
        b = b_list[data["question_id"][i]]
        b_frac = b * frac / (frac + 1)
        c = data["is_correct"][i]
        #diff = theta[data["user_id"][i]] - beta[data["question_id"][i]]
        #print(b_frac)
        #print(a + b_frac)
        log_lklihood += c * np.log(a + b_frac) + (1-c) * np.log(1 - a - b_frac)
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return -log_lklihood


def update_theta_beta(data, lr, theta, beta, a_list, b_list):
    """ Update theta and beta using gradient descent.

    You are using alternating gradient descent. Your update should look:
    for i in iterations ...
        theta <- new_theta
        beta <- new_beta

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param theta: Vector
    :param beta: Vector
    :return: tuple of vectors
    """
    #####################################################################
    # wasTODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################

    theta_copy = theta.copy()
    beta_copy = beta.copy()
    a_copy = a_list.copy()
    b_copy = b_list.copy()

    for i in range(len(data["user_id"])):
        frac = np.exp(theta_copy[data["user_id"][i]] -
                      beta_copy[data["question_id"][i]])
        #theta[data["user_id"][i]] += lr * (data["is_correct"][i] -
        #                                   (frac / (1 + frac)))
        c = data["is_correct"][i]
        a = a_copy[data["question_id"][i]]
        b = b_copy[data["question_id"][i]]
        b_frac = b * frac / (frac + 1)
        a_list[data["question_id"][i]] += lr * (c / (a + b_frac) - (1 - c) / (1 - a - b_frac))
        b_list[data["question_id"][i]] -= lr * (c / (a + b_frac) - (1 - c) / (1 - a - b_frac))
        theta[data["user_id"][i]] += lr * ((1 - c) * (b_frac ** 2 - b_frac) / (-b_frac - a + 1) + c * (b_frac - b_frac ** 2) / (b_frac + a))
        beta[data["question_id"][i]] += - lr * ((1 - c) * (b_frac ** 2 - b_frac) / (-b_frac - a + 1) + c * (b_frac - b_frac ** 2) / (b_frac + a))
    for i in range(len(data["user_id"])):
        if a_list[data["question_id"][i]] <= 0:
            a_list[data["question_id"][i]] = 0.001
        if a_list[data["question_id"][i]] >= 1:
            a_list[data["question_id"][i]] = 0.999
        if a_list[data["question_id"][i]] + b_list[data["question_id"][i]] >= 1:
            b_list[data["question_id"][i]] = 1 - a_list[data["question_id"][i]]
        if a_list[data["question_id"][i]] + b_list[data["question_id"][i]] <= 0:
            b_list[data["question_id"][i]] = 0.001

    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return theta, beta, a_list, b_list


def irt(data, val_data, lr, iterations, quiet=False):
    """ Train IRT model.

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param val_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param iterations: int
    :param quiet: bool
    :return: (theta, beta, val_acc_lst)
    """
    # wasTODO: Initialize theta and beta.
    theta = np.ones(542) * 0.1
    beta = np.ones(1774) * 0.1
    a = np.ones(1774) * 0.7
    b = np.ones(1774) * 0.3

    validation_log_likelihood = []
    train_log_likelihood = []

    for i in range(iterations):
        train_neg_lld = neg_log_likelihood(data, theta=theta, beta=beta, a_list=a, b_list=b)
        train_log_likelihood.append(train_neg_lld)
        validation_neg_lld = neg_log_likelihood(val_data, theta=theta,
                                                beta=beta, a_list=a, b_list=b)
        validation_log_likelihood.append(validation_neg_lld)
        score_train = evaluate(data=data, theta=theta, beta=beta, a_list=a, b_list=b)
        score_validation = evaluate(data=val_data, theta=theta, beta=beta, a_list=a, b_list=b)
        # val_acc_lst.append(score_train)
        if not quiet:
            print("NLLK: {} \t Train Score: {} \t Validation Score: {}".format(
                train_neg_lld, score_train, score_validation))
        theta, beta, a, b = update_theta_beta(data, lr, theta, beta, a, b)

    # wasTODO: You may change the return values to achieve what you want.
    return theta, beta, a, b, validation_log_likelihood, train_log_likelihood


def evaluate(data, theta, beta, a_list, b_list):
    """ Evaluate the model given data and return the accuracy.
    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}

    :param theta: Vector
    :param beta: Vector
    :return: float
    """
    pred = []
    for i, q in enumerate(data["question_id"]):
        u = data["user_id"][i]
        a = a_list[data["question_id"][i]]
        b = b_list[data["question_id"][i]]
        x = (theta[u] - beta[q]).sum()
        p_a = sigmoid(x) * b + a
        pred.append(p_a >= 0.5)
    return np.sum((data["is_correct"] == np.array(pred))) \
           / len(data["is_correct"])


def main():
    train_data = load_train_csv("../data")
    # You may optionally use the sparse matrix.
    # sparse_matrix = load_train_sparse("../data")
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")

    #####################################################################
    # wasTODO:                                                             #
    # Tune learning rate and number of iterations. With the implemented #
    # code, report the validation and test accuracy.                    #
    #####################################################################
    lr = 0.005
    iterations = 40
    theta, beta, a, b, validation_log_likelihood, training_log_likelihood = irt(
        train_data, val_data, lr, iterations)

    plt.title("validation log likelihood")
    plt.xlabel("iteration")
    plt.ylabel("validation log likelihood")
    plt.plot(range(iterations), validation_log_likelihood)
    plt.show()

    plt.title("training log likelihood")
    plt.xlabel("iteration")
    plt.ylabel("training log likelihood")
    plt.plot(range(iterations), training_log_likelihood)
    plt.show()

    acc_test = evaluate(test_data, theta, beta, a, b)
    print("Test Score: {}".format(acc_test))
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################

    #####################################################################
    # wasTODO:                                                             #
    # Implement part (d)                                                #
    #####################################################################
    questions = np.array([150, 300, 450])
    theta = theta.reshape(-1)
    theta.sort()
    for question in questions:
        e = np.exp(theta - beta[question])
        plt.plot(theta, e / (1 + e), label="Question {}".format(question))
    plt.ylabel("Probability")
    plt.xlabel("Theta")
    plt.title("Probablity to theta")
    plt.legend()
    plt.show()

    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    main()
