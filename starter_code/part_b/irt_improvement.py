from starter_code.utils import *
import starter_code.part_a.item_response

import numpy as np
import matplotlib.pyplot as plt
import csv


def sigmoid(x):
    """ Apply sigmoid function.
    """
    return np.exp(x) / (1 + np.exp(x))


def neg_log_likelihood(data, theta, beta, randomness, slopes):
    """ Compute the negative log-likelihood.

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param theta: Vector
    :param beta: Vector
    :return: float
    """
    log_lklihood = 0.
    for i in range(len(data["user_id"])):
        a = randomness[data["question_id"][i]]
        k = slopes[data["question_id"][i]]
        c = data["is_correct"][i]
        diff = theta[data["user_id"][i]] - beta[data["question_id"][i]]
        frac = np.exp(diff * k)/ (1 + np.exp(diff * k))
        log_lklihood += c * np.log(a + (1-a)*frac) + (1-c) * np.log(1-a- (1-a)*frac)
    return -log_lklihood


def update_theta_beta(data, lr, theta, beta, randomness, slopes):
    """ Update theta and beta using gradient descent.

    You are using alternating gradient descent. Your update should look:
    for i in iterations ...
        theta <- new_theta
        beta <- new_beta

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param theta: Vector
    :param beta: Vector
    :return: tuple of vectors
    """
    theta_copy = theta.copy()
    beta_copy = beta.copy()
    randomness_copy = randomness.copy()
    slopes_copy = slopes.copy()

    for i in range(len(data["user_id"])):
        c = data["is_correct"][i]
        k = slopes_copy[data["question_id"][i]]
        a = randomness_copy[data["question_id"][i]]
        x = theta_copy[data["user_id"][i]]
        y = beta_copy[data["question_id"][i]]
        theta[data["user_id"][i]] += lr * (k * np.exp(x) * ((c - 1) * k * np.exp(x) + (c - a) * np.exp(y))) / (
                    k * np.exp(x) + np.exp(y)) / (k * np.exp(x) + a * np.exp(y))
        beta[data["question_id"][i]] += - lr * (k * np.exp(x) * ((c - a) * np.exp(y) + (c - 1) * k * np.exp(x))) / (
                    k * np.exp(x) + np.exp(y)) / (k * np.exp(x) + a * np.exp(y))
        slopes[data["question_id"][i]] += 0.1 * lr * (np.exp(y) * a - c * np.exp(y) + (1 - c) * k * np.exp(x)) / (
                    a - 1) / (np.exp(y) * a + k * np.exp(x))
        randomness[data["question_id"][i]] += 0.05 * lr * np.exp(x) * (
                    (c - 1) * k * np.exp(x) + (c - a) * np.exp(y)) / (np.exp(x) * k + np.exp(y)) / (
                                                          np.exp(x) * k + a * np.exp(y))
    for i in range(len(data["user_id"])):
        if randomness[data["question_id"][i]] <= 0:
            randomness[data["question_id"][i]] = 0.0001
        if randomness[data["question_id"][i]] >= 1:
            randomness[data["question_id"][i]] = 0.9999
    return theta, beta, randomness, slopes


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
    randomness = np.ones(1774) * 0.0
    slopes = np.ones(1774) * 1.0

    validation_log_likelihood = []
    train_log_likelihood = []

    for i in range(iterations):
        train_neg_lld = neg_log_likelihood(data, theta=theta, beta=beta, randomness=randomness, slopes=slopes)
        train_log_likelihood.append(train_neg_lld)
        validation_neg_lld = neg_log_likelihood(val_data, theta=theta,
                                                beta=beta, randomness=randomness, slopes=slopes)
        validation_log_likelihood.append(validation_neg_lld)
        score_train = evaluate(data=data, theta=theta, beta=beta, randomness=randomness, slopes=slopes)
        score_validation = evaluate(data=val_data, theta=theta, beta=beta, randomness=randomness, slopes=slopes)
        if not quiet:
            print("Iterations: {} \t NLLK: {} \t Train Score: {} \t Validation Score: {}".format(
                i + 1, train_neg_lld, score_train, score_validation))
        theta, beta, randomness, slopes = update_theta_beta(data, lr, theta, beta, randomness, slopes)

    return theta, beta, randomness, slopes, validation_log_likelihood, train_log_likelihood


def evaluate(data, theta, beta, randomness, slopes):
    """ Evaluate the model given data and return the accuracy.
    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}

    :param theta: Vector
    :param beta: Vector
    :param randomness: Vector
    :param slopes: Vector
    :return: float
    """
    pred = []
    for i, q in enumerate(data["question_id"]):
        u = data["user_id"][i]
        k = slopes[data["question_id"][i]]
        a = randomness[data["question_id"][i]]
        x = (theta[u] - beta[q]).sum() * k
        p_a = sigmoid(x) * (1 - a) + a
        pred.append(p_a >= 0.5)
    return np.sum((data["is_correct"] == np.array(pred))) \
           / len(data["is_correct"])


def predict(question_id, user_id, theta, beta, randomness, slopes):
    x = (theta[user_id] - beta[question_id]).sum() * slopes[question_id]
    p_a = sigmoid(x) * (1 - randomness[question_id]) + randomness[question_id]
    return str(int(p_a >= 0.5))


def main():
    train_data = load_train_csv("../data")
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")

    write = False
    lr = 0.015
    iterations = 25
    theta, beta, randomness, slopes, validation_log_likelihood, training_log_likelihood = irt(
        train_data, val_data, lr, iterations)

    if write:
        rows = []
        with open('../data/private_test_data.csv') as f:
            f_csv = csv.reader(f)
            for row in f_csv:
                #print(row)
                rows.append(row)
        rows[0] = ["id", "is_correct"]
        for row_index in range(1, len(rows)):
            rows[row_index][2] = predict(int(rows[row_index][0]), int(rows[row_index][1]), theta, beta, randomness, slopes)
            rows[row_index] = [row_index, rows[row_index][2]]

        print(rows)
        with open('../data/private_test_data_upload.csv', 'w', newline='') as f:
            f_csv = csv.writer(f)
            #f_csv.writerow(headers)
            f_csv.writerows(rows)
            exit(0)

    lr_org = 0.01
    iterations_org = 30
    theta_org, beta_org, validation_log_likelihood_org, training_log_likelihood_org = starter_code.part_a.item_response.irt(
        train_data, val_data, lr_org, iterations_org)

    acc_test = evaluate(test_data, theta, beta, randomness, slopes)
    print("Modified Final Test Score: {}".format(acc_test))

    acc_test = starter_code.part_a.item_response.evaluate(test_data, theta_org, beta_org)
    print("Original Final Test Score: {}".format(acc_test))
    #exit(0)

    questions = np.array([200, 1000, 1300])
    #print(randomness[questions])
    #print(slopes[questions])
    theta = theta.reshape(-1)
    theta.sort()
    for question in questions:
        e = np.exp(theta - beta[question])
        plt.plot(theta, e / (1 + e), label="Question {} Modified".format(question))

    theta_org = theta_org.reshape(-1)
    theta_org.sort()
    for question in questions:
        e = np.exp(theta_org - beta_org[question])
        plt.plot(theta_org, e / (1 + e), label="Question {} Original".format(question))
    plt.ylabel("Probability")
    plt.xlabel("Theta")
    plt.title("Probablity to theta")
    plt.legend()
    plt.show()
    #exit(0)

    plt.title("validation log likelihood")
    plt.xlabel("iteration")
    plt.ylabel("validation log likelihood")
    plt.plot(range(iterations), validation_log_likelihood, label="modified")
    plt.plot(range(iterations_org), validation_log_likelihood_org, label="original")
    plt.legend()
    plt.show()

    plt.title("training log likelihood")
    plt.xlabel("iteration")
    plt.ylabel("training log likelihood")
    plt.plot(range(iterations), training_log_likelihood, label="modified")
    plt.plot(range(iterations_org), training_log_likelihood_org, label="original")
    plt.legend()
    plt.show()



if __name__ == "__main__":
    main()
