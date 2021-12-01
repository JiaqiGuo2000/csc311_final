# wasTODO: complete this file.
from starter_code.utils import *
from starter_code.part_a.matrix_factorization import *
from starter_code.part_a.item_response import *
import numpy as np
from sklearn.impute import KNNImputer


def resample(train_data: dict, size: int):
    """
    resample a given dataset into a smaller data set of the given size
    """
    ret = {"user_id": [], "question_id": [], "is_correct": []}

    for i in range(size):
        index = np.random.choice(len(train_data["question_id"]), 1)[0]
        ret["user_id"].append(train_data["user_id"][index])
        ret["question_id"].append(train_data["question_id"][index])
        ret["is_correct"].append(train_data["is_correct"][index])

    return ret


def evaluate_ensemble(data, thetas, betas):
    """ Evaluate ensemble models given data and return the accuracy.
    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}

    :param theta: list of Vectors
    :param beta: list of Vectors
    :return: float
    """
    pred = []
    for i, q in enumerate(data["question_id"]):
        u = data["user_id"][i]
        p_a = 0
        for theta_index in range(len(thetas)):
            theta = thetas[theta_index]
            beta = betas[theta_index]
            x = (theta[u] - beta[q]).sum()
            # vote!
            p_a += sigmoid(x) >= 0.5
        # check vote results
        pred.append(p_a >= 2)
    return np.sum((data["is_correct"] == np.array(pred))) \
           / len(data["is_correct"])


def sparse_matrix_evaluate_ensemble(data, matrices, threshold=0.5):
    """ Given the sparse matrix represent, return the accuracy of the prediction on data.

    :param data: A dictionary {user_id: list, question_id: list, is_correct: list}
    :param matrices: 2D matrixs
    :param threshold: float
    :return: float
    """
    total_prediction = 0
    total_accurate = 0
    for i in range(len(data["is_correct"])):
        cur_user_id = data["user_id"][i]
        cur_question_id = data["question_id"][i]
        correct_count = 0
        for matrix in matrices:
            if matrix[cur_user_id, cur_question_id] >= threshold and data["is_correct"][i]:
                correct_count += 1
            if matrix[cur_user_id, cur_question_id] < threshold and not data["is_correct"][i]:
                correct_count += 1
        if correct_count >= len(matrices)/2:
            total_accurate += 1
        total_prediction += 1
    return total_accurate / float(total_prediction)


def knn_ensemble():
    train_matrix = load_train_sparse("../data").toarray()
    train_data = load_train_csv("../data")
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")

    mats = []
    for i in range(3):
        resampled_data = resample(train_data, int(len(train_data["user_id"])/1.5))
        resampled_matrix = train_matrix.copy()
        for index in range(len(resampled_data["user_id"])):
            resampled_matrix[resampled_data["user_id"][index]][resampled_data["question_id"][index]] = resampled_matrix[0][0]

        nbrs = KNNImputer(n_neighbors=11)
        mat = nbrs.fit_transform(resampled_matrix)
        mats.append(mat)
    train_acc = sparse_matrix_evaluate_ensemble(train_data, mats)
    val_acc = sparse_matrix_evaluate_ensemble(val_data, mats)
    test_acc = sparse_matrix_evaluate_ensemble(test_data, mats)
    print("Ensemble Train Score: {}\t Validation Score: {} \t Test Score: {}".format(train_acc, val_acc, test_acc))

    nbrs = KNNImputer(n_neighbors=11)
    mat = nbrs.fit_transform(train_matrix)
    train_acc = sparse_matrix_evaluate(train_data, train_matrix)
    val_acc = sparse_matrix_evaluate(val_data, mat)
    test_acc = sparse_matrix_evaluate(test_data, mat)
    print("Original Train Score: {}\t Validation Score: {} \t Test Score: {}".format(train_acc, val_acc, test_acc))


def pca_ensemble():
    train_matrix = load_train_sparse("../data").toarray()
    train_data = load_train_csv("../data")
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")

    mats = []
    for i in range(3):
        resampled_data = resample(train_data, int(len(train_data["user_id"])/1.5))
        resampled_matrix = train_matrix.copy()
        for index in range(len(resampled_data["user_id"])):
            resampled_matrix[resampled_data["user_id"][index]][resampled_data["question_id"][index]] = resampled_matrix[0][0]

        mat = svd_reconstruct(resampled_matrix, 9)
        mats.append(mat)
    train_acc = sparse_matrix_evaluate_ensemble(train_data, mats)
    val_acc = sparse_matrix_evaluate_ensemble(val_data, mats)
    test_acc = sparse_matrix_evaluate_ensemble(test_data, mats)
    print("Ensemble Train Score: {}\t Validation Score: {} \t Test Score: {}".format(train_acc, val_acc, test_acc))

    reconst_matrix = svd_reconstruct(train_matrix, 9)
    train_acc = sparse_matrix_evaluate(train_data, reconst_matrix)
    val_acc = sparse_matrix_evaluate(val_data, reconst_matrix)
    test_acc = sparse_matrix_evaluate(test_data, reconst_matrix)
    print("Original Train Score: {}\t Validation Score: {} \t Test Score: {}".format(train_acc, val_acc, test_acc))


def irt_ensemble():
    # load data
    train_matrix = load_train_sparse("../data").toarray()
    train_data = load_train_csv("../data")
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")

    # resample & run irt 3 times
    lr = 0.01
    iterations = 30
    thetas = []
    betas = []
    for i in range(3):
        resampled_data = resample(train_data, int(len(train_data["user_id"])/3))
        theta, beta, validation_log_likelihood, training_log_likelihood = irt(resampled_data, val_data, lr, iterations, True)
        thetas.append(theta)
        betas.append(beta)

    train_acc = evaluate_ensemble(train_data, thetas, betas)
    val_acc = evaluate_ensemble(val_data, thetas, betas)
    test_acc = evaluate_ensemble(test_data, thetas, betas)
    print("Ensemble Train Score: {}\t Validation Score: {} \t Test Score: {}".format(train_acc, val_acc, test_acc))

    # compare the score with the non-bagging accuracy
    theta, beta, validation_log_likelihood, training_log_likelihood = irt(train_data, val_data, lr, iterations, True)
    train_acc = evaluate(train_data, theta, beta)
    val_acc = evaluate(val_data, theta, beta)
    test_acc = evaluate(test_data, theta, beta)
    print("Original Train Score: {}\t Validation Score: {} \t Test Score: {}".format(train_acc, val_acc, test_acc))


def main():
    #print("knn")
    #knn_ensemble()
    print("irt")
    irt_ensemble()
    #print("pca")
    #pca_ensemble()


if __name__ == "__main__":
    main()