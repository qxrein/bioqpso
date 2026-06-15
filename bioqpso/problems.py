import numpy as np
import opfunu
from sklearn.datasets import fetch_openml, load_breast_cancer, load_wine
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder


def sphere(x):
    return np.sum(x ** 2)


def rastrigin(x):
    D = len(x)
    return 10 * D + np.sum(x ** 2 - 10 * np.cos(2 * np.pi * x))


def ackley(x):
    D = len(x)
    sum1 = np.sum(x ** 2)
    sum2 = np.sum(np.cos(2 * np.pi * x))
    term1 = -20 * np.exp(-0.2 * np.sqrt(sum1 / D))
    term2 = -np.exp(sum2 / D)
    return term1 + term2 + 20 + np.e


def griewank(x):
    sum_term = np.sum(x ** 2 / 4000)
    prod_term = np.prod(np.cos(x / np.sqrt(np.arange(1, len(x) + 1))))
    return sum_term - prod_term + 1


def rosenbrock(x):
    x = np.asarray(x)
    return np.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (x[:-1] - 1.0) ** 2)


def schwefel(x):
    x = np.asarray(x)
    D = len(x)
    return 418.9829 * D - np.sum(x * np.sin(np.sqrt(np.abs(x))))


def levy(x):
    x = np.asarray(x)
    w = 1 + (x - 1) / 4.0
    term1 = np.sin(np.pi * w[0]) ** 2
    term3 = (w[-1] - 1) ** 2 * (1 + np.sin(2 * np.pi * w[-1]) ** 2)
    wi = w[:-1]
    term2 = np.sum((wi - 1) ** 2 * (1 + 10 * np.sin(np.pi * wi + np.pi) ** 2))
    return term1 + term2 + term3


def zakharov(x):
    x = np.asarray(x)
    i = np.arange(1, len(x) + 1)
    sum1 = np.sum(x ** 2)
    sum2 = np.sum(0.5 * i * x)
    return sum1 + sum2 ** 2 + sum2 ** 4


def styblinski_tang(x):
    x = np.asarray(x)
    return 0.5 * np.sum(x ** 4 - 16 * x ** 2 + 5 * x)


def dixon_price(x):
    x = np.asarray(x)
    i = np.arange(2, len(x) + 1)
    term1 = (x[0] - 1) ** 2
    term2 = np.sum(i * (2 * x[1:] ** 2 - x[:-1]) ** 2)
    return term1 + term2


PROBLEMS = {
    "sphere": {"func": sphere, "min_bound": -100, "max_bound": 100, "D": 30},
    "rastrigin": {"func": rastrigin, "min_bound": -5.12, "max_bound": 5.12, "D": 30},
    "ackley": {
        "func": ackley,
        "min_bound": -32.768,
        "max_bound": 32.768,
        "D": 30,
    },
    "griewank": {"func": griewank, "min_bound": -600, "max_bound": 600, "D": 30},
    "rosenbrock": {
        "func": rosenbrock,
        "min_bound": -2.048,
        "max_bound": 2.048,
        "D": 30,
    },
    "schwefel": {"func": schwefel, "min_bound": -500, "max_bound": 500, "D": 30},
    "levy": {"func": levy, "min_bound": -10, "max_bound": 10, "D": 30},
    "zakharov": {"func": zakharov, "min_bound": -5, "max_bound": 10, "D": 30},
    "styblinski_tang": {
        "func": styblinski_tang,
        "min_bound": -5,
        "max_bound": 5,
        "D": 30,
    },
    "dixon_price": {"func": dixon_price, "min_bound": -10, "max_bound": 10, "D": 30},
}


CEC_PROBLEMS = {}
for i in range(1, 11):
    try:
        func_obj = opfunu.cec_based.CEC2017(func_num=i, ndim=30)
    except AttributeError:
        from opfunu.cec_based import cec2017

        func_obj = getattr(cec2017, f"F{i}2017")(ndim=30)

    CEC_PROBLEMS[f"CEC2017_F{i}"] = {
        "name": f"CEC2017_F{i}",
        "func": func_obj.evaluate,
        "min_bound": -100,
        "max_bound": 100,
        "D": 30,
    }


def make_feature_selection_problem_from_arrays(X, y):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)
    n_features = X.shape[1]

    def fitness(x):
        x = np.asarray(x)
        mask = x >= 0.5
        selected = np.sum(mask)
        if selected == 0:
            return 1.0

        X_sel = X[:, mask]
        clf = KNeighborsClassifier(n_neighbors=5)
        scores = cross_val_score(clf, X_sel, y, cv=5)
        mean_acc = np.mean(scores)

        feature_frac = selected / n_features
        return 0.7 * (1.0 - mean_acc) + 0.3 * feature_frac

    return {
        "func": fitness,
        "min_bound": 0.0,
        "max_bound": 1.0,
        "D": n_features,
        "X": X,
        "y": y,
    }


def make_feature_selection_problem():
    data = load_breast_cancer()
    return make_feature_selection_problem_from_arrays(data.data, data.target)


def make_wine_feature_selection_problem():
    data = load_wine()
    return make_feature_selection_problem_from_arrays(data.data, data.target)


def make_ionosphere_feature_selection_problem():
    data = fetch_openml(name="ionosphere", version=1, as_frame=False, parser="liac-arff")
    X = np.asarray(data.data, dtype=float)
    y = LabelEncoder().fit_transform(data.target)
    return make_feature_selection_problem_from_arrays(X, y)

