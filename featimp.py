import pandas as pd
import numpy as np
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error


def plot_imp(df, title=None, sort=False):
    """
    Simple bar plot for feature importance visualization.
    :param df:
    :return:
    """
    if sort==True:
        df.sort_values(by="score", ascending=False, inplace=True)
    fig, ax = plt.subplots()
    sns.barplot(x="score", y="feature", data=df, palette="Blues_d")
    ax.set_title(title)
    ax.set_xlabel("")
    ax.set_ylabel("")
    plt.show()

def rank_corr(X, y):
    """
    Implementation of Spearman's correlation coefficient ranking.
    :param X:
    :param y:
    :return:
    """
    features = X.columns
    coef = []
    for f in features:
        coef.append(np.abs(spearmanr(y, X[f])[0]))
    spearmancc = pd.DataFrame({"feature":features, "score":coef})
    return spearmancc

def mRMR(X, y):
    """
    Implementation of mRMR algorithm.
    :param X:
    :param y:
    :return:
    """
    features = X.columns
    F = [_ for _ in range(len(features))]  # index of features
    coef_mat = np.abs(spearmanr(y, X)[0])  # coefficient matrix
    Jm = [max(coef_mat[1:, 0])]  # initialize the J value
    j = np.argmax(coef_mat[1:, 0])  # the feature index has the max J
    S = [j]  # S feature set
    F.remove(j)
    while len(F) > 0:  # find the next feature
        J = coef_mat[1:, 0][F] - coef_mat[1:, 1:][F][:, S].sum(axis=1) / len(S)
        Jm.append(max(J))  # repeatedly update J, S and F
        j = F[np.argmax(J)]
        S.append(j)
        F.remove(j)
    mrmr = pd.DataFrame({"feature":features[S], "score":Jm})
    return mrmr

def permutation_importances(rf, X_test, y_test):
    baseline = rf.score(X_test, y_test)
    features = X_test.columns
    imp = []
    for col in features:
        X_copy = X_test.copy()
        X_copy[col] = np.random.permutation(X_copy[col])
        score = rf.score(X_copy, y_test)
        imp.append(baseline - score)
    permute_imp = pd.DataFrame({"feature":features, "score":imp})
    return permute_imp

def dropcol_importances(rf, X, y):
    baseline = rf.oob_score_
    features = X.columns
    imp = []
    for col in features:
        X_drop = X.drop(columns=col)
        rf_ = RandomForestRegressor(random_state=24, oob_score=True)
        rf_.fit(X_drop, y)
        score = rf_.oob_score_
        imp.append(baseline - score)
    dropcol_imp = pd.DataFrame({"feature":features, "score":imp})
    return dropcol_imp

def top_features(X, y, algo):
    features = X.columns
    feat_imp = algo(X, y).sort_values(by="score", ascending=False)
    top_feat = feat_imp["feature"].values
    lr_score = []
    rf_score = []
    gbm_score = []
    for k in range(1, len(features)+1):
        top_k = top_feat[:k]
        X_k = X[top_k]
        lr = LinearRegression().fit(X_k, y)
        rf = RandomForestRegressor().fit(X_k, y)
        gbm = GradientBoostingRegressor().fit(X_k, y)
        lr_score.append(mean_squared_error(y, lr.predict(X_k)))
        rf_score.append(mean_squared_error(y, rf.predict(X_k)))
        gbm_score.append(mean_squared_error(y, gbm.predict(X_k)))
    top_k_score = pd.DataFrame({"top_k":range(1, len(features)+1),
                                "lr_score":lr_score,
                                "rf_score":rf_score,
                                "gbm_score":gbm_score})
    return top_k_score

def auto_select(rf, X, y):
    feat_imp = dropcol_importances(rf, X, y).sort_values(by="score", ascending=False)
    features = feat_imp.feature.values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    baseline = mean_squared_error(y_test, y_pred)
    score = baseline
    while score <= baseline:
        drop = features[-1]
        print("Dropped", drop)
        features = features[:-1]
        X_train, X_test, y_train, y_test = train_test_split(X[features], y, test_size=0.25)
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_test)
        score = mean_squared_error(y_test, y_pred)
        print("score", f"{score:.4f}", "Baseline", f"{baseline:.4f}")
        feat_imp = dropcol_importances(rf, X[features], y).sort_values(by="score", ascending=False)
        features = feat_imp.feature.values
    features = np.append(features, drop)
    feat_imp = dropcol_importances(rf, X[features], y)
    return features

def var_imp(X, y, algo, k=10):
    features = X.columns
    df = pd.concat([y, X], axis=1)
    scores = [[] for _ in range(X.shape[1])]
    for i in range(k):
        bootstrap = df.sample(frac=1, replace=True)
        X = bootstrap.iloc[:, 1:]
        y = bootstrap["mpg"]
#         rf = RandomForestRegressor(random_state=24, oob_score=True)
#         rf.fit(X, y)
#         feat_imp = algo(rf, X, y)
        feat_imp = algo(X, y)
        for _ in range(X.shape[1]):
            scores[_].append(feat_imp.loc[_, "score"])
    var_scores = pd.DataFrame({"feature":features,
                               "score":np.array(scores).mean(axis=1),
                               "std":np.array(scores).std(axis=1)})
#     var_scores.sort_values(by="mean", ascending=False, inplace=True)
    return var_scores
