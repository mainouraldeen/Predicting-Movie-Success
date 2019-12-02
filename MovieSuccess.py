import json
from collections import defaultdict
import numpy as np
import pandas as pd
from sklearn import linear_model, metrics
from sklearn.linear_model import Lasso, Ridge, ElasticNet
from sklearn.svm import SVR
# from sklearn.metrics import
from mlxtend.feature_selection import SequentialFeatureSelector as sfs
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.feature_selection import RFE, RFECV, f_regression, SelectKBest, chi2
import matplotlib.pyplot as plt

# region Public

moviesFile = pd.read_csv('Train Set/tmdb_5000_movies_train.csv', keep_default_na=True)#, na_values=[""])
creditsFile = pd.read_csv('Train Set/tmdb_5000_credits_train.csv', keep_default_na=True)#, na_values=[""])

# print("From public:")
# w = pd.get_dummies(moviesFile, columns=['title', 'tagline', 'homepage'], drop_first=True)
# print(moviesFile["title"][0])

# inner y keep rl 7agat el comman fl 2 df bs
wholeFile = pd.merge(moviesFile, creditsFile, sort=True, how='inner', left_on=['id', 'title'],
                    right_on=['movie_id', 'title'])
# wholeFile = pd.read_csv('out.csv')
Y = wholeFile["vote_average"]

# endregion

def normalizeData(columnName):
    min_element = wholeFile[columnName].min()
    max_element = wholeFile[columnName].max()
    wholeFile[columnName] = 0.9 * (wholeFile[columnName] - min_element) / (max_element - min_element)


def convertDictColumnToScore(columnName, uniqueKey):
    i = 0
    counter = 0
    _dict = defaultdict(tuple)
    _list = []
    for cell in wholeFile[columnName]:  # cell: group of dictionaries
        oneCell = json.loads(cell)
        _list.append(oneCell)
        if oneCell != {}:
            counter += 1
            for list_element in oneCell:
                for key, value in list_element.items():
                    if key == uniqueKey:
                        # each dictionary element has: #occurrencess, its score(each occurrence: add Y[i])
                        if not _dict[value]:
                            _dict[value] = (0, 0)

                        _dict[value] = (
                            _dict[value][0] + 1, _dict[value][1] + Y[i])
        else:
            print("at i:", i, " EMPTY CELL!!!!!!")
            break
        i += 1

    score_dict = defaultdict(float)
    for key, (num_occurr, score) in _dict.items():
        score_dict[key] = score / num_occurr

    sumForOneCell = 0
    sumForAllCells = []
    for element in _list:
        for list_element in element:
            for key, value in list_element.items():
                if key == uniqueKey:
                    sumForOneCell += score_dict[value]
        sumForAllCells.append(sumForOneCell)
        sumForOneCell = 0
    wholeFile[columnName] = sumForAllCells


def convertStringColToScore(colName):
    i = 0
    _dict = defaultdict(tuple)
    _list = []
    for cell in wholeFile[colName]:
        if not _dict[cell]:
            _dict[cell] = (0, 0)

        _dict[cell] = (_dict[cell][0] + 1, _dict[cell][1] + Y[i])
        i += 1

    lst = []
    for language in wholeFile[colName]:
        lst.append(_dict[language][1] / _dict[language][0])

    wholeFile[colName] = lst


def dataPreprocessing():
    # region drop cols

    # endregion
    wholeFile["release_date"] = wholeFile["release_date"].astype('datetime64[ns]')
    # replace el date b-el year bs: first try.
    wholeFile["release_date"] = [i.year for i in wholeFile["release_date"]]

    # region Normalization of scalar data
    normalizeData("budget")
    normalizeData("popularity")
    normalizeData("vote_count")
    normalizeData("vote_average")
    normalizeData("revenue")
    normalizeData("runtime")

    # endregion

    # region cast pre-processing
    convertDictColumnToScore("cast", "id")
    normalizeData("cast")
    # print("cast corr",wholeFile["cast"].corr(wholeFile['vote_average']))

    # endregion

    # region crew pre-processing

    convertDictColumnToScore("crew", "id")
    # normalize crew col
    normalizeData("crew")

    # endregion

    # region keywords Pre-processing
    # converting keywords into dictionaries
    convertDictColumnToScore("keywords", "name")
    normalizeData("keywords")
    # print("keywords corr",wholeFile["keywords"].corr(wholeFile['vote_average']))

    # endregion

    # region spoken_languages pre-processing
    convertDictColumnToScore("spoken_languages", "name")
    normalizeData("spoken_languages")
    # print("spoken languages corr",wholeFile["spoken_languages"].corr(wholeFile['vote_average']))
    # endregion

    # region genres pre-processing
    convertDictColumnToScore("genres", "id")
    normalizeData("genres")
    # print("genres corr",wholeFile["genres"].corr(wholeFile['vote_average']))
    # endregion

    # region production_companies pre-processing
    convertDictColumnToScore("production_companies", "id")
    normalizeData("production_companies")
    # endregion

    # region production_countries pre-processing
    convertDictColumnToScore("production_countries", "iso_3166_1")
    normalizeData("production_countries")
    # endregion

    # region language
    convertStringColToScore("original_language")
    normalizeData("original_language")
    # endregion

    # print(wholeFile["release_date"].describe())
    # print("----------------------")

    # region release_date
    convertStringColToScore("release_date")
    normalizeData("release_date")
    # endregion

    # print(wholeFile['original_language'].corr(wholeFile['vote_average']))


def main():

    # print("len of whole file before drop", len(wholeFile))
    wholeFile.drop(labels=['original_title', 'status', 'homepage', 'overview', 'tagline', 'title', 'id', 'movie_id'],
                   axis=1, inplace=True)
    wholeFile.replace(['', ' ', [[]], [], None, {}], np.nan, inplace=True)  # 3shan .drop b t-drop no.nan bs
    # wholeFile.dropna(inplace=True, how='any')

    median = wholeFile['runtime'].median()
    # print("NULL before", wholeFile.isnull().sum())
    wholeFile['runtime'].fillna(median, inplace=True)
    # print("NULL after", wholeFile.isnull().sum())
    # print("len of whole file after drop", len(wholeFile))

    # pre-processing:
    dataPreprocessing()

    # after pre-processing:
    X = wholeFile.drop(axis=1, labels="vote_average")
    Y = wholeFile["vote_average"]

    # corr = wholeFile.corr()
    # corr.style.background_gradient(cmap='coolwarm', axis=None)

    # Draw correlation matrix:
    # plt.matshow(wholeFile.corr())
    # plt.show()

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)
    # X_train, X_validation, Y_train, Y_validation = train_test_split(X_train, Y_train, test_size=0.2, shuffle=True)

    # region Select top features
    top_features = wholeFile.corr().index[abs(wholeFile.corr()['vote_average'] >= 0.3)]
    top_features = top_features.drop("vote_average")
    # print("top_features using corr()")
    # print(top_features)
    # print(wholeFile["budget"].corr(wholeFile['vote_average']))
    #
    # # best features
    # top_features = SelectKBest(chi2, k=4).fit(X_train, Y_train)
    # print("Top features using chi2")
    # print(top_features)

    #endregion

    # region Multiple linear regression with all features
    # Technique 1
    print("Multiple linear:")
    regression = linear_model.LinearRegression()
    regression.fit(X_train, Y_train)
    # regression.fit(X_train[top_features], Y_train)
    predictions = regression.predict(X_test)
    # predictions = regression.predict(X_test[top_features])
    # print("*mean_absolute_error:", metrics.mean_absolute_error(Y_test, predictions))
    print("Accuracy:", 100 * metrics.r2_score(Y_test, predictions))
    print("MSE:", metrics.mean_squared_error(Y_test, predictions))
    # print("*root mean_squared_error:", np.sqrt(metrics.mean_squared_error(Y_test, predictions)))
    # print("*regression.score", regression.score(X_test, Y_test))

    # endregion
    print("---------------------------------")

    # region Polynomial regression
    # Technique 2
    print("Polynomial")
    poly = PolynomialFeatures(3)
    poly_fit = poly.fit_transform(X_train)
    poly.fit(poly_fit, Y_train)
    # print("Accuracy:", 100 * metrics.r2_score(Y_test, predictions))
    print("MSE:", metrics.mean_squared_error(Y_test, predictions))

    # endregion
    print("---------------------------------")

    # region SVR regression

    print("SVR")
    svr_rbf = SVR(kernel='rbf', C=1, gamma=1, epsilon=.1).fit(X_train[top_features], Y_train)
    # svr_lin = SVR(kernel='linear', C=1).fit(X_train[top_features], Y_train)
    # print("after second")
    # svr_poly = SVR(kernel='poly', C=1, gamma=1, degree=2, epsilon=.1, coef0=1).fit(X_train[top_features], Y_train)
    # print("after third")
    # print("linear coeff", svr_lin.coef_)
    # print("bias", svr_lin.intercept_)

    predictions = svr_rbf.predict(X_test[top_features])
    print("Accuracy:", metrics.r2_score(Y_test, predictions)*100)#np.mean(predictions == Y_validation)*100)
    print("MSE:", metrics.mean_squared_error(Y_test, predictions))

    # predictions = svr_lin.predict(X_test[top_features])
    # print("**svr_lin accuracy:", metrics.r2_score(Y_test, predictions)*100)#np.mean(predictions == Y_validation)*100)
    # print("svr_lin MSE:", metrics.mean_squared_error(Y_test, predictions), "\n")
    #
    # predictions = svr_poly.predict(X_test[top_features])
    # print("**svr_poly accuracy:", metrics.r2_score(Y_test, predictions)*100)#np.mean(predictions == Y_validation)*100)
    # print("svr_poly MSE:", metrics.mean_squared_error(Y_test, predictions))

    # endregion
    print("---------------------------------")


    # region Lasso Regression

    print("Lasso")

    lasso = Lasso()
    lasso.fit(X_train, Y_train)
    # lasso.fit(X_train[top_features], Y_train)

    # # test_score = lasso.score(X_test, Y_test)
    # # test_score = lasso.score(X_test[top_features], Y_test)

    predictions = lasso.predict(X_test)
    # predictions = lasso.predict(X_test[top_features])

    test_score = metrics.r2_score(Y_test, predictions)
    print("Accuracy:", test_score*100)
    print("MSE:", metrics.mean_squared_error(Y_test, predictions))

    # endregion
    print("---------------------------------")

    #region Ridge Regression
    print("Ridge")
    ridge_regression = Ridge(alpha=.5)
    # ridge_regression.fit(X_train, Y_train)
    ridge_regression.fit(X_train[top_features], Y_train)
    # test_score = ridge_regression.score(X_test, Y_test)
    # test_score = ridge_regression.score(X_test[top_features], Y_test)
    # print("Ridge regression score", test_score*100)
    # predictions = ridge_regression.predict(X_test)
    predictions = ridge_regression.predict(X_test[top_features])
    print("Accuracy:", metrics.r2_score(Y_test, predictions)*100)
    print("MSE:", metrics.mean_squared_error(Y_test, predictions))

    # endregion

    print("---------------------------------")
    # region Elastic Regression
    print("Elastic")
    elastic_regression = ElasticNet(random_state=0)
    elastic_regression.fit(X_train, Y_train)
    # elastic_regression.fit(X_train[top_features], Y_train)
    predictions = elastic_regression.predict(X_test)
    # predictions = elastic_regression.predict(X_test[top_features])
    accuracy = metrics.r2_score(Y_test, predictions)*100
    print("Accuracy:", accuracy)
    test_error = metrics.mean_squared_error(Y_test, predictions)
    print("MSE:", test_error)

    # endregion

    # c = pd.DataFrame(regression.coef_, X.columns, columns=["CCC"])
    # print("C")
    # print(c)
    wholeFile.to_csv("finalOutput.csv", index=False)


main()
