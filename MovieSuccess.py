import json
from collections import defaultdict

import numpy as np
import pandas as pd

# public
# region

moviesFile = pd.read_csv('Train Set/tmdb_5000_movies_train.csv', keep_default_na=False, na_values=[""])
creditsFile = pd.read_csv('Train Set/tmdb_5000_credits_train.csv', keep_default_na=False, na_values=[""])

# inner y keep rl 7agat el comman fl 2 df bs
wholeFile = pd.merge(moviesFile, creditsFile, sort=True, how='inner', left_on=['id', 'title'],
                     right_on=['movie_id', 'title'])
Y = wholeFile["vote_average"]


# endregion

def normalizeData(columnName):
    min_element = wholeFile[columnName].min()
    max_element = wholeFile[columnName].max()
    wholeFile[columnName] = (wholeFile[columnName] - min_element) / (max_element - min_element)

def convertDictColumnToScore(columnName,uniqueKey):
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


def dataPreprocessing():

    wholeFile["release_date"] = wholeFile["release_date"].astype('datetime64[ns]')

    # region Normalization of scaler data
    normalizeData("budget")
    normalizeData("popularity")
    normalizeData("vote_count")
    normalizeData("vote_average")
    normalizeData("revenue")
    normalizeData("runtime")
    # endregion

    # region cast pre-proccessing
    convertDictColumnToScore("cast", "id")
    # normalize cast col
    normalizeData("cast")
    # print("cast corr",wholeFile["cast"].corr(wholeFile['vote_average']))
    # ana 3ayza el key yb2a id el film w el value list of scores el momsleen gaya mn list of tuples kol wa7d etkrr kam mra w score el aflam ele etkkrr feha
    # endregion

    # region crew pre-proccessing
    convertDictColumnToScore("crew","id")
    # normalize crew col
    normalizeData("crew")
    # print("crew corr",wholeFile["crew"].corr(wholeFile['vote_average']))
    # ana 3ayza el key yb2a id el film w el value list of scores el momsleen gaya mn list of tuples kol wa7d etkrr kam mra w score el aflam ele etkkrr feha
    # endregion

    # region keywords Pre-processing
    # converting keywords into dictionaries
    convertDictColumnToScore("keywords","name")
    normalizeData("keywords")
    #print("keywords corr",wholeFile["keywords"].corr(wholeFile['vote_average']))
    #print(keywords_score)
    # endregion

    # region spoken_languagues pre-processing
    convertDictColumnToScore("spoken_languages","name")
    normalizeData("spoken_languages")
    #print("spoken languages corr",wholeFile["spoken_languages"].corr(wholeFile['vote_average']))
    # endregion

    # region genres pre-processing
    convertDictColumnToScore("genres", "id")
    normalizeData("genres")
    # print("genres corr",wholeFile["genres"].corr(wholeFile['vote_average']))
    # endregion

    # region production_companies pre-processing
    convertDictColumnToScore("production_companies", "id")
    normalizeData("production_companies")
    # print("sproduction_companies corr",wholeFile["production_companies"].corr(wholeFile['vote_average']))
    # endregion

    # region production_countries pre-processing
    convertDictColumnToScore("production_countries", "id")
    normalizeData("production_countries")
    # print("production_countries corr",wholeFile["production_countries"].corr(wholeFile['vote_average']))
    # endregion


def main():
    # region
    """
    The sign of the covariance can be interpreted as whether the two variables change
    in the same direction (positive)or change in different directions (negative).
    The magnitude of the covariance is not easily interpreted.
    A covariance value of zero indicates that both variables are completely independent.

    # print(np.cov(x, y))
    """
    # endregion

    wholeFile.replace(['', ' ', [[]], [], None, {}], np.nan, inplace=True)  # 3shan .drop b t-drop no.nan bs
    wholeFile.dropna(thresh=1, inplace=True)

    dataPreprocessing()

    # X = wholeFile.drop(axis=1, labels="vote_average")
    # Y = wholeFile["vote_average"] #at public region
    # print(type(Y[0]))
    #
    # X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
    # X_train, X_validation, Y_train, Y_validation = train_test_split(X_train, Y_train, test_size=0.2)

    # print(wholeFile["tagline"].describe())

    # select top features
    # top_features = wholeFile.corr().index[abs(wholeFile.corr()['vote_average'] >= 0.3)]
    # # print(top_features)
    # # print(wholeFile["budget"].corr(wholeFile['vote_average']))
    #
    # X = wholeFile.drop(axis=1, labels="vote_average")
    # Y = wholeFile["vote_average"]
    #
    # X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
    # X_train, X_validation, Y_train, Y_validation = train_test_split(X_train, Y_train, test_size=0.2)

    # Multiple linear regression
    # regression = LinearRegression()
    # regression.fit(X_train, Y_train)
    # predictions = regression.predict(X_validation)
    # print("MSE:", metrics.mean_absolute_error(Y_validation, predictions))
    # print("MSE:", metrics.mean_squared_error(Y_validation, predictions))
    #
    # c = pd.DataFrame(regression.coef_, X.columns, columns=["CCC"])
    # print("C")
    # print(c)
    wholeFile.to_csv("finalOutput.csv", index=False)


main()

# CAST
# {"cast_id": 45, XX
# "character": "Sam the Bellhop", XX
# "credit_id": "52fe420dc3a36847f80001c3", XX
# "gender": 2, XX
# "id": 3140, (***)
# "name": "Marc Lawrence", (***)
# "order": 23} XX
##

# CREW
# {"credit_id": "5770143fc3a3683733000f3a",
# "department": "Writing",
# "gender": 2,
# "id": 7,
# "job": "Story",
# "name": "Andrew Stanton"}
