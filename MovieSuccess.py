import json
from collections import defaultdict
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from sklearn import linear_model, metrics, feature_selection
from sklearn.feature_selection import SelectKBest, chi2, f_regression
from sklearn.preprocessing import PolynomialFeatures


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

def convertStringColToScore(colName):
    i = 0
    counter = 0
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
    #region drop cols

    #endregion
    wholeFile["release_date"] = wholeFile["release_date"].astype('datetime64[ns]')
    # print(type(wholeFile["release_date"][0]))
    # print((wholeFile["release_date"][0].year))

    # replace el date b-el year bs: first try.
    wholeFile["release_date"] = [ i.year for i in wholeFile["release_date"]]
    # print(wholeFile["release_date"])

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
    # print(wholeFile["release_date"].describe())


    # print(wholeFile['original_language'].corr(wholeFile['vote_average']))


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
    print("len of whole file before drop",len(wholeFile))
    wholeFile.drop(labels=['original_title', 'status', 'homepage','overview','tagline', 'title', 'id', 'movie_id'], axis=1, inplace=True)
    wholeFile.replace(['', ' ', [[]], [], None, {}], np.nan, inplace=True)  # 3shan .drop b t-drop no.nan bs
    # wholeFile.dropna(inplace=True,how='any')
    median = wholeFile['runtime'].median()
    wholeFile['runtime'].fillna(median, inplace=True)
    # print(wholeFile.isnull().sum())
    print("len of whole file after drop",len(wholeFile))

    dataPreprocessing()

    X = wholeFile.drop(axis=1, labels="vote_average")
    Y = wholeFile["vote_average"]


    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
    X_train, X_validation, Y_train, Y_validation = train_test_split(X_train, Y_train, test_size=0.2)

    # print(wholeFile.columns.values)


    # select top features
    # top_features = wholeFile.corr().index[abs(wholeFile.corr()['vote_average'] >= 0.3)]
    # print(top_features)
    # print(wholeFile["budget"].corr(wholeFile['vote_average']))

    #best features
    # top_featurs = SelectKBest(chi2,k=5).fit(X_train, Y_train)
    # print("Top feAtures", top_featurs)

    # Technique 1
    # Multiple linear regression with all features
    # regression = linear_model.LinearRegression()
    # regression.fit(X_train, Y_train)
    # predictions = regression.predict(X_validation)
    # print("*mean_absolute_error:", metrics.mean_absolute_error(Y_validation, predictions))
    # print("*mean_squared_error:", metrics.mean_squared_error(Y_validation, predictions))
    # print("*root mean_squared_error:", np.sqrt(metrics.mean_squared_error(Y_validation, predictions)))
    # print("*regression.score", regression.score(X_validation, Y_validation))
    # # print("accu:", metrics.accuracy_score(Y_validation, predictions))

    # Technique 2
    # poly = PolynomialFeatures(2)
    #
    # poly_fit = poly.fit_transform(X_train)
    # poly.fit(poly_fit, Y_train)
    # print("poly")
    # print(metrics.mean_squared_error(Y_validation, predictions))
    # print("poly_fit")
    # print(poly_fit)



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
