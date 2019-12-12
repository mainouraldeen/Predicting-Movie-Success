import json
from collections import defaultdict
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
from sklearn import linear_model, metrics
from sklearn.linear_model import Lasso, Ridge, ElasticNet
from sklearn.svm import SVR
# from sklearn.metrics import
# from mlxtend.feature_selection import SequentialFeatureSelector as sfs
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.feature_selection import RFE, RFECV, f_regression, SelectKBest, chi2
import matplotlib.pyplot as plt

# region Public

moviesFile = pd.read_csv('Train Set/tmdb_5000_movies_classification.csv')
creditsFile = pd.read_csv('Train Set/tmdb_5000_credits.csv')

# inner y keep rl 7agat el comman fl 2 df bs
wholeFile = pd.merge(moviesFile, creditsFile, sort=True, how='inner', left_on=['id', 'title'],
                     right_on=['movie_id', 'title'])
# wholeFile = pd.read_csv('out.csv')
Y = wholeFile["vote_average"]


# endregion

def normalizeData(columnName):
    min_element = wholeFile[columnName].min()
    max_element = wholeFile[columnName].max()
    wholeFile[columnName] = 3 * (wholeFile[columnName] - min_element) / (max_element - min_element)


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

                        _dict[value] = (_dict[value][0] + 1, _dict[value][1] + Y[i])

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

        if len(element) == 0:
            sumForAllCells.append(sumForOneCell)
        else:
            sumForAllCells.append(sumForOneCell / len(element))
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
    wholeFile["release_date"] = wholeFile["release_date"].astype('datetime64[ns]')
    # replace el date b-el year bs: first try.
    wholeFile["release_date"] = [i.year for i in wholeFile["release_date"]]

    # region Normalization of scalar data
    normalizeData("budget")
    normalizeData("popularity")
    normalizeData("vote_count")
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

    # region release_date
    convertStringColToScore("release_date")
    normalizeData("release_date")
    # endregion

    # print(wholeFile['original_language'].corr(wholeFile['vote_average']))


def main():
    # print("len of whole file before drop", len(wholeFile))
    wholeFile.drop(labels=['original_title', 'status', 'homepage', 'overview', 'tagline', 'title', 'id', 'movie_id'],
                   axis=1, inplace=True)
    wholeFile.replace(['', ' ', [[]], [], None, {}], np.nan, inplace=True)
    # wholeFile.dropna(inplace=True, how='any')

    median = wholeFile['runtime'].median()
    wholeFile['runtime'].fillna(median, inplace=True)

    # pre-processing:
    dataPreprocessing()

    # after pre-processing:
    X = wholeFile.drop(axis=1, labels="vote_average")
    Y = wholeFile["vote_average"]

    corr = wholeFile.corr()
    # corr.style.background_gradient(cmap='coolwarm', axis=None)

    # Draw correlation matrix:
    plt.matshow(wholeFile.corr())
    plt.show()

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)

    # region SelectKBest
    k = 4
    selector = SelectKBest(f_regression, k=k)
    selector_fit_transform = selector.fit_transform(X_train, Y_train)
    top_features = X_train.columns[selector.get_support()]
    print("top_features  from SelectKBest k =", k)
    print(top_features)
    top_corr = wholeFile[top_features].corr()
    sns.heatmap(top_corr, annot=True)
    plt.show()
    # endregion

    wholeFile.to_csv("finalOutput.csv", index=False)


main()
