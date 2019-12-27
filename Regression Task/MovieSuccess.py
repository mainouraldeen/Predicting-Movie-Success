import json
from collections import defaultdict
import numpy as np
import pandas as pd
import seaborn as sns
import re
from nltk.stem.porter import PorterStemmer  # 3shan ttla3 el esm lwa7do, zay fish mn fisher aw fishing
from nltk.corpus import stopwords  # 3shan nsheel el stopwords
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib
from sklearn import linear_model, metrics
from sklearn.linear_model import Lasso, Ridge, ElasticNet
from sklearn.svm import SVR
# from sklearn.metrics import
# from mlxtend.feature_selection import SequentialFeatureSelector as sfs
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, scale
from sklearn.feature_selection import RFE, RFECV, f_regression, SelectKBest, chi2
import matplotlib.pyplot as plt

# region Public

moviesFile = pd.read_csv('Train Set/tmdb_5000_movies_train.csv', keep_default_na=True)  # , na_values=[""])
creditsFile = pd.read_csv('Train Set/tmdb_5000_credits_train.csv', keep_default_na=True)  # , na_values=[""])
testfilemovies=pd.read_excel('/home/nourhan/Downloads/Predicting-Movie-Success-master(1)/Predicting-Movie-Success-master/tmdb_5000_movies_testing_regression.xlsx')
testfilecredits=pd.read_csv('/home/nourhan/Downloads/Predicting-Movie-Success-master(1)/Predicting-Movie-Success-master/Classification Task/Movies success/tmdb_5000_credits_test.csv')



# print("From public:")
# w = pd.get_dummies(moviesFile, columns=['title', 'tagline', 'homepage'], drop_first=True)
# print(moviesFile["title"][0])

# inner y keep rl 7agat el comman fl 2 df bs
wholeFile = pd.merge(moviesFile, creditsFile, sort=True, how='inner', left_on=['id', 'title'],
                     right_on=['movie_id', 'title'])
# wholeFile = pd.read_csv('out.csv')
Y = wholeFile["vote_average"]

wholeTestFile=pd.merge(testfilemovies,testfilecredits,sort=True, how='inner', left_on=['id', 'title'],
                     right_on=['movie_id', 'title'])
# endregion

def normalizeData(file_to_process,columnName):
    min_element = file_to_process[columnName].min()
    max_element = file_to_process[columnName].max()
    file_to_process[columnName] = 3 * (file_to_process[columnName] - min_element) / (max_element - min_element)

def standardizationData(file_to_process, columnName):
    file_to_process[columnName]=scale(file_to_process[columnName])

def convertDictColumnToScore(columnName, uniqueKey):  # with Y
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

        if len(element) == 0:
            sumForAllCells.append(sumForOneCell)
        else:
            sumForAllCells.append(sumForOneCell / len(element))
        sumForOneCell = 0
    wholeFile[columnName] = sumForAllCells
    return score_dict

#
# def convertDictColumnToScore(columnName, uniqueKey):
#     i = 0
#     counter = 0
#     _dict = defaultdict(float)  # el value: just #ocuurences || #ocuurences / size el file
#     _list = []
#     for cell in wholeFile[columnName]:  # cell: group of dictionaries
#         oneCell = json.loads(cell)
#         _list.append(oneCell)
#         if oneCell != {}:
#             counter += 1
#             for list_element in oneCell:
#                 for key, value in list_element.items():
#                     if key == uniqueKey:
#                         # each dictionary element has: #occurrencess, its score(each occurrence: add Y[i])
#                         if not _dict[value]:
#                             _dict[value] = 0.0
#
#                         _dict[value] = _dict[value] + 1  # first option
#                         # _dict[value] = (_dict[value] + 1) / len(file_to_process)  #second option
#
#         i += 1
#
#     # score_dict = defaultdict(float)
#     # for key, (num_occurr, score) in _dict.items():
#     #     score_dict[key] = score / num_occurr
#
#     sumForOneCell = 0
#     sumForAllCells = []
#     for element in _list:  # element: group of dictionaries
#         for list_element in element:  # list_element: one dictonary
#             for key, value in list_element.items():
#                 if key == uniqueKey:
#                     sumForOneCell += _dict[value]
#
#         if len(element) == 0:  # leh el condition dih ??? 3shan law cell fadya
#             sumForAllCells.append(sumForOneCell)
#         else:
#             sumForAllCells.append(sumForOneCell / len(_list))  # hn2sm bardo 3la size el cell ???
#         sumForOneCell = 0
#     wholeFile[columnName] = sumForAllCells
#     return _dict
#
def convertStringColToScore(colName):  # with Y
    i = 0
    _dict = defaultdict(tuple)
    _list = []
    for cell in wholeFile[colName]:
        if not _dict[cell]:
            _dict[cell] = (0, 0)

        _dict[cell] = (_dict[cell][0] + 1, _dict[cell][1] + Y[i])
        i += 1

    score_dict = defaultdict(float)
    for key, (num_occurr, score) in _dict.items():
        score_dict[key] = score / num_occurr
    lst = []
    for language in wholeFile[colName]:
        lst.append(_dict[language][1] / _dict[language][0])

    wholeFile[colName] = lst
    return score_dict

# def convertStringColToScore(colName):
#     i = 0
#     _dict = defaultdict(float)
#     _list = []
#     for cell in wholeFile[colName]:
#         if not _dict[cell]:
#             _dict[cell] = 0.0
#
#         _dict[cell] = _dict[cell] + 1
#         # _dict[cell] = (_dict[cell] + 1) / len(file_to_process)
#         i += 1
#
#     lst = []
#     for language in wholeFile[colName]:
#         lst.append(_dict[language])
#
#     wholeFile[colName] = lst
#     return _dict

def dataPreprocessing(file_to_process):

    file_to_process.drop(
        labels=['homepage', 'status', 'title', 'movie_id'],
        axis=1, inplace=True)
    file_to_process.replace(['', ' ', [[]], [], None, {}], np.nan, inplace=True)

    median = file_to_process['runtime'].median()
    file_to_process['runtime'].fillna(median, inplace=True)
    file_to_process["release_date"] = file_to_process["release_date"].astype('datetime64[ns]')
    # replace el date b-el year bs: first try.
    file_to_process["release_date"] = [i.year for i in file_to_process["release_date"]]

    # region NLP
    file_to_process["overview"].fillna("", inplace=True)
    overviewDictNLP = nlpData(file_to_process, "overview")
    normalizeData(file_to_process, 'overview')

    file_to_process["original_title"].fillna("", inplace=True)
    originalTitleDictNLP = nlpData(file_to_process, "original_title")
    normalizeData(file_to_process, 'original_title')

    file_to_process["tagline"].fillna("", inplace=True)
    taglineDictNLP = nlpData(file_to_process, "tagline")
    normalizeData(file_to_process, 'tagline')

    nlpDictColumns = []
    nlpDictColumns.append(originalTitleDictNLP)
    nlpDictColumns.append(overviewDictNLP)
    nlpDictColumns.append(taglineDictNLP)

    file_to_process.drop(labels=['id'], axis=1, inplace=True)

    # endregion

    # region Normalization of scalar data
    normalizeData(file_to_process,"budget")
    normalizeData(file_to_process,"popularity")
    normalizeData(file_to_process,"vote_count")
    # normalizeData("vote_average")
    normalizeData(file_to_process,"revenue")
    normalizeData(file_to_process,"runtime")

    # endregion

    # region pre-processing

    cast_dict = convertDictColumnToScore("cast", "id")
    # print("cast corr",wholeFile["cast"].corr(wholeFile['vote_average']))
    crew_dict = convertDictColumnToScore("crew", "id")
    Keywords_dict = convertDictColumnToScore("keywords", "name")
    spokenlang_dict = convertDictColumnToScore("spoken_languages", "name")
    genres_dict = convertDictColumnToScore("genres", "id")
    companies_dict = convertDictColumnToScore("production_companies", "id")
    countries_dict = convertDictColumnToScore("production_countries", "name")

    origLang_dict = convertStringColToScore("original_language")
    releaseDate_dict = convertStringColToScore("release_date")

    allcolumns_dict = []
    allcolumns_dict.append(genres_dict)
    allcolumns_dict.append(Keywords_dict)
    allcolumns_dict.append(companies_dict)
    allcolumns_dict.append(countries_dict)
    allcolumns_dict.append(spokenlang_dict)
    allcolumns_dict.append(cast_dict)
    allcolumns_dict.append(crew_dict)
    stringColumnsDict = []
    stringColumnsDict.append(origLang_dict)
    stringColumnsDict.append(releaseDate_dict)

    # endregion

    # region Normalize not scalar data
    normalizeData(file_to_process,"cast")
    normalizeData(file_to_process,"crew")
    normalizeData(file_to_process,"keywords")
    normalizeData(file_to_process,"spoken_languages")
    normalizeData(file_to_process,"genres")
    normalizeData(file_to_process,"production_companies")
    normalizeData(file_to_process,"production_countries")
    normalizeData(file_to_process,"original_language")
    normalizeData(file_to_process,"release_date")

    # endregion

    # print(wholeFile['original_language'].corr(wholeFile['vote_average']))

    return allcolumns_dict,stringColumnsDict, nlpDictColumns


def nlpData(file_to_process, columnName):
    colList = []
    for cell in file_to_process[columnName]:
        cell = re.sub('[^a-zA-Z]', ' ', cell)  # btsheel ay 7aga msh 7arf zay \
        cell = cell.lower()
        cell = cell.split()
        ps = PorterStemmer()  # bta5ud l noun bs
        cell = [ps.stem(word) for word in cell
                if not word in set(stopwords.words('english'))]
        cell = ' '.join(cell)
        colList.append(cell)

    cv = CountVectorizer(max_features=100)
    colData = cv.fit_transform(colList).toarray()
    colList = []
    colDict = dict()
    index = 0
    for i in colData:
        s = np.sum(i)
        colDict[file_to_process['id'][index]] = s
        colList.append(s)
        index+=1
    # colList.append(np.sum(i) for i in colData)
    file_to_process[columnName] = colList
    return colDict


def Convert_dict_TestFile(i,col,allcolumnsDict):
    col_List = []
    for cell in wholeTestFile[col]:  # cell: group of dictionaries
        oneCell = json.loads(cell)
        sumForOneCell = 0
        if oneCell != np.nan:
            for list_element in oneCell:
                for key, value in list_element.items():
                    if col == 'spoken_languages' or col =='production_countries':
                        if key == 'name':
                            # each dictionary element has: #occurrencess, its score(each occurrence: add Y[i])
                            if not allcolumnsDict[i][value]:
                                values = 0
                                for keys, value in allcolumnsDict[i].items():
                                    values += value
                                mean = values / len(allcolumnsDict[i])
                                sumForOneCell += mean
                            if allcolumnsDict[i][value]:
                                sumForOneCell += allcolumnsDict[i][value]
                    else:
                        if key == 'id':
                            # each dictionary element has: #occurrencess, its score(each occurrence: add Y[i])
                            if not allcolumnsDict[i][value]:
                                values = 0
                                for keys , value in allcolumnsDict[i].items():
                                    values+= value
                                mean = values/len(allcolumnsDict[i])
                                sumForOneCell += mean
                            if allcolumnsDict[i][value]:
                                sumForOneCell+=allcolumnsDict[i][value]

        elif oneCell == np.nan:
            values = 0
            for keys, value in allcolumnsDict[i].items():
                values += value
            mean = values / len(allcolumnsDict[i])
            sumForOneCell += mean
        col_List.append(sumForOneCell)
    wholeTestFile[col]=col_List


def Convert_string_TestFile(i,col,stringColDict):
    lst=[]
    values=0
    for cell in wholeTestFile[col]:
        if cell != np.nan:
            if stringColDict[i][cell]: #found in train dict
                lst.append(stringColDict[i][cell])
            if not stringColDict[i][cell]: #not found
                for keys, value in stringColDict[i].items():  # calculate mean for all values in column train
                    values += value

                mean = values / len(stringColDict[i])
                lst.append(mean)

        else:
            for keys, value in stringColDict[i].items():  # calculate mean for all values in column train
                values += value
            mean = values / len(stringColDict[i])
            lst.append(mean)

    wholeTestFile[col] = lst


def nlpTestFile(i, colName, nlpDictColumns):
    colList = []
    for index in range(len(wholeTestFile[colName])):
        idCheck = wholeTestFile['id'][index]
        if idCheck in nlpDictColumns[i].keys():  # bshuf law id el film el ana wa2fa feh fel test mwgod f dict bta3 el nlp
            colList.append(nlpDictColumns[i][idCheck])
        else:
            colList.append(0)
    wholeTestFile[colName] = colList


def Testing(X_train,Y_train,allcolumnsDict,stringColDict,nlpDictColumns):

    wholeTestFile.drop(
        labels=['homepage', 'status','title', 'movie_id'],
        axis=1, inplace=True)
    wholeTestFile.replace(['', ' ', [[]], [], None, {}], np.nan, inplace=True)

    median = wholeTestFile['runtime'].median()
    wholeTestFile['runtime'].fillna(median, inplace=True)

    median = wholeTestFile['revenue'].median()
    wholeTestFile['revenue'].fillna(median, inplace=True)

    median = wholeTestFile['budget'].median()
    wholeTestFile['budget'].fillna(median, inplace=True)

    median = wholeTestFile['vote_count'].median()
    wholeTestFile['vote_count'].fillna(median, inplace=True)


    wholeTestFile["release_date"] = wholeTestFile["release_date"].astype('datetime64[ns]')
    # replace el date b-el year bs: first try.
    wholeTestFile["release_date"] = [i.year for i in wholeTestFile["release_date"]]

    # region Dictionary columns

    ColumnNamesList=['genres','keywords','production_companies','production_countries','spoken_languages','cast','crew']
    for i in range(len(ColumnNamesList)):
        Convert_dict_TestFile(i,ColumnNamesList[i],allcolumnsDict)

    # endregion

    # region String Columns
    stringColumnsNames=['original_language','release_date']
    for i in range(len(stringColumnsNames)):
        Convert_string_TestFile(i,stringColumnsNames[i],stringColDict)
    # endregion

    #region NLP
    columnsNLP = ['original_title','overview','tagline']
    for i in range(len(columnsNLP)):
        nlpTestFile(i, columnsNLP[i], nlpDictColumns)
    # endregion

    #region Normalization
    normalizeData(wholeTestFile, "popularity")
    normalizeData(wholeTestFile, "revenue")
    normalizeData(wholeTestFile, "runtime")
    normalizeData(wholeTestFile, "vote_count")
    normalizeData(wholeTestFile, "budget")
    normalizeData(wholeTestFile, "cast")
    normalizeData(wholeTestFile, "crew")
    normalizeData(wholeTestFile, "keywords")
    normalizeData(wholeTestFile, "spoken_languages")
    normalizeData(wholeTestFile, "genres")
    normalizeData(wholeTestFile, "production_companies")
    normalizeData(wholeTestFile, "production_countries")
    normalizeData(wholeTestFile, "original_language")
    normalizeData(wholeTestFile, "release_date")
    #endregion

    wholeTestFile.drop(labels=['id'], axis=1, inplace=True)

    wholeTestFile.to_csv("finalOutputTest.csv", index=False)

    wholeTestFileNew =wholeTestFile
    wholeTestFileNew = wholeTestFileNew.iloc[:,0:18]
    X_test = wholeTestFileNew.drop(axis=1,labels="vote_average")

    Y_test = wholeTestFileNew["vote_average"]

    print("Testing")
    # regressions(X_train,X_test,Y_train,Y_test)
    return X_test, Y_test

def drawFeatures(model, colName, X_test, X_train, Y_test, Y_train):
    #
    x1 = X_test[colName]
    x2 = Y_test
    label = Y_test
    colors = ['blue', 'purple']

    plt.scatter(x1, x2, c=label, cmap=matplotlib.colors.ListedColormap(colors))
    plt.xlabel('X1', fontsize=20)
    plt.ylabel('X2', fontsize=20)

    cb = plt.colorbar()
    loc = np.arange(0, max(label), max(label) / float(len(colors)))
    cb.set_ticks(loc)
    cb.set_ticklabels([colName, 'vote_average'])

    # line
    col_indx = X_test.columns.get_loc(colName)
    x = X_train
    x = np.array(X_train)
    x = x[:, col_indx]
    x = x.reshape(-1, 1)
    Y_train = np.array(Y_train).reshape(-1, 1)  # v=np.array(X_train[colName])
    model.fit(x, Y_train)
    x_test = np.array(X_test[colName]).reshape(-1, 1)
    predictions = model.predict(x_test)
    plt.plot(x_test, predictions, color='black', linewidth=1.5)
    plt.show()

def regressions(X_train,X_test,Y_train,Y_test):
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
    # region Multiple linear regression with all features
    # Technique 1
    # Top Features
    print("Multiple linear:Top Features")
    regression2 = linear_model.LinearRegression()
    regression2.fit(X_train[top_features], Y_train)
    predictions = regression2.predict(X_test[top_features])
    print("Accuracy:", 100 * metrics.r2_score(Y_test, predictions))
    print("MSE:", metrics.mean_squared_error(Y_test, predictions))
    print("__________________________")
    print("Multiple linear:All Features")
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
    print("Polynomial Top Features")
    # poly_features_train = PolynomialFeatures(2).fit_transform(X_train)
    # poly_features_test = PolynomialFeatures(2).fit_transform(X_test)

    poly_features_train = PolynomialFeatures(2).fit_transform(X_train[top_features])
    poly_features_test = PolynomialFeatures(2).fit_transform(X_test[top_features])

    poly_model = linear_model.LinearRegression()
    poly_model.fit(poly_features_train, Y_train)

    predictions = poly_model.predict(poly_features_test)
    print("Accuracy:", 100 * metrics.r2_score(Y_test, predictions))
    print("MSE:", metrics.mean_squared_error(Y_test, predictions))
    print("__________________________")

    print("Polynomial All Features")
    poly_features_train = PolynomialFeatures(2).fit_transform(X_train)
    poly_features_test = PolynomialFeatures(2).fit_transform(X_test)

    poly_model2 = linear_model.LinearRegression()
    poly_model2.fit(poly_features_train, Y_train)

    predictions = poly_model2.predict(poly_features_test)
    print("Accuracy:", 100 * metrics.r2_score(Y_test, predictions))
    print("MSE:", metrics.mean_squared_error(Y_test, predictions))

    # endregion

    # region SVR regression

    print("SVR Kernal: rbf Top Features")
    svr_rbf = SVR(kernel='rbf', C=1, gamma=1, epsilon=.1).fit(X_train[top_features], Y_train)
    svr_lin = SVR(kernel='linear', C=1).fit(X_train[top_features], Y_train)
    svr_poly = SVR(kernel='poly', C=1, gamma=1, degree=2, epsilon=.1, coef0=1).fit(X_train[top_features], Y_train)
    # print("linear coeff", svr_lin.coef_)
    # print("bias", svr_lin.intercept_)

    predictions_rbf = svr_rbf.predict(X_test[top_features])
    print("Accuracy:", metrics.r2_score(Y_test, predictions_rbf) * 100)  # np.mean(predictions == Y_validation)*100)
    print("MSE:", metrics.mean_squared_error(Y_test, predictions_rbf))

    print("__________________________")
    print("SVR Kernal: lin Top Features")
    predictions_lin = svr_lin.predict(X_test[top_features])
    print("Accuracy:", metrics.r2_score(Y_test, predictions_lin) * 100)  # np.mean(predictions == Y_validation)*100)
    print("MSE:", metrics.mean_squared_error(Y_test, predictions_lin))
    print("__________________________")
    print("SVR Kernal: poly Top Features")
    predictions_poly = svr_poly.predict(X_test[top_features])
    print("Accuracy:", metrics.r2_score(Y_test, predictions_poly) * 100)  # np.mean(predictions == Y_validation)*100)
    print("MSE:", metrics.mean_squared_error(Y_test, predictions_poly))
    print("__________________________")

    print("SVR rbf: All Features")
    svr_rbf2 = SVR(kernel='rbf', C=1, gamma=1, epsilon=.1).fit(X_train, Y_train)
    predictions = svr_rbf2.predict(X_test)
    print("Accuracy:", metrics.r2_score(Y_test, predictions) * 100)  # np.mean(predictions == Y_validation)*100)
    print("MSE:", metrics.mean_squared_error(Y_test, predictions))

    print("__________________________")

    # # print("SVR poly: All Features")
    # svr_poly = SVR(kernel='poly', C=1, gamma=1, epsilon=.1).fit(X_train, Y_train)
    # predictions = svr_poly.predict(X_test)
    # print("Accuracy:", metrics.r2_score(Y_test, predictions) * 100)
    # print("MSE:", metrics.mean_squared_error(Y_test, predictions))
    # print("__________________________")

    print("SVR lin: All Features")
    svr_lin = SVR(kernel='linear', C=1, gamma=1, epsilon=.1).fit(X_train, Y_train)
    predictions = svr_lin.predict(X_test)
    print("Accuracy:", metrics.r2_score(Y_test, predictions) * 100)  # np.mean(predictions == Y_validation)*100)
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

    # region Ridge Regression
    print("Ridge Top Features")
    ridge_regression = Ridge(alpha=.5)
    # ridge_regression.fit(X_train, Y_train)
    ridge_regression.fit(X_train[top_features], Y_train)
    # test_score = ridge_regression.score(X_test, Y_test)
    # test_score = ridge_regression.score(X_test[top_features], Y_test)
    # print("Ridge regression score", test_score*100)
    # predictions = ridge_regression.predict(X_test)
    predictions = ridge_regression.predict(X_test[top_features])
    print("Accuracy:", metrics.r2_score(Y_test, predictions) * 100)
    print("MSE:", metrics.mean_squared_error(Y_test, predictions))
    print("__________________________")
    print("Ridge All Features")
    ridge_regression2 = Ridge(alpha=.5)

    ridge_regression2.fit(X_train, Y_train)

    predictions = ridge_regression2.predict(X_test)
    print("Accuracy:", metrics.r2_score(Y_test, predictions) * 100)
    print("MSE:", metrics.mean_squared_error(Y_test, predictions))
    # print("ridge coeff", ridge_regression.coef_)
    # drawFeatures("runtime", ridge_regression.coef_, ridge_regression.intercept_, X_test, Y_test)
    for name in top_features:
        drawFeatures(ridge_regression2, name, X_test, X_train, Y_test, Y_train)
    # drawFeatures("production_companies", X_test, Y_test, ridge_regression.coef_, ridge_regression.intercept_)

    # drawFeatures("cast", X_test, Y_test, ridge_regression.coef_, ridge_regression.intercept_)
    # drawFeatures("crew", X_test, Y_test, ridge_regression.coef_, ridge_regression.intercept_)

    # endregion
    # region Lasso Regression

    '''print("Lasso")

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
    print("---------------------------------")'''

    '''print("---------------------------------")
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
    print("MSE:", test_error)'''

    # endregion

    # c = pd.DataFrame(regression.coef_, X.columns, columns=["CCC"])
    # print("C")
    # print(c)
    return regression2,regression,poly_model,poly_model2, svr_rbf,svr_rbf2,ridge_regression,ridge_regression2

# endregion

# def drawFeatures(colName, weights, b, X_test, Y_test):
# #
#     x1 = X_test[colName]
#     x2 = Y_test
#     label = Y_test
#     colors = ['blue', 'purple']
#
#     plt.scatter(x1, x2, c=label, cmap=matplotlib.colors.ListedColormap(colors))
#     plt.xlabel('X1', fontsize=20)
#     plt.ylabel('X2', fontsize=20)
#
#     cb = plt.colorbar()
#     loc = np.arange(0, max(label), max(label) / float(len(colors)))
#     cb.set_ticks(loc)
#     cb.set_ticklabels([colName, 'vote_average'])
#
#     x2 = min(X_test[colName]) - 5
#     col_indx = X_test.columns.get_loc(colName)
#     w2 = weights[col_indx]
#     w1=1
#     x1 = ((-w2 * x2) - b) / w1
#
#     pointX, pointY = [], []
#     pointX.append(x2)
#     pointY.append(x1)
#
#     x1 = max(X_test[colName]) + 5
#     x2 = ((-w1 * x1) - b) / w2
#
#     pointX.append(x1)
#     pointY.append(x2)
#
#     plt.plot(pointX, pointY, color='green', linewidth=2)
#
#     plt.show()
#     #endregion

def main():
    # print("len of whole file before drop", len(wholeFile))


    # pre-processing:
    allcolDict,stringColDict, nlpDictColumns = dataPreprocessing(wholeFile)

    # after pre-processing:
    X = wholeFile.drop(axis=1, labels="vote_average")
    Y = wholeFile["vote_average"]

    corr = wholeFile.corr()
    # corr.style.background_gradient(cmap='coolwarm', axis=None)

    # Draw correlation matrix:
    plt.matshow(wholeFile.corr())
    plt.show()

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)
    # X_train, X_validation, Y_train, Y_validation = train_test_split(X_train, Y_train, test_size=0.2, shuffle=True)

    # region Select top features
    # top_features = wholeFile.corr().index[abs(wholeFile.corr()['vote_average'] >= 0.3)]
    # top_features = top_features.drop("vote_average")
    # print("top_features using corr()")
    # print(top_features)

    regression2,regression,poly_model,poly_model2, svr_rbf,svr_rbf2,ridge_regression,ridge_regression2= regressions(X_train,X_test,Y_train,Y_test)

    X_testNew , Y_testNew = Testing(X, Y, allcolDict, stringColDict, nlpDictColumns)

    # predictions = regression2.predict(X_testNew)
    # print("Multiple linear:Top Features")
    # print("Accuracy:", 100 * metrics.r2_score(Y_testNew, predictions))
    # print("MSE:", metrics.mean_squared_error(Y_testNew, predictions))
    # print("__________________________")

    print("Multiple linear:All Features")
    predictions = regression.predict(X_testNew)
    print("Accuracy:", 100 * metrics.r2_score(Y_testNew, predictions))
    print("MSE:", metrics.mean_squared_error(Y_testNew, predictions))
    print("__________________________")

    # print("Polynomial Top Features")
    # predictions = poly_model.predict(X_testNew)
    # print("Accuracy:", 100 * metrics.r2_score(Y_testNew, predictions))
    # print("MSE:", metrics.mean_squared_error(Y_testNew, predictions))
    # print("__________________________")

    print("Polynomial All Features")
    poly_features_test = PolynomialFeatures(2).fit_transform(X_testNew)
    predictions = poly_model2.predict(poly_features_test)
    print("Accuracy:", 100 * metrics.r2_score(Y_testNew, predictions))
    print("MSE:", metrics.mean_squared_error(Y_testNew, predictions))
    print("__________________________")

    # predictions = svr_rbf.predict(X_testNew)
    # print("Accuracy:", 100 * metrics.r2_score(Y_testNew, predictions))
    # print("MSE:", metrics.mean_squared_error(Y_testNew, predictions))
    # print("__________________________")

    predictions = svr_rbf2.predict(X_testNew)
    print("Accuracy:", 100 * metrics.r2_score(Y_testNew, predictions))
    print("MSE:", metrics.mean_squared_error(Y_testNew, predictions))
    print("__________________________")

    # predictions = ridge_regression.predict(X_testNew)
    # print("Accuracy:", 100 * metrics.r2_score(Y_testNew, predictions))
    # print("MSE:", metrics.mean_squared_error(Y_testNew, predictions))
    # print("__________________________")


wholeFile.to_csv("finalOutput.csv", index=False)


main()
