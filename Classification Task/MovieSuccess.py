import json
from collections import defaultdict
import numpy as np
import pandas as pd
from sklearn import tree
import re
import nltk

nltk.download('stopwords')
from nltk.stem.porter import PorterStemmer  # 3shan ttla3 el esm lwa7do, zay fish mn fisher aw fishing
from nltk.corpus import stopwords  # 3shan nsheel el stopwords
from sklearn.feature_extraction.text import CountVectorizer
# import seaborn as sns
# import matplotlib
# from sklearn import linear_model, metrics
# from sklearn.linear_model import Lasso, Ridge, ElasticNet
# from sklearn.svm import SVR
# from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures, LabelEncoder, scale
from sklearn.feature_selection import SelectKBest, chi2, f_classif
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
import matplotlib.pyplot as plt
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.decomposition import PCA
import timeit
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.externals import joblib
from sklearn.preprocessing import StandardScaler
import os

# region Public
# for ploting
accuracies_list = []
classifiers_names = []
train_time_list = []
test_time_list = []

##
moviesFile = pd.read_csv('Train Set/tmdb_5000_movies_classification.csv')
creditsFile = pd.read_csv('Train Set/tmdb_5000_credits.csv')
testfilemovies = pd.read_excel('movies_samples/samples_tmdb_5000_movies_testing_classification.xlsx')
testfilecredits = pd.read_csv('movies_samples/samples_tmdb_5000_credits_test.csv')

# inner y keep rl 7agat el comman fl 2 df bs
wholeFile = pd.merge(moviesFile, creditsFile, sort=True, how='inner', left_on=['id', 'title'],
                     right_on=['movie_id', 'title'])
# wholeFile = pd.read_csv('out.csv')
# print("Y", type(Y))
X = wholeFile.drop(axis=1, labels="rate")
Y = wholeFile["rate"]

wholeTestFile = pd.merge(testfilemovies, testfilecredits, sort=True, how='inner', left_on=['id', 'title'],
                         right_on=['movie_id', 'title'])


# endregion
def plot_graphs(list_, classifier_names, title):
    print("len(list_)", len(list_))
    print(list_)
    n_groups = len(list_)

    # create plot
    fig, ax = plt.subplots()
    bar_width = 0.35
    opacity = 0.8
    index = np.arange(n_groups)
    index = index + bar_width
    color = ['blue', 'c', 'pink', 'purple', 'green', 'brown', 'orange', "yellow", "Black"]

    for i in range(len(list_)):
        plt.bar(index[i], list_[i], bar_width, alpha=opacity, color=color[i], label=classifier_names[i])

    plt.xlabel('Classifiers')
    plt.ylabel(title)
    plt.title(title)
    plt.xticks(index + bar_width, (classifier_names[0], classifier_names[1], classifier_names[2], classifier_names[3]))
    plt.legend()

    plt.tight_layout()
    plt.show()


def standardizationData(file_to_process, columnName):
    scaler = StandardScaler()
    print("file_to_process[columnName])", type(file_to_process[columnName]))
    col = scaler.fit_transform(file_to_process[columnName])
    print("type(col)", type(col))

    file_to_process[columnName] = col.tolist()


def normalizeData(file_to_process, columnName):
    min_element = file_to_process[columnName].min()
    max_element = file_to_process[columnName].max()
    file_to_process[columnName] = 3 * (file_to_process[columnName] - min_element) / (max_element - min_element)
    return min_element, max_element


def normalizeTest_Data(min_element, max_element, columnName):
    wholeTestFile[columnName] = 3 * (wholeFile[columnName] - min_element) / (max_element - min_element)


def convertDictColumnToScore(file_to_process, columnName, uniqueKey):
    i = 0
    counter = 0
    _dict = defaultdict(float)  # el value: just #ocuurences || #ocuurences / size el file
    _list = []
    for cell in file_to_process[columnName]:  # cell: group of dictionaries
        oneCell = json.loads(cell)
        _list.append(oneCell)
        if oneCell != {}:
            counter += 1
            for list_element in oneCell:
                for key, value in list_element.items():
                    if key == uniqueKey:
                        # each dictionary element has: #occurrencess, its score(each occurrence: add Y[i])
                        if not _dict[value]:
                            _dict[value] = 0.0

                        _dict[value] = _dict[value] + 1  # first option
                        # _dict[value] = (_dict[value] + 1) / len(file_to_process)  #second option

        i += 1

    # score_dict = defaultdict(float)
    # for key, (num_occurr, score) in _dict.items():
    #     score_dict[key] = score / num_occurr

    sumForOneCell = 0
    sumForAllCells = []
    for element in _list:  # element: group of dictionaries
        for list_element in element:  # list_element: one dictonary
            for key, value in list_element.items():
                if key == uniqueKey:
                    sumForOneCell += _dict[value]

        if len(element) == 0:  # leh el condition dih ??? 3shan law cell fadya
            sumForAllCells.append(sumForOneCell)
        else:
            sumForAllCells.append(sumForOneCell / len(_list))  # hn2sm bardo 3la size el cell ???
        sumForOneCell = 0
    file_to_process[columnName] = sumForAllCells


def convertStringColToScore(file_to_process, colName):
    i = 0
    _dict = defaultdict(float)
    _list = []
    for cell in file_to_process[colName]:
        if not _dict[cell]:
            _dict[cell] = 0.0

        _dict[cell] = _dict[cell] + 1
        # _dict[cell] = (_dict[cell] + 1) / len(file_to_process)
        i += 1

    lst = []
    for language in file_to_process[colName]:
        lst.append(_dict[language])

    file_to_process[colName] = lst


def dataPreprocessing(file_to_process):
    file_to_process.drop(
        labels=['id', 'homepage', 'status', 'tagline', 'title', 'movie_id'],
        axis=1, inplace=True)
    file_to_process.replace(['', ' ', [[]], [], None, {}], np.nan, inplace=True)

    median = file_to_process['runtime'].median()
    file_to_process['runtime'].fillna(median, inplace=True)
    file_to_process["release_date"] = file_to_process["release_date"].astype('datetime64[ns]')
    # replace el date b-el year bs: first try.
    file_to_process["release_date"] = [i.year for i in file_to_process["release_date"]]
    WholeFileCopy = file_to_process
    # region NLP
    file_to_process["overview"].fillna("", inplace=True)
    nlpData(file_to_process, "overview")
    # standardizationData(file_to_process, "overview")
    normalizeData(file_to_process, "overview")
    # print("done nlp with overview")

    file_to_process["original_title"].fillna("", inplace=True)
    nlpData(file_to_process, "original_title")
    # standardizationData(file_to_process, "original_title")
    normalizeData(file_to_process, "original_title")
    # print("done nlp with original_title")
    # endregion

    # region Standardization data
    '''standardizationData(file_to_process, "popularity")
    standardizationData(file_to_process, "revenue")
    standardizationData(file_to_process, "runtime")
    standardizationData(file_to_process, "vote_count")'''
    # endregion
    # region normalization
    normalizeData(file_to_process, "budget")
    normalizeData(file_to_process, "popularity")
    normalizeData(file_to_process, "vote_count")
    # normalizeData("vote_average")
    normalizeData(file_to_process, "revenue")
    normalizeData(file_to_process, "runtime")
    # end region
    # region pre-processing
    convertDictColumnToScore(file_to_process, "cast", "id")
    # print("cast corr",wholeFile["cast"].corr(wholeFile['rate']))
    convertDictColumnToScore(file_to_process, "crew", "id")
    convertDictColumnToScore(file_to_process, "keywords", "name")
    convertDictColumnToScore(file_to_process, "spoken_languages", "name")
    convertDictColumnToScore(file_to_process, "genres", "id")
    convertDictColumnToScore(file_to_process, "production_companies", "id")
    convertDictColumnToScore(file_to_process, "production_countries", "iso_3166_1")
    convertStringColToScore(file_to_process, "original_language")
    convertStringColToScore(file_to_process, "release_date")
    # endregion

    # region standardization of non scalar data
    '''standardizationData(file_to_process, "cast")
    standardizationData(file_to_process, "crew")
    standardizationData(file_to_process, "keywords")
    standardizationData(file_to_process, "spoken_languages")
    standardizationData(file_to_process, "genres")
    standardizationData(file_to_process, "production_companies")
    standardizationData(file_to_process, "production_countries")
    standardizationData(file_to_process, "original_language")
    standardizationData(file_to_process, "release_date")'''

    # endregion
    normalizeData(file_to_process, "cast")
    normalizeData(file_to_process, "crew")
    normalizeData(file_to_process, "keywords")
    normalizeData(file_to_process, "spoken_languages")
    normalizeData(file_to_process, "genres")
    normalizeData(file_to_process, "production_companies")
    normalizeData(file_to_process, "production_countries")
    normalizeData(file_to_process, "original_language")
    normalizeData(file_to_process, "release_date")
    # region encoding Y column ()rate
    le = LabelEncoder()
    le.fit(file_to_process["rate"])
    file_to_process["rate"] = le.transform(file_to_process["rate"])
    # endregion

    # print("overview corr", wholeFile["overview"].corr(wholeFile['rate']))
    # print("original_title corr", wholeFile["original_title"].corr(wholeFile['rate']))
    return WholeFileCopy


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
    for i in colData:
        s = np.sum(i)
        colList.append(s)
    # colList.append(np.sum(i) for i in colData)
    file_to_process[columnName] = colList


# region classifiers
def DT_Classifier(X_train, X_test, Y_train, Y_test):
    start_train = timeit.default_timer()
    DT_Classifier = tree.DecisionTreeClassifier()
    DT_Classifier.fit(X_train, Y_train)
    stop_train = timeit.default_timer()
    DT_train_time = stop_train - start_train

    start_test = timeit.default_timer()
    predictions = DT_Classifier.predict(X_test)
    DT_accuracy = np.mean(predictions == Y_test)
    stop_test = timeit.default_timer()
    DT_test_time = stop_test - start_test

    train_time_list.append(DT_train_time)
    test_time_list.append(DT_test_time)
    classifiers_names.append('DT')
    accuracies_list.append(DT_accuracy)

    print("Accuracy of decision tree is", DT_accuracy * 100, '%')
    return DT_Classifier

def Naive_Bias(X_train, X_test, Y_train, Y_test):
    start_train = timeit.default_timer()
    gaussian_Classifier = GaussianNB()
    gaussian_Classifier.fit(X_train, Y_train)
    stop_train = timeit.default_timer()
    gaussian_train_time = stop_train - start_train

    start_test = timeit.default_timer()
    predictions = gaussian_Classifier.predict(X_test)
    gaussian_accuracy = np.mean(predictions == Y_test)
    stop_test = timeit.default_timer()
    gaussian_test_time = stop_test - start_test

    train_time_list.append(gaussian_train_time)
    test_time_list.append(gaussian_test_time)
    classifiers_names.append('Naive_Bias')
    accuracies_list.append(gaussian_accuracy  )

    print("Accuracy of Naive Bayes is", gaussian_accuracy * 100, '%')
    return gaussian_Classifier
    # print("Confusion matrix:")
    # print(confusion_matrix(Y_test, predictions))
    # print("Classification report:")
    # print(classification_report(Y_test, predictions))


def MLP_Classifier(X_train, X_test, Y_train, Y_test):
    start_train = timeit.default_timer()
    MLP_Classifier = MLPClassifier()
    MLP_Classifier.fit(X_train, Y_train)
    stop_train = timeit.default_timer()
    MLP_train_time = stop_train - start_train

    start_test = timeit.default_timer()

    predictions = MLP_Classifier.predict(X_test)
    MLP_accuracy = np.mean(predictions == Y_test)
    stop_test = timeit.default_timer()

    MLP_test_time = stop_test - start_test

    train_time_list.append(MLP_train_time)
    test_time_list.append(MLP_test_time)
    classifiers_names.append('MLP')
    accuracies_list.append(MLP_accuracy  )
    print("Accuracy of MLPClassifier is", MLP_accuracy * 100, '%')
    return MLP_Classifier
    # print("train time", MLP_train_time)
    # print("test time", MLP_test_time)
    #
    # print("Confusion matrix:")
    # print(confusion_matrix(Y_test, predictions))
    # print("Classification report:")
    # print(classification_report(Y_test, predictions))


def logisticRegCLF(X_train, X_test, Y_train, Y_test):
    start_train = timeit.default_timer()
    logisticRegCLF = LogisticRegression()
    logisticRegCLF.fit(X_train, Y_train)

    stop_train = timeit.default_timer()
    logisticRegCLF_train_time = stop_train - start_train

    start_test = timeit.default_timer()
    logisticRegCLF.predict(X_test)
    accuracyLogReg = logisticRegCLF.score(X_test, Y_test)
    stop_test = timeit.default_timer()
    logisticRegCLF_test_time = stop_test - start_test

    train_time_list.append(logisticRegCLF_train_time)
    test_time_list.append(logisticRegCLF_test_time)
    classifiers_names.append('LogReg')
    accuracies_list.append(accuracyLogReg  )
    print("Accuracy of logistic regression: ", accuracyLogReg * 100, '%')
    return logisticRegCLF

def knnCLF(X_train, X_test, Y_train, Y_test):
    # first choice of hyperpparameter n, n = 5
    knnCLF = KNeighborsClassifier(n_neighbors=3)
    knnCLF.fit(X_train, Y_train)
    Y_pred = knnCLF.predict(X_test)
    accuracyKNN = np.mean(Y_pred == Y_test)
    print("KNN (n=5) accuracy: ", accuracyKNN *100 , '%')
    # print("KNN confusion matrix:")
    # print(confusion_matrix(Y_test, Y_pred))
    # print("KNN classification report:")
    # print(classification_report(Y_test, Y_pred))

    # first choice of hyperpparameter n, n = 5
    knnCLF = KNeighborsClassifier(n_neighbors=5)
    knnCLF.fit(X_train, Y_train)
    Y_pred = knnCLF.predict(X_test)
    accuracyKNN = np.mean(Y_pred == Y_test)
    print("KNN (n=3) accuracy: ", accuracyKNN * 100, '%')
    # print("KNN confusion matrix:")
    # print(confusion_matrix(Y_test, Y_pred))
    # print("KNN classification report:")
    # print(classification_report(Y_test, Y_pred))

    # first choice of hyperpparameter n, n = 10

    start_train = timeit.default_timer()
    knnCLF = KNeighborsClassifier(n_neighbors=11)
    knnCLF.fit(X_train, Y_train)
    stop_train = timeit.default_timer()
    knn_train_time = stop_train - start_train

    start_test = timeit.default_timer()
    Y_pred = knnCLF.predict(X_test)
    accuracyKNN = np.mean(Y_pred == Y_test)
    stop_test = timeit.default_timer()
    knn_test_time = stop_test - start_test

    train_time_list.append(knn_train_time)
    test_time_list.append(knn_test_time)
    classifiers_names.append('KNN')
    accuracies_list.append(accuracyKNN  )
    print("KNN (n=11) accuracy: ", accuracyKNN * 100, '%')
    return knnCLF

    # print("KNN confusion matrix:")
    # print(confusion_matrix(Y_test, Y_pred))
    # print("KNN classification report:")
    # print(classification_report(Y_test, Y_pred))


def GradientBoost(X_train, X_test, Y_train, Y_test):
    start_train = timeit.default_timer()
    GradientBooster = GradientBoostingClassifier(criterion='mse', warm_start=True, n_estimators=200)
    GradientBooster.fit(X_train, Y_train)
    stop_train = timeit.default_timer()
    gb_train_time = stop_train - start_train

    start_test = timeit.default_timer()
    predictions = GradientBooster.predict(X_test)
    accuracy = np.mean(predictions == Y_test) * 100
    acc = GradientBooster.score(X_test, Y_test)
    stop_test = timeit.default_timer()
    gb_test_time = stop_test - start_test

    train_time_list.append(gb_train_time)
    test_time_list.append(gb_test_time)
    classifiers_names.append('gb')
    accuracies_list.append(accuracy  )
    print("Accuracy of Gradient Booster is " + str(accuracy) + '%')
    return GradientBooster


def AdaBoostCLS(X_train, X_test, Y_train, Y_test):
    start_train = timeit.default_timer()
    bdt = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), algorithm="SAMME.R", n_estimators=100)
    bdt.fit(X_train, Y_train)
    stop_train = timeit.default_timer()
    AdaBoost_train_time = stop_train - start_train

    start_test = timeit.default_timer()
    y_prediction = bdt.predict(X_test)
    accuracy = np.mean(y_prediction == Y_test) * 100
    stop_test = timeit.default_timer()
    AdaBoost_test_time = stop_test - start_test

    train_time_list.append(AdaBoost_train_time)
    test_time_list.append(AdaBoost_test_time)
    classifiers_names.append('AdaBoost')
    accuracies_list.append(accuracy  )
    print("Accuracy of Adaboost is " + str(accuracy) + '%')
    return bdt


def SVM_One_VS_Rest(X_train, X_test, Y_train, Y_test):
    start_train = timeit.default_timer()

    svm_model_linear_ovr = OneVsRestClassifier(SVC(kernel='linear', C=1)).fit(X_train, Y_train)
    stop_train = timeit.default_timer()
    OVR_train_time = stop_train - start_train

    start_test = timeit.default_timer()
    svm_predictions = svm_model_linear_ovr.predict(X_test)
    accuracyOVR = svm_model_linear_ovr.score(X_test, Y_test)
    stop_test = timeit.default_timer()
    OVR_test_time = stop_test - start_test

    train_time_list.append(OVR_train_time)
    test_time_list.append(OVR_test_time)
    classifiers_names.append('OVR')
    accuracies_list.append(accuracyOVR  )
    print('Accuracy of One VS Rest SVM is', accuracyOVR * 100, '%')
    return svm_model_linear_ovr


def SVM_One_VS_One(X_train, X_test, Y_train, Y_test):
    start_train = timeit.default_timer()
    svm_model_linear_ovo = SVC(kernel='linear', C=1).fit(X_train, Y_train)
    stop_train = timeit.default_timer()
    OVO_train_time = stop_train - start_train

    start_test = timeit.default_timer()
    svm_predictions = svm_model_linear_ovo.predict(X_test)
    accuracyOVO = svm_model_linear_ovo.score(X_test, Y_test)

    stop_test = timeit.default_timer()
    OVO_test_time = stop_test - start_test

    train_time_list.append(OVO_train_time)
    test_time_list.append(OVO_test_time)
    classifiers_names.append('OVO')
    accuracies_list.append(accuracyOVO  )
    print('Accuracy of One VS Rest SVM is', accuracyOVO * 100, '%')
    return svm_model_linear_ovo


# endregion

def PCA_DimnRedBest(X_train, X_test):  # ngrb n8yar 3dd el components b 3 values
    pca = PCA(n_components=2)
    print(X_train.shape)
    new_Xtrain = pca.fit_transform(X_train)
    print(new_Xtrain.shape)

    print(X_test.shape)
    new_Xtest = pca.transform(X_test)
    print(new_Xtest.shape)
    return new_Xtrain, new_Xtest


def PCA_DimnRedAll(X_train, X_test):  # ngrb n8yar 3dd el components b 3 values
    pca = PCA(n_components=8)
    print(X_train.shape)
    new_Xtrain = pca.fit_transform(X_train)
    print(new_Xtrain.shape)

    print(X_test.shape)
    new_Xtest = pca.transform(X_test)
    print(new_Xtest.shape)
    return new_Xtrain, new_Xtest


def Testing(X_train, Y_train, trainCopyFile):
    # dataPreprocessing(wholeTestFile)
    wholeTestFile.drop(
        labels=['id', 'homepage', 'status', 'tagline', 'title', 'movie_id', 'overview', 'original_title'],
        axis=1, inplace=True)
    wholeTestFile.replace(['', ' ', [[]], [], None, {}], np.nan, inplace=True)
    # fill empty rows

    wholeTestFile["release_date"] = wholeTestFile["release_date"].astype('datetime64[ns]')
    # replace el date b-el year bs: first try.
    wholeTestFile["release_date"] = [i.year for i in wholeTestFile["release_date"]]

    median = wholeFile['runtime'].median()
    wholeTestFile['runtime'].fillna(median, inplace=True)
    # check lw ba2y col feha null n fill b median el train
    ##convert list to score
    convertDictColumnToScore(wholeTestFile, "cast", "id")
    convertDictColumnToScore(wholeTestFile, "crew", "id")
    convertDictColumnToScore(wholeTestFile, "keywords", "name")
    convertDictColumnToScore(wholeTestFile, "spoken_languages", "name")
    convertDictColumnToScore(wholeTestFile, "genres", "id")
    convertDictColumnToScore(wholeTestFile, "production_companies", "id")
    convertDictColumnToScore(wholeTestFile, "production_countries", "iso_3166_1")
    convertStringColToScore(wholeTestFile, "original_language")
    convertStringColToScore(wholeTestFile, "release_date")
    ##standardization

    # encoding
    for colname in trainCopyFile:
        min, max = normalizeData(trainCopyFile, colname)
        normalizeTest_Data(min, max, colname)

    le = LabelEncoder()
    le.fit(wholeTestFile["rate"])
    wholeTestFile["rate"] = le.transform(wholeTestFile["rate"])

    X_test = wholeTestFile.drop(axis=1, labels="rate")

    Y_test = wholeTestFile["rate"]

    # scaler = StandardScaler()
    # scaler.fit(X_train)
    # X_train = scaler.transform(X_train)
    # X_test = scaler.transform(X_test)

    print("Testing")
    AdaBoostCLS(X_train, X_test, Y_train, Y_test)
    GradientBoost(X_train, X_test, Y_train, Y_test)
    DT_Classifier(X_train, X_test, Y_train, Y_test)
    Naive_Bias(X_train, X_test, Y_train, Y_test)
    MLP_Classifier(X_train, X_test, Y_train, Y_test)
    logisticRegCLF(X_train, X_test, Y_train, Y_test)
    knnCLF(X_train, X_test, Y_train, Y_test)
    # SVM_One_VS_Rest(X_train, X_test, Y_train, Y_test)
    # SVM_One_VS_One(X_train, X_test, Y_train, Y_test)


def main():
    # print("len wholeFile before:", len(wholeFile.columns))
    # print(wholeFile.columns)
    # pre-processing:
    TrainCopyFile = dataPreprocessing(wholeFile)
    # print("len wholeFile after:", len(wholeFile.columns))
    # print(wholeFile.columns)

    # after pre-processing:
    # X = wholeFile.drop(axis=1, labels="rate")
    # Y = wholeFile["rate"]

    corr = wholeFile.corr()
    # corr.style.background_gradient(cmap='coolwarm', axis=None)

    # Draw correlation matrix:
    # plt.matshow(wholeFile.corr())
    # plt.show()

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

    '''print(type(X_train))
    print(X_train.shape)
    scaler=StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_train=pd.DataFrame(X_train,index= X_train.index)'''

    # region All Features

    print("***** Classifing with All features - Before PCA *****")
    if os.path.exists("DT_Classifier.sav"):
        DT = joblib.load("DT_Classifier.sav")
        acc = DT.score(X_test, Y_test)

    else:
        DT = DT_Classifier(X_train, X_test, Y_train, Y_test)
        joblib.dump(DT, "DT_Classifier.sav")

    if os.path.exists("Naive_Bias.sav"):
        NB = joblib.load("Naive_Bias.sav")
        acc = NB.score(X_test, Y_test)

    else:
       NB = Naive_Bias(X_train, X_test, Y_train, Y_test)
       joblib.dump(NB, "Naive_Bias.sav")

    if os.path.exists("MLP_Classifier.sav"):
        MLP = joblib.load("MLP_Classifier.sav")
        acc = MLP.score(X_test, Y_test)

    else:
        MLP = MLP_Classifier(X_train, X_test, Y_train, Y_test)
        joblib.dump(MLP, "MLP_Classifier.sav")

    if os.path.exists("SVM_One_VS_Rest.sav"):
        OVR = joblib.load("SVM_One_VS_Rest.sav")
        acc = OVR.score(X_test, Y_test)

    else:
         OVR = SVM_One_VS_Rest(X_train, X_test, Y_train, Y_test)
         joblib.dump(OVR, "SVM_One_VS_Rest.sav")

    if os.path.exists("SVM_One_VS_One.sav"):
        OVO = joblib.load("SVM_One_VS_One.sav")
        acc = OVO.score(X_test, Y_test)

    else:
        OVO = SVM_One_VS_One(X_train, X_test, Y_train, Y_test)
        joblib.dump(OVO, "SVM_One_VS_One.sav")

    if os.path.exists("logisticRegCLF.sav"):
        logReg = joblib.load("logisticRegCLF.sav")
        acc = logReg.score(X_test, Y_test)

    else:
        logReg = logisticRegCLF(X_train, X_test, Y_train, Y_test)
        joblib.dump(logReg, "logisticRegCLF.sav")

    if os.path.exists("knnCLF.sav"):
        knn = joblib.load("knnCLF.sav")
        acc = knn.score(X_test, Y_test)

    else:
        knn = knnCLF(X_train, X_test, Y_train, Y_test)
        joblib.dump(knn, "knnCLF.sav")

    if os.path.exists("AdaBoostCLS.sav"):
        adb = joblib.load("AdaBoostCLS.sav")
        acc = adb.score(X_test, Y_test)

    else:
        adb = AdaBoostCLS(X_train, X_test, Y_train, Y_test)
        joblib.dump(adb, "AdaBoostCLS.sav")

    if os.path.exists("GradientBoost.sav"):
        adb = joblib.load("GradientBoost.sav")
        acc = adb.score(X_test, Y_test)

    else:
        gb = GradientBoost(X_train, X_test, Y_train, Y_test)
        joblib.dump(gb, "GradientBoost.sav")
####################
    if os.path.exists('accuracies_list.npy'):
        accuracies_list = np.load('accuracies_list.npy', allow_pickle=True)

    np.save('accuracies_list.npy', accuracies_list)

    if os.path.exists('train_time_list.npy'):
        train_time_list = np.load('train_time_list.npy', allow_pickle=True)

    np.save('train_time_list.npy', train_time_list)

    if os.path.exists('test_time_list.npy'):
        test_time_list = np.load('test_time_list.npy', allow_pickle=True)

    np.save('test_time_list.npy', test_time_list)


    if os.path.exists('classifiers_names.npy'):
        classifiers_names = np.load('classifiers_names.npy', allow_pickle=True)

    np.save('classifiers_names.npy', classifiers_names)



    plot_graphs(accuracies_list, classifiers_names, 'Accuracies')
    plot_graphs(train_time_list, classifiers_names, 'training time')
    plot_graphs(test_time_list, classifiers_names, 'testing time')

    print("__________________________________________________________")
    print("***** Classifing with All features - After PCA *****")
    newX_train, newX_test = PCA_DimnRedAll(X_train, X_test)

    DT_Classifier(newX_train, newX_test, Y_train, Y_test)

    Naive_Bias(newX_train, newX_test, Y_train, Y_test)

    MLP_Classifier(newX_train, newX_test, Y_train, Y_test)

    SVM_One_VS_Rest(newX_train, newX_test, Y_train, Y_test)

    SVM_One_VS_One(newX_train, newX_test, Y_train, Y_test)

    logisticRegCLF(newX_train, newX_test, Y_train, Y_test)

    knnCLF(newX_train, newX_test, Y_train, Y_test)

    AdaBoostCLS(newX_train, newX_test, Y_train, Y_test)

    GradientBoost(newX_train, newX_test, Y_train, Y_test)
    # endregion

    print("*****************************************************")

    # region K-Best
    # region SelectKBest
    k = 6
    # selector = SelectKBest(chi2, k=k)
    selector = SelectKBest(f_classif, k=k)
    selctor_fit_transform = selector.fit_transform(X_train, Y_train)
    top_features = X_train.columns[selector.get_support()]
    print("top_features  from SelectKBest k =", k)
    print(top_features)
    top_corr = wholeFile[top_features].corr()
    # sns.heatmap(top_corr, annot=True)
    # plt.show()
    # endregion
    print("__________________________________________________________")
    print("***** Classifing with K-best features - Before PCA *****")
    # region without PCA

    DT_Classifier(X_train[top_features], X_test[top_features], Y_train, Y_test)

    Naive_Bias(X_train[top_features], X_test[top_features], Y_train, Y_test)

    MLP_Classifier(X_train[top_features], X_test[top_features], Y_train, Y_test)

    SVM_One_VS_Rest(X_train[top_features], X_test[top_features], Y_train, Y_test)

    SVM_One_VS_One(X_train[top_features], X_test[top_features], Y_train, Y_test)

    logisticRegCLF(X_train[top_features], X_test[top_features], Y_train, Y_test)

    knnCLF(X_train[top_features], X_test[top_features], Y_train, Y_test)

    AdaBoostCLS(X_train[top_features], X_test[top_features], Y_train, Y_test)

    GradientBoost(X_train[top_features], X_test[top_features], Y_train, Y_test)
    print("__________________________________________________________")
    # endregion
    print("***** Classifing with K-best features - After PCA *****")

    # region PCA
    newX_train, newX_test = PCA_DimnRedBest(X_train[top_features], X_test[top_features])

    # endregion

    DT_Classifier(newX_train, newX_test, Y_train, Y_test)

    Naive_Bias(newX_train, newX_test, Y_train, Y_test)

    MLP_Classifier(newX_train, newX_test, Y_train, Y_test)

    SVM_One_VS_Rest(newX_train, newX_test, Y_train, Y_test)

    SVM_One_VS_One(newX_train, newX_test, Y_train, Y_test)

    logisticRegCLF(newX_train, newX_test, Y_train, Y_test)

    knnCLF(newX_train, newX_test, Y_train, Y_test)

    AdaBoostCLS(newX_train, newX_test, Y_train, Y_test)

    GradientBoost(newX_train, newX_test, Y_train, Y_test)
    # endregion
    print("__________________________________________________________")
    Testing(X, Y, TrainCopyFile)


main()
wholeFile.to_csv("finalOutput.csv", index=False)
