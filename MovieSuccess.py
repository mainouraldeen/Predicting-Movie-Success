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


def dataPreprocessing():

    # region Normalization of scaler data
    min_element_budget = wholeFile["budget"].min()
    max_element_budget = wholeFile["budget"].max()

    min_element_popularity = wholeFile["popularity"].min()
    max_element_popularity = wholeFile["popularity"].max()

    min_element_vote_count = wholeFile["vote_count"].min()
    max_element_vote_count = wholeFile["vote_count"].max()

    min_element_vote_average = wholeFile["vote_average"].min()
    max_element_vote_average = wholeFile["vote_average"].max()

    min_element_revenue = wholeFile["revenue"].min()
    max_element_revenue = wholeFile["revenue"].max()

    min_element_run_time = wholeFile["runtime"].min()
    max_element_run_time = wholeFile["runtime"].max()

    wholeFile["budget"] = (wholeFile["budget"] - min_element_budget) / (max_element_budget - min_element_budget)
    wholeFile["vote_average"] = (wholeFile["vote_average"] - min_element_vote_average) / (
            max_element_vote_average - min_element_vote_average)
    wholeFile["vote_count"] = (wholeFile["vote_count"] - min_element_vote_count) / (
            max_element_vote_count - min_element_vote_count)
    wholeFile["popularity"] = (wholeFile["popularity"] - min_element_popularity) / (
            max_element_popularity - min_element_popularity)
    wholeFile["runtime"] = (wholeFile["runtime"] - min_element_run_time) / (
            max_element_run_time - min_element_run_time)
    wholeFile["revenue"] = (wholeFile["revenue"] - min_element_revenue) / (max_element_revenue - min_element_revenue)

    # endregion

    wholeFile["release_date"] = wholeFile["release_date"].astype('datetime64[ns]')

    # region cast pre-proccessing
    i = 0
    counter = 0
    cast_dict = defaultdict(tuple)
    for cell in wholeFile["cast"]:  # cell: group of dictionaries
        cast = json.loads(cell)
        if cast != {}:
            counter += 1
            for list_element in cast:  # kol dict
                for key, value in list_element.items():
                    if key == "id":
                        # each dictionary element has: #occurrencess, its score(each occurrence: add Y[i])
                        if not cast_dict[value]:
                            cast_dict[value] = (0, 0)

                        cast_dict[value] = (cast_dict[value][0] + 1, cast_dict[value][1] + Y[i])

        i += 1

    # assign each keyword a score
    cast_score = defaultdict(float)
    for key, (num_occurr, score) in cast_dict.items():
        cast_score[key] = score / num_occurr

    print("cast_score", cast_score)  # id kol momsl w score bta3o

    j = 0
    for cell in wholeFile["cast"]:  # cell: group of dictionaries
        cell_score = 0
        cast = json.loads(cell)
        if cast != {}:
            for list_element in cast:  # kol dict
                for key, value in list_element.items():
                    if key == "id":
                        cell_score = cell_score + cast_dict[value][1]

        wholeFile["cast"][j] = cell_score
        j += 1

    # normalize cast col
    min_element_cast = wholeFile["cast"].min()
    max_element_cast = wholeFile["cast"].max()
    wholeFile["cast"] = (wholeFile["cast"] - min_element_cast) / (max_element_cast - min_element_cast)
    # print("cast corr",wholeFile["cast"].corr(wholeFile['vote_average']))
    # ana 3ayza el key yb2a id el film w el value list of scores el momsleen gaya mn list of tuples kol wa7d etkrr kam mra w score el aflam ele etkkrr feha
    # endregion

    # region crew pre-proccessing
    k = 0
    counter = 0
    crew_dict = defaultdict(tuple)
    for cell in wholeFile["crew"]:  # cell: group of dictionaries
        crew = json.loads(cell)
        if crew != {}:
            counter += 1
            for list_element in crew:  # kol dict
                for key, value in list_element.items():
                    if key == "id":
                        # each dictionary element has: #occurrencess, its score(each occurrence: add Y[i])
                        if not crew_dict[value]:
                            crew_dict[value] = (0, 0)

                        crew_dict[value] = (crew_dict[value][0] + 1, crew_dict[value][1] + Y[k])
        k += 1

    # assign each keyword a score
    crew_score = defaultdict(float)
    for key, (num_occurr, score) in crew_dict.items():
        crew_score[key] = score / num_occurr

    print("crew_score", crew_score)  # id kol momsl w score bta3o

    l = 0
    for cell in wholeFile["crew"]:  # cell: group of dictionaries
        cell_score = 0
        crew = json.loads(cell)
        if crew != {}:
            for list_element in crew:  # kol dict
                for key, value in list_element.items():
                    if key == "id":
                        cell_score = cell_score + crew_dict[value][1]

        wholeFile["crew"][l] = cell_score
        l += 1

    # normalize crew col
    min_element_crew = wholeFile["crew"].min()
    max_element_crew = wholeFile["crew"].max()
    wholeFile["crew"] = (wholeFile["crew"] - min_element_crew) / (max_element_crew - min_element_crew)
    # print("crew corr",wholeFile["crew"].corr(wholeFile['vote_average']))
    # ana 3ayza el key yb2a id el film w el value list of scores el momsleen gaya mn list of tuples kol wa7d etkrr kam mra w score el aflam ele etkkrr feha
    # endregion

    # region keywords Pre-processing
    # converting keywords into dictionaries
    i = 0
    counter = 0
    keywords_dict = defaultdict(tuple)
    # keywords_dict = {(float, float)}
    # print(">>type", type(keywords_dict))
    for cell in wholeFile["keywords"]:  # cell: group of dictionaries
        # print("i:", i)
        keywords = json.loads(cell)
        if keywords != {}:
            counter += 1
            for list_element in keywords:
                for key, value in list_element.items():
                    if key == "name":
                        # each dictionary element has: #occurrencess, its score(each occurrence: add Y[i])
                        if not keywords_dict[value]:
                            keywords_dict[value] = (0, 0)

                        keywords_dict[value] = (keywords_dict[value][0] + 1, keywords_dict[value][1] + Y[i])
        else:
            print("at i:", i, " EMPTY CELL!!!!!!")
            break
        i += 1

    # print("---------------------------")

    print("#non-empty cells:", counter)

    # assign each keyword a score
    keywords_score = defaultdict(float)
    for key, (num_occurr, score) in keywords_dict.items():
        keywords_score[key] = score / num_occurr

    print(keywords_score)
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

    print("len wholeFile before dropna", len(wholeFile))
    wholeFile.replace(['', ' ', [[]], [], None, {}], np.nan, inplace=True)  # 3shan .drop b t-drop no.nan bs
    wholeFile.dropna(thresh=1, inplace=True)

    print("len wholeFile after dropna", len(wholeFile))

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
