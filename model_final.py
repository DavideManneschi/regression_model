import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import statsmodels.formula.api as smf
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import make_scorer
import sys


def import_file():
     file="C:/Users/Davide Manneschi/Documents/datasets/GOT_character_predictions.xlsx"
     read_excel=pd.read_excel(file,na_values=np.nan)
     return read_excel


def dataset_info(file):
     print(file.info())


def display_seaborn(file):

    file_keys=file.keys()
    shape_file=file.shape

    descriptive_statistics_file=file.describe()
    correlation_matrix_raw=file.corr()
    skewness_data=file.skew()


    print("keys_of_the_file:",file_keys)
    print("shape_file:",shape_file)
    print("descriptive_stats:",descriptive_statistics_file)
    print("skeweness_data:",skewness_data)
    print("raw_correlation:",correlation_matrix_raw)












def dataset_summarizing_values(file):
     pd.set_option('display.max_rows', None)
     pd.set_option('display.max_columns', None)




     x_variable,y_variable=file.iloc[:,0:-1],file.iloc[:,-1:]



     #print(x_variable)
     list_keys=[(i) for i in x_variable.keys()]
     missing_variables=file[list_keys].isna().sum()
     print(missing_variables,"percentage==>",missing_variables/x_variable.count())


     #counting_the_variables=file[[]]



     #print(x_variable)
     #print(y_variable)

     #missing_variables_sum=([file[list_keys]==np.NAN])
     #print(missing_variables_sum)


def dataset_inputing(file):



    data_X,data_y = file.iloc[:, 0:-1], file.iloc[:, -1:]



    list_indipendent = []
    final_list_1=[]
    final_list_2=[]
    list_median_1 = []
    uknown=[]
    uknown_float=[]
    uknown_int=[]
    list_objects=[]
    list_float=[]
    list_int=[]



    for i, z in zip(data_X.keys(), data_X.keys().values):

     if i == "age" or i=="dateOfBirth":
         list_indipendent.append(z)


    #print(list_indipendent)

    for y in list_indipendent:

        data_to_evaluate=data_X[y]

        if y=="age":

            for iter in data_to_evaluate:
            #print(iter)

                if iter>=0:
                    final_list_1.append(iter)

                else:
                    iter=np.nan
                    final_list_1.append(iter)

        if y=="dateOfBirth":

            for iter in data_to_evaluate:
            #print(iter)

                if iter>=0:
                    final_list_2.append(iter)


                else:
                    iter=np.nan
                    final_list_2.append(iter)


    data_X.drop(data_X[[i for i in list_indipendent]],axis=1,inplace=True)

    data_X[list_indipendent[1]]=final_list_1
    data_X[list_indipendent[0]]=final_list_2


    for l, k in data_X[[i for i in list_indipendent]].iteritems():
        median= k.median()


        new_dataframe_columns=k.replace(np.nan,median)

        list_median_1.append(new_dataframe_columns)

    final_dataframe_with_median = (pd.DataFrame(list_median_1).transpose())


    data_X.drop(data_X[[i for i in list_indipendent]],axis=1,inplace=True)


    final_data_def = pd.concat([data_X, final_dataframe_with_median], axis=1)



    for f,z in zip( final_data_def.dtypes,final_data_def[[i for i in final_data_def.keys()]]):


        if f=="object":
            list_objects.append(z)


            object_1=final_data_def[z].fillna(-1)
            uknown.append(object_1)

        elif f=="float64":
            list_float.append(z)


            float_1 = final_data_def[z].fillna(-1)
            uknown_float.append(float_1)


        elif f=="int64":
            list_int.append(z)

            int_1 = final_data_def[z].fillna(-1)
            uknown_int.append(int_1)



    dataframe_with_uk=pd.DataFrame(uknown,).transpose()
    dataframe_with_uknown_float=pd.DataFrame(uknown_float).transpose()
    dataframe_with_uknown_int=pd.DataFrame(uknown_int).transpose()


    final_data_def.drop(final_data_def[[ z for z in list_objects]],axis=1,inplace=True)
    final_data_def.drop(final_data_def[[ z for z in list_float]],axis=1,inplace=True)
    final_data_def.drop(final_data_def[[ z for z in list_int]],axis=1,inplace=True)





    very_final_dataframe=pd.concat([final_data_def,dataframe_with_uk,dataframe_with_uknown_float,dataframe_with_uknown_int,data_y]
                                   , axis=1)


    return very_final_dataframe






def find_complete_data(file):

     x_variable, y_variable = file.iloc[:, 1:-1], file.iloc[:, -1:]
     sliced_dataframe=x_variable[["culture","house",]]
     drop=sliced_dataframe.dropna()
     unique_values=drop.groupby("house").count()

     pd.set_option('display.max_rows', None)
     pd.set_option('display.max_columns', None)

     print(unique_values)


def encoding_variables(file):

    data_X, data_y = file.iloc[:, 0:-1], file.iloc[:, -1:]



    dataframe_to_encode=data_X[["title","culture","mother",
                                "father","heir","house",
                                "spouse"]]

    data_X.drop(data_X[["title","culture","mother",
                                "father","heir","house",
                                "spouse"]],axis=1,inplace=True)





    encode_title=pd.get_dummies(dataframe_to_encode["title"])
    encode_culture=pd.get_dummies(dataframe_to_encode["culture"])
    encode_mother=pd.get_dummies(dataframe_to_encode["mother"])
    encode_father=pd.get_dummies(dataframe_to_encode["father"])
    encode_heir=pd.get_dummies(dataframe_to_encode["heir"])
    encode_house=pd.get_dummies(dataframe_to_encode["house"])
    encode_spouse=pd.get_dummies(dataframe_to_encode["spouse"])

    final_encoded_dataframe=pd.concat([data_X,encode_title,encode_culture,encode_mother,
                                       encode_father,encode_heir,encode_house,encode_spouse,data_y],axis=1)




    return  final_encoded_dataframe






def encoding_test(file):



    data_X, data_y = file.iloc[:, 0:-1], file.iloc[:, -1:]
    print(data_X)



    encoded_dataset=pd.get_dummies(data_X)


    encoded_final_dataframe_2=pd.concat([encoded_dataset,data_y],axis=1)

    return encoded_final_dataframe_2



def correlation_analisis(file):
    pd.set_option('display.max_rows', None)




    df_corr = file.corr().round(2)

    print(pd.DataFrame(df_corr['isAlive'].sort_values(ascending=False)))


def sets(file):
    first_set = file[["book4_A_Feast_For_Crows",
                      "numDeadRelations", "popularity", "age",
                      "dateOfBirth", "book1_A_Game_Of_Thrones",
                      "book2_A_Clash_Of_Kings", "book3_A_Storm_Of_Swords",
                      "book1_A_Game_Of_Thrones"]]

    second_set = file[
        ["book4_A_Feast_For_Crows", "culture_Ironborn", "culture_Valyrian", "age", "book2_A_Clash_Of_Kings",
         "book3_A_Storm_Of_Swords",
         "book1_A_Game_Of_Thrones", "numDeadRelations", "popularity", "dateOfBirth", "isNoble", "isAliveHeir",
         "isAliveMother", "isAliveFather",
         "isAliveSpouse"]]

    return first_set


def  k_fold_performance_test(file):
    data_X, data_y = file[["book4_A_Feast_For_Crows","culture_Ironborn","culture_Valyrian",
                          "numDeadRelations","popularity","age","isAliveMother","isAliveHeir","isAliveFather",
                           "dateOfBirth","book1_A_Game_Of_Thrones",
                           "house_House Targaryen","house_House Drumm","house_House Baratheon of Dragonstone",
                           "house_House Baelish","title_Maester","isMarried","title_Prince of Dragonstone"]], file.iloc[:, -1:]

    cross_validation_model=KFold(n_splits=50,random_state=219,shuffle=True)
    first_model=LogisticRegression(solver="lbfgs",C=1,random_state=219)
    second_model=DecisionTreeClassifier(max_depth=8,random_state=219)
    third_model= RandomForestClassifier(n_estimators=1000,criterion='gini',max_depth=8,random_state=219)
    fourth_model=GradientBoostingClassifier(loss="deviance",learning_rate=0.1,n_estimators=100,criterion="friedman_mse",max_depth=8,warm_start=False,random_state=219)

    logistic_result=cross_val_score(first_model,data_X,data_y,cv=cross_validation_model)
    tree_result=cross_val_score(second_model,data_X,data_y,cv=cross_validation_model)
    random_forest_result=cross_val_score(third_model,data_X,data_y,cv=cross_validation_model)
    gradient_boost=cross_val_score(fourth_model,data_X,data_y,cv=cross_validation_model)

    print(f"the result of the logistic_regression is: {logistic_result.mean()}\n"
          f"the result of the tree is: {tree_result.mean()}\n"
          f"the result of the random_forest is: {random_forest_result.mean()}\n"
          f"the result of the gradient_Boosting is: {gradient_boost.mean()}")






                           
                           










def model_evaluation(file):

    print(file["S.No"])




    data_X, data_y = file[["book4_A_Feast_For_Crows",

                           "book1_A_Game_Of_Thrones", "numDeadRelations", "S.No","popularity","dateOfBirth"]],file.iloc[:, -1:]





    x_train,x_test,y_train,y_test=train_test_split(data_X,data_y,test_size=0.1,
                                                   random_state=219,stratify=data_y)




    logistic_regression=LogisticRegression(solver="lbfgs",C=1,random_state=219)
    logistic_regression.fit(x_train,y_train)
    pred=logistic_regression.predict(x_test)


    train_score=(logistic_regression.score(x_train,y_train))
    test_score=(logistic_regression.score(x_test,y_test))

    auc=roc_auc_score(y_test,pred)
    print("auc",auc)
    delta=test_score-train_score

    print("delta=",delta)




    ############################################################### second_model__classification_trees##################




    tree=DecisionTreeClassifier(max_depth=7,random_state=219,criterion="gini",splitter="best",min_samples_leaf=11)
    fitting=tree.fit(x_train,y_train)
    tree_prediction=fitting.predict(x_test)
    tree_auc_score=roc_auc_score(y_test,tree_prediction)

    train_score_tree=tree.score(x_train,y_train)
    test_score_tree=tree.score(x_test,y_test)



    delta_tree=test_score_tree-train_score_tree
    print("delta==",delta_tree.round(2))
    print("tree",tree_auc_score)





    ###### random_forest################

    random_forest = RandomForestClassifier(n_estimators = 1000,
                                     criterion = 'gini',
                                     max_depth = 8,
                                     min_samples_leaf = 4,
                                     bootstrap = True,
                                     warm_start = False,
                                     random_state = 508)


    fitting_random_forest=random_forest.fit(x_train,y_train)
    random_forest_prediction=fitting_random_forest.predict(x_test)
    random_forest_auc=roc_auc_score(y_test,random_forest_prediction)
    print(random_forest_auc)


    ### gradient_boosting########


    gradient_boosting=GradientBoostingClassifier(loss="deviance",
                                                 learning_rate=0.03,
                                                 n_estimators=300,
                                                 criterion="friedman_mse",
                                                 max_depth=5,
                                                 warm_start=False,
                                                 random_state=219)

    fitting_gradient_boosting=gradient_boosting.fit(x_train,y_train)
    predicting_gradient_boosting=fitting_gradient_boosting.predict(x_test)
    auc_score=roc_auc_score(y_test,predicting_gradient_boosting)
    print("gradient",auc_score)
    train_score_gradient = gradient_boosting.score(x_train, y_train)
    test_score_gradient = gradient_boosting.score(x_test, y_test)
    delta_gradient=test_score_gradient-train_score_gradient
    print(delta_gradient)















def tune_modeling_tree(file):
    data_X, data_y = file[["book4_A_Feast_For_Crows","culture_Ironborn","culture_Valyrian","age","book2_A_Clash_Of_Kings","book3_A_Storm_Of_Swords",
                           "book1_A_Game_Of_Thrones","numDeadRelations", "popularity","dateOfBirth","isNoble","isAliveHeir","isAliveMother","isAliveFather",
                           "isAliveSpouse"

                           ]], file.iloc[:, -1:]

    x_train, x_test, y_train, y_test = train_test_split(data_X, data_y, test_size=0.1,
                                                        random_state=219, stratify=data_y)



    range_values=["gini","entropy"]
    slit_range=["best","random"]
    depth_range = np.arange(1, 8, 1)
    leaf_range = np.arange(1, 100, 1)

    grid={ "criterion":range_values,
           "splitter":slit_range,
           "max_depth":depth_range,
           "min_samples_leaf":leaf_range



    }

    tuning_the_tree=DecisionTreeClassifier(random_state=219)

    tuned_tree_cv = RandomizedSearchCV(estimator=tuning_the_tree,
                                       param_distributions=grid,
                                       n_jobs=-1,
                                       cv=20,
                                       n_iter=1000,
                                       random_state=219,
                                       scoring=make_scorer(roc_auc_score,
                                                           needs_threshold=False))

    tuned_tree_cv.fit(x_train, y_train)

    print("Tuned Parameters  :", tuned_tree_cv.best_params_)
    print("Tuned Training AUC:", tuned_tree_cv.best_score_)







def tune_gradient(file):
    data_X, data_y = file[["book4_A_Feast_For_Crows", "culture_Ironborn", "culture_Valyrian", "age",
                           "book2_A_Clash_Of_Kings", "book3_A_Storm_Of_Swords",
                           "book1_A_Game_Of_Thrones", "numDeadRelations", "popularity", "dateOfBirth", "isNoble",
                           "isAliveHeir", "isAliveMother", "isAliveFather",
                           "isAliveSpouse"
                           ]], file.iloc[:, -1:]
    print(data_y)






    parameters={"learning_rate":[0.1,0.01,0.2,0.02,0.3,0.03],
                "n_estimators":[100,200,300,400,500,600,700,1000],
                "criterion":["friedman_mse","mse"],
                "max_depth":[1,2,3,4,5,6,7,8]






    }

    estimator=GradientBoostingClassifier(random_state=219)

    tuning_the_gradient=GridSearchCV(estimator=estimator,param_grid=parameters)
    tuning_the_gradient.fit(data_X,data_y)

    print("Tuned Parameters  :", tuning_the_gradient.best_params_)
    print("Tuned Training AUC:", tuning_the_gradient.best_score_)






def tune_random(file):
    data_X, data_y = file[["book4_A_Feast_For_Crows", "culture_Ironborn", "culture_Valyrian", "age",
                           "book2_A_Clash_Of_Kings", "book3_A_Storm_Of_Swords",
                           "book1_A_Game_Of_Thrones", "numDeadRelations", "popularity", "dateOfBirth", "isNoble",
                           "isAliveHeir", "isAliveMother", "isAliveFather",
                           "isAliveSpouse"
                           ]], file.iloc[:, -1:]


    parameters={



    }






#correlation_analisis(encoding_test(dataset_inputing(import_file())))
#dataset_info(import_file())
model_evaluation(encoding_test(dataset_inputing(import_file())))
#dataset_inputing(import_file())
#k_fold_performance_test(encoding_test(dataset_inputing(import_file())))
#tune_modeling_tree(encoding_test(dataset_inputing(import_file())))
#tune_gradient(encoding_test(dataset_inputing(import_file())))
