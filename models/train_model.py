import numpy as np
import pandas as pd 
from sklearn.model_selection import train_test_split,GridSearchCV
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import seaborn as sns 
import itertools
from sklearn.metrics import accuracy_score, confusion_matrix
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier




df=pd.read_pickle("../data/interim/03_data_features.pkl")

df_train=df.drop(["participant","category","set"],axis=1)

X=df_train.drop("label",axis=1)
Y=df_train["label"]

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.25,random_state=42,stratify=Y)





basic_features=["acc_x","acc_y","acc_z","gyr_x","gyr_y","gyr_z"]
square_features=["acc_r","gyr_r"]
pca_features=["pca_1","pca_2","pca_3"]
time_features=[f for f in df_train.columns if "_temp_" in f ]
frequency_features=[f for f in df_train.columns if (("_freq" in f) or ("_pse" in f)) ]
cluster_features=["cluster"]

print("basic_features",len(basic_features))
print("time_features",len(time_features))
print("frequency_features",len(frequency_features))


feature_set_1=basic_features
feature_set_2=list(set(basic_features+square_features+pca_features))
feature_set_3=list(set(feature_set_2+time_features))
feature_set_4=list(set(feature_set_3+frequency_features+cluster_features))

probable_feature_sets=[
    feature_set_1,
    feature_set_2,
     feature_set_3,
     feature_set_4
    
]



iter=1
combined_score=pd.DataFrame(columns=["feature set","model","score","best_param"])


for featureset in probable_feature_sets:
    print("feature set " ,iter)
    X_train_current=X_train[featureset]
    
    
    #####################################
    #                                   #
    #         Random forest             #
    #                                   #
    #####################################
    
    param_grid_rf = {
    'n_estimators': [50, 100, 200],
    
    'max_depth': [None, 10, 30,50,100],
    'criterion': ['gini', 'entropy']
     }
    rf = RandomForestClassifier(random_state=42)

    # Initialize GridSearchCV
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid_rf, cv=5, n_jobs=-1, verbose=2)

    # Fit the model
    grid_search.fit(X_train_current, Y_train)

    # Get the best parameters and the best estimator
    best_params = grid_search.best_params_
    best_rf = grid_search.best_estimator_

    print(f"Best Params random forest {best_params}")

    # Predict on the test set
    y_pred = best_rf.predict(X_test[featureset])

    # Evaluate the model
    accuracy_rf = accuracy_score(Y_test, y_pred)
    print("random forest accuracy" ,accuracy_rf)
    combined_score.loc[len(combined_score.index)]=['featureset'+str(iter),"random forest",accuracy_rf,best_params]
 
   
    
    #####################################
    #                                   #
    #          NEURAL NETWORK           #
    #                                   #
    #####################################


    
    model = models.Sequential([
        layers.Dense(128, activation='relu',input_dim=X_train_current.shape[1]), 
        layers.Dropout(0.2), 
        layers.Dense(64, activation='relu') ,
        layers.Dense(6, activation='softmax')  

    ])

    model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
    
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(Y_train)
    y_one_hot = tf.keras.utils.to_categorical(y_encoded, num_classes=6)
   
    model.fit(X_train_current, y_one_hot, epochs=40)
    predictions = model.predict(X_test[featureset])
    
    y_encoded_test = label_encoder.transform(Y_test)
    y_one_hot_test = tf.keras.utils.to_categorical(y_encoded_test, num_classes=6)
   
    test_loss, test_acc = model.evaluate(X_test[featureset], y_one_hot_test, verbose=2)
    print('neural network accuracy:', test_acc)
    
    combined_score.loc[len(combined_score.index)]=['featureset'+str(iter),"ann",test_acc, np.nan]
    
  
    
    #####################################
    #                                   #
    #      Support vector machine       #
    #                                   #
    #####################################  
    
    
    svm_model = SVC()
    
    
    param_grid_SVM = {
        'C': [ 1, 10, 100,1000],
        'gamma': [1, 0.1, 0.01, 0.001,0.0001],
        'kernel': ['rbf']
    }
   
    grid_search = GridSearchCV(estimator=svm_model, param_grid=param_grid_SVM, cv=5, n_jobs=-1, verbose=2)
    grid_search.fit(X_train_current,Y_train)
    
    print(f"Best paramS FOR SVM: {grid_search.best_params_}")
    
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test[featureset])
    accuracy_SVM = accuracy_score(Y_test, y_pred)
    print("accuracy for svm")
    combined_score.loc[len(combined_score.index)]=['featureset'+str(iter),"SVM",accuracy_SVM,grid_search.best_params_ ]
    
    
    #####################################
    #                                   #
    #       Naive Bayes Classifier      #
    #                                   #
    #####################################

    
    nb_model = GaussianNB()
    nb_model.fit(X_train_current, Y_train)

    
    y_pred = nb_model.predict(X_test[featureset])
    accuracy_nb = accuracy_score(Y_test, y_pred)
    print(f"Naive Bayes accuracy: {accuracy_nb}")
 
    combined_score.loc[len(combined_score.index)]=['featureset'+str(iter),"NB",accuracy_nb,np.nan ]
    
    #####################################
    #                                   #
    #         XGBoost Classifier        #
    #                                   #
    #####################################

    xgb_model = XGBClassifier( eval_metric='mlogloss',)

    # Define the hyperparameter grid
    param_grid_XGB = {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0]
    }
    label_encoder2 = LabelEncoder()
    y_encoded = label_encoder2.fit_transform(Y_train)
    y_encoded_test = label_encoder2.transform(Y_test)


    grid_search = GridSearchCV(xgb_model, param_grid_XGB, cv=5, verbose=2, n_jobs=-1)
    grid_search.fit(X_train_current, y_encoded)

    print(f"Best parameters for XGBOOST: {grid_search.best_params_}")

    best_model_xg = grid_search.best_estimator_
    y_pred = best_model_xg.predict(X_test[featureset])
    accuracy_xgb = accuracy_score(y_encoded_test, y_pred)
    print(f"XGBoost accuracy: {accuracy_xgb}")

    combined_score.loc[len(combined_score.index)]=['featureset'+str(iter),"XGboost",accuracy_xgb,best_model_xg ]

    iter+=1
    
    

combined_score.sort_values(by='score',ascending=False,inplace=True)

combined_score.to_pickle("../data/interim/04_All_model_summary.pkl")       
            
            

#now splitting based on partiicpants


participant_df=df_train
participant_df["participant"]=df["participant"]

X_train_new=participant_df[participant_df["participant"]!="A"].drop("label",axis=1)
Y_train_new=participant_df[participant_df["participant"]!="A"]["label"]

X_test_new=participant_df[participant_df["participant"]=="A"].drop("label",axis=1)
Y_test_new=participant_df[participant_df["participant"]=="A"]["label"]


X_train_new=X_train_new.drop(["participant"],axis=1)
X_test_new=X_test_new.drop(["participant"],axis=1)          
          
