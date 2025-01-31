# Import statements
import os
import joblib
import matplotlib
import numpy as np
import pandas as pd
from sklearn import neighbors
from sklearn import svm
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import

matplotlib.use('Agg')
pd.set_option('display.max_columns', 10)
pd.set_option('display.max_rows', 100)
np.set_printoptions(suppress=True, threshold=5000, edgeitems=10)

# bring in the data
full_path = os.path.join(os.getcwd(), 'streamingdata.json')
my_df = pd.read_json(full_path)

# Checking if data was loaded properly
print(my_df.sample(5))
print(my_df.info())

# Checking for duplicates
print(len(my_df['cust_id'].unique()))
print(len(my_df['cust_id']))
my_df = my_df.drop_duplicates(subset=['cust_id'])
my_df.info()
print(my_df.nunique())
print(my_df.isnull().sum())

# Dummy coding the gender field
print(my_df['gender'].unique())
my_df['gender'].fillna('Unknown', inplace=True)
dummies = pd.get_dummies(my_df['gender'], prefix='gender', dummy_na=True)
my_df = pd.concat([my_df, dummies], axis=1)

# creating the age variable
my_df['extract_date'] = pd.to_datetime(my_df['extract_date'], format='%Y%m%d')
my_df['dob'] = pd.to_datetime(my_df['dob'], format='%Y%m%d')
# Age of current period in days.
my_df['age'] = my_df['extract_date'] - my_df['dob']
my_df['age'] = pd.to_numeric(my_df['age'].dt.days, downcast='integer') - 30
# Converting age from days to years
my_df['age'] = np.round((my_df['age'] / 365.2425), 2)
print(my_df["age"].min())
print(my_df["age"].max())

# Creating age groups
bins = list(range(0, 90, 10))
print(bins)
labels = ['0-9', '10-19', '20-29', '30-39', '40-49', '50-59', '60-69', '70-79']
my_df['age_groups'] = pd.cut(my_df['age'], bins=bins, labels=labels, right=False, include_lowest=True)
print(my_df['age_groups'].unique())
my_df['age_groups'] = my_df['age_groups'].cat.add_categories('Unknown')
my_df['age_groups'].fillna('Unknown', inplace=True)
print(my_df.sample(5))

# Dummy coding age groups
age_dummies = pd.get_dummies(my_df['age_groups'], prefix='age', dummy_na=True)
my_df = pd.concat([my_df, age_dummies], axis=1)
print(my_df.sample(5))

# drop null columns
print(my_df.isnull().sum())
my_df = my_df.dropna()
print(my_df.isnull().sum())
print(my_df.info())

# Setting up features
my_features = my_df[['delta1_friend_cnt', 'delta1_avg_friend_age', 'delta1_female_friend_cnt',
                     'delta1_male_friend_cnt', 'delta1_friend_country_cnt', 'delta1_subscriber_friend_cnt',
                     'delta1_songsListenedIndependent', 'delta1_songsListenedPopular',
                     'delta1_lovedTracksIndependent', 'delta1_lovedTracksPopular', 'delta1_posts', 'delta1_playlists',
                     'delta1_shouts', 'delta1_avgActiveMinutesPerDay', 'delta1_country', 'delta1_tenure',
                     'delta1_profilePicture', 'delta1_paid_subscriber', 'age_Unknown', 'age_0-9', 'age_10-19',
                     'age_20-29', 'age_30-39', 'age_40-49', 'age_50-59', 'age_60-69', 'age_70-79', 'gender_Unknown',
                     'gender_M', 'gender_F']]
my_features.values
targets = my_df['paid_subscriber'].values.flatten()

# Scaling data using standard scaler
sc = StandardScaler().fit(my_features)
features_sc = sc.transform(my_features)

# Using PCA to reduce features
pca = PCA(random_state=43543, n_components=5)
pca_fitted = pca.fit(features_sc)
features_reduced = pca_fitted.fit_transform(features_sc)

# Create training and testing variables
f_train, f_test, t_train, t_test = train_test_split(features_reduced, targets, test_size=0.25,
                                                    random_state=77)

# Decision Tree Model:
# Finding which hyper-parameters to use using GridsearchCV
hyper_parameters = {"criterion": ['gini', 'entropy'], "max_depth": [1, 5, 10], "min_samples_split": [10, 20, 30],
                    "min_samples_leaf": [5, 10, 15],
                    "ccp_alpha": [0, 0.1, 0.01]}

grid_search = GridSearchCV(DecisionTreeClassifier(), hyper_parameters)
gs_dt_model = grid_search.fit(f_train, t_train)
print(gs_dt_model.best_params_)

gs_dt_model_score_test = gs_dt_model.score(f_test, t_test)
gs_dt_model_score_train = gs_dt_model.score(f_train, t_train)
print(f'The decision tree model with gridsearch had an accuracy score of {gs_dt_model_score_test:,.2f} in'
      f' the unseen testing data. This model had a score of {gs_dt_model_score_train:,.2f} in the training data, '
      f'so the variance was {(gs_dt_model_score_train-gs_dt_model_score_test):,.2f}')

# Random Forrest Model:
# Finding which hyper-parameters to use using RandomizedSearchCV
hyper_params = {'max_samples': [800, 1200],
                'max_features': ['sqrt', 'log2'],
                'criterion': ['gini', 'entropy'],
                'max_depth': [5, 15],
                'min_samples_leaf': [5, 10],
                'min_samples_split': [15, 20]}

rf_random = RandomizedSearchCV(RandomForestClassifier(random_state=42,
                                                      n_estimators=50,
                                                      oob_score=True,
                                                      bootstrap=True),
                               param_distributions=hyper_params,
                               n_iter=10, cv=2, random_state=42, scoring='accuracy')

#Train and Evaluate the model
rf_rscv_model = rf_random.fit(f_train, t_train)
rf_rscv_model_score_test = rf_rscv_model.score(f_test, t_test)
rf_rscv_model_score_train = rf_rscv_model.score(f_train, t_train)
print(f'The random forest model with randomized search cv had an accuracy score of {rf_rscv_model_score_test:,.2f} in'
      f' the unseen testing data. This model had a score of {rf_rscv_model_score_train:,.2f} in the testing data, '
      f'so the variance was {(rf_rscv_model_score_train-rf_rscv_model_score_test):,.2f}')

# KNN Model:
num_neighbors = 37
knn = neighbors.KNeighborsClassifier(n_neighbors=num_neighbors, metric='euclidean', weights='uniform')
knn_model = knn.fit(f_train, t_train)

score_test = 100 * knn_model.score(f_test, t_test)
print(f'KNN ({num_neighbors} neighbors) prediction accuracy with test data = {score_test:.1f}%')

score_train = 100 * knn_model.score(f_train, t_train)
print(f'KNN ({num_neighbors} neighbors) prediction accuracy with training data '
      f'to evaluate potential over-fitting = {score_train:.1f}%')

# SGD Model:
sgd_algorithm = SGDRegressor(fit_intercept=True, loss='squared_error',
                             learning_rate='invscaling', eta0=0.0001, penalty='l2',
                             max_iter=10000)
print(sgd_algorithm.get_params())
sgd_model = sgd_algorithm.fit(f_train, t_train)

test_score = 100.0 * sgd_model.score(f_test, t_test)
print(f'SGDRegressor model score with the unseen testing data is {test_score:5.4f}%')

# Overfitting?
train_score = 100.0 * sgd_model.score(f_train, t_train)
print(f'SGDRegressor model with the training data is {train_score:5.4f}%')
print(f'The difference between training and testing is {train_score-test_score:.4f} percentage points')

# SVM Model
k = 'rbf'
svm_algorithm = svm.SVC(C=1, gamma=0.5, kernel=k, probability=True)

svm_model = svm_algorithm.fit(f_train, t_train)
predicted = svm_model.predict(f_test)
score = 100.0 * accuracy_score(t_test, predicted)
print(f'SVM using {k} kernel had an accuracy score of {score:4.1f}% with the unseen testing data')

predicted = svm_model.predict(f_train)
score = 100.0 * accuracy_score(t_train, predicted)
print(f'SVM using {k} kernel had an accuracy score of {score:4.1f}% with the training data to determine '
      f'potential overfitting.')

# Saving best model (x) as a pkl file
filename = 'svm-model.pkl'
with open(filename, 'wb') as fout:
    joblib.dump(svm_model, fout)
with open(filename, 'rb') as fin:
    svm_from_pkl = joblib.load(fin)

# Importing data from implementaion_sample.txt
imp_filename = os.path.join(os.getcwd(), 'Mattson Music', 'implementation_sample.txt')
imp_df = pd.read_csv(imp_filename, delimiter='|')
imp_df.info()

# Dummy coding gender
print(imp_df['gender'].unique())
imp_df['gender'].fillna('Unknown', inplace=True)
dummies = pd.get_dummies(imp_df['gender'], prefix='gender', dummy_na=True)
imp_df = pd.concat([imp_df, dummies], axis=1)
imp_df.info()

# Creating age field
imp_df['extract_date'] = pd.to_datetime(imp_df['extract_date'], format='%Y%m%d')
imp_df['dob'] = pd.to_datetime(imp_df['dob'], format='%Y%m%d')
# Age of current period in days.
imp_df['age'] = imp_df['extract_date'] - imp_df['dob']
imp_df['age'] = pd.to_numeric(imp_df['age'].dt.days, downcast='integer') - 30
# Converting age from days to years (mindful of leap years)
imp_df['age'] = np.round((imp_df['age'] / 365.2425), 2)
print(imp_df["age"].min())
print(imp_df["age"].max())

# Creating age groups
bins = list(range(10, 100, 10))
print(bins)
labels = ['10-19', '20-29', '30-39', '40-49', '50-59', '60-69', '70-79', '80-89']
imp_df['age_groups'] = pd.cut(imp_df['age'], bins=bins, labels=labels, right=False, include_lowest=True)
print(imp_df['age_groups'].unique())
imp_df['age_groups'] = imp_df['age_groups'].cat.add_categories('Unknown')
imp_df['age_groups'].fillna('Unknown', inplace=True)
print(imp_df.sample(5))

# Dummy coding age groups
age_dummies = pd.get_dummies(imp_df['age_groups'], prefix='age', dummy_na=True)
imp_df = pd.concat([imp_df, age_dummies], axis=1)
print(imp_df.sample(5))

# Must drop the nulls to run the PCA
print(imp_df.isnull().sum())
imp_df = imp_df.dropna()
print(imp_df.isnull().sum())
print(imp_df.info())

# Setting up features and target
imp_features = imp_df[['friend_cnt', 'avg_friend_age', 'female_friend_cnt', 'male_friend_cnt',
                       'friend_country_cnt', 'subscriber_friend_cnt', 'songsListenedIndependent',
                       'songsListenedPopular', 'lovedTracksIndependent', 'lovedTracksPopular', 'posts',
                       'playlists', 'shouts', 'avgActiveMinutesPerDay', 'country', 'tenure', 'profilePicture',
                       'paid_subscriber', 'age_Unknown', 'age_10-19', 'age_20-29', 'age_30-39', 'age_40-49',
                       'age_50-59', 'age_60-69', 'age_70-79', 'age_80-89', 'gender_Unknown', 'gender_M',
                       'gender_F']].values
targets = imp_df['paid_subscriber'].values.flatten()

# Applying standard scaler
features = StandardScaler().fit(imp_features)
features_sc = features.transform(imp_features)

# If the standard scaler worked, our data should have a mean of 0 and a standard deviation of 1.
print(np.mean(features_sc, axis=0))
print(np.std(features_sc, axis=0))

# Run a PCA to reduce features
imp_pca = PCA(random_state=43543, n_components=5)
imp_pca_fitted = pca.fit(features_sc)
imp_features_reduced = pca_fitted.fit_transform(features_sc)

# Use the knn model to make predictions about subscribers
model_predictions = svm_from_pkl.predict(imp_features_reduced)
probabilities = svm_from_pkl.predict_proba(imp_features_reduced)
imp_df['predicted_subscriber_next_period'] = pd.Series(model_predictions, index=imp_df.index)
imp_df['probability_no_subscription'] = pd.Series(probabilities[:, 0], index=imp_df.index)
imp_df['probability_subscription'] = pd.Series(probabilities[:, 1], index=imp_df.index)
print(imp_df.sample(20))
print(imp_df.isnull().sum())

# Clean up the dataframe a bit
imp_df['predicted_subscriber_next_period'] = imp_df['predicted_subscriber_next_period'].map({1: 'Yes',
                                                                                             0: 'No'})
imp_df = imp_df.drop(columns=['gender_F', 'gender_M', 'gender_Unknown', 'gender_nan', 'age', 'age_groups',
                              'age_10-19', 'age_20-29', 'age_30-39', 'age_40-49', 'age_50-59', 'age_60-69',
                              'age_70-79', 'age_80-89', 'age_Unknown', 'age_nan'])

# Export to excel and apply some conditional formatting
full_path = os.path.join(os.getcwd(), "Mattson Music", "Predictions.xlsx")
with pd.ExcelWriter(full_path, engine='xlsxwriter') as writer:
    imp_df.to_excel(writer, sheet_name='Sheet1', index=False)
    workbook = writer.book
    worksheet = writer.sheets['Sheet1']
    num_rows = len(imp_df.index)
    format_green = workbook.add_format({'bg_color': '#C6EFCE', 'font_color': '#006100'})
    worksheet.conditional_format('X2:X{}'.format(num_rows + 1), {
        'type': 'cell',
        'criteria': '>',
        'value': 0.5,
        'format': format_green
    })
    worksheet.conditional_format('Y2:Y{}'.format(num_rows + 1), {
        'type': 'cell',
        'criteria': '>',
        'value': 0.5,
        'format': format_green
    })












