# Remember to run the following commands!
# git clone https://github.com/Chiesi/ai_ml_deep_learning
# pyenv activate example_rag
# pip install pandas
# pip install matplotlib
# pip install scikit-learn
# pip install jupyterlab
# pip install ipywidgets
# jupyter labextension enable widgetsnbextension
# pip install kagglehub

import kagglehub
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split
# For this first example we'll use Random Forest
from sklearn.ensemble import RandomForestClassifier
# "Measure what is measurable, make measurable what is not"
from sklearn.metrics import accuracy_score

# Download base dataset
path = kagglehub.dataset_download("mathchi/churn-for-bank-customers")
print("Path to dataset files:", path)

# Import into pandas
churn_data = pd.read_csv(path)
churn_data.head()

# Generate statistics about various features 
churn_data.describe()
# Remember pixel histograms?
churn_data.hist(['CreditScore', 'Age', 'Balance'])

# Generate pure numbers from textual representations
# for geographical and gender data
encoder_1 = OrdinalEncoder()
encoder_2 = OrdinalEncoder()
churn_data['Geography_code'] = encoder_1.fit_transform(
    churn_data[['Geography']]
)
churn_data['Gender_code'] = encoder_2.fit_transform(
    churn_data[['Gender']]
)

# Now, one of the greatest feature engineering techniques: drop unused columns
churn_data.drop(
    columns = ['Geography','Gender','RowNumber','Surname'],
    inplace=True
)

# Find correlation amongst features
churn_data.corr()

# Divide between training (80%) and testing (20%).
churn_train, churn_test = train_test_split(
    churn_data, test_size=0.2
)

# We'll use "Exited" as our target column - a churned customer
# is, by definition, one who has discontinued a service
churn_train_X = churn_train.loc[:, churn_train.columns != 'Exited']
churn_train_y = churn_train['Exited']
churn_test_X = churn_test.loc[:, churn_test.columns != 'Exited']
churn_test_y = churn_test['Exited']
bank_churn_clf = RandomForestClassifier(
    max_depth=2, random_state=0
)
bank_churn_clf.fit(churn_train_X, churn_train_y)
churn_prediction_y = bank_churn_clf.predict(churn_test_X)

# Finally, let's determine the accuracy
accuracy_score(churn_test_y, churn_prediction_y)

# Remember to run the file from the browser window opened by jupyter lab, or a 
# local extension!
