import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE
from sklearn.cluster import KMeans
from sklearn.model_selection import StratifiedKFold




df = pd.read_csv('food-inspections.csv')


df_clean = df.dropna(axis=1, how='all')

df_clean = df_clean.drop( ['DBA Name', 'AKA Name', "Address", "City", "Inspection ID", "State", "Location", "Zip"], axis=1)

df_clean = df_clean.dropna(subset=['Risk',  'Inspection Type', "License #", 'Longitude', "Latitude", "Risk"])

df_clean.fillna({'Facility Type':'Unknown'}, inplace=True)

df_clean.fillna({'Violations':"0. No Violation"}, inplace=True)

df_clean = df_clean[df_clean['Results'].str.contains('pass|fail', case=False, na=False)]

df_clean['Results'] = df_clean['Results'].replace({'Pass w/ Conditions' : 'Pass'})


df_clean['Inspection Type'] = df_clean['Inspection Type'].apply(lambda x: x if x in ["Canvass", "License", 'Canvass Re-Inspection', 'Complaint', \
                                                                                     'License Re-Inspection', 'Complaint Re-Inspection'] else 'Other')

df_clean['Inspection Type'] = df_clean['Inspection Type'].replace({"Canvass Re-Inspection": "Canvass", \
                                                                   "License Re-Inspection": "License", \
                                                                   "Complaint Re-Inspection" : "Complaint"})

df_clean['Facility Type'] = df_clean['Facility Type'].apply(lambda x: x if x in ['Restaurant', 'Grocery Store', 'School', \
                                                                                 'Bakery', 'Daycare (2 - 6 Years)' \
                                                                                 'Daycare Above and Under 2 Years' \
                                                                                 'Children\'s Services Facility', 'Long Term Care'] else 'Other')

df_clean = df_clean[df_clean['License #'] != 0]

df_clean['Inspection Date'] = pd.to_datetime(df_clean['Inspection Date'], errors='coerce')

risk_mapping = {'Risk 1 (High)': 3, 'Risk 2 (Medium)': 2, 'Risk 3 (Low)': 1, 'All': 0}
df_clean['Risk_encoded'] = df_clean['Risk'].map(risk_mapping)


label_encoder = LabelEncoder()
label_encoders = {}
df_clean['Facility Type_numeric'] = label_encoder.fit_transform(df_clean['Facility Type'])

facility_type_mapping = dict(zip(label_encoders, label_encoder.transform(label_encoder.classes_)))

df_clean['Inspection Type_numeric'] = label_encoder.fit_transform(df_clean['Inspection Type'])

facility_type_mapping = dict(zip(label_encoders, label_encoder.transform(label_encoder.classes_)))

df_clean = pd.get_dummies(df_clean, columns=['Inspection Type', 'Facility Type'])



df_clean["Results_encoded"] = label_encoder.fit_transform(df_clean["Results"])
label_encoders["Results"] = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))    
df_clean.drop("Results", axis=1, inplace=True)

def extract_violation_codes(violation_str):
    codes = []
    if pd.isnull(violation_str):
        return codes
    for v in violation_str.split('|'):
        v = v.strip()
        if not v:
            continue
        code_part = v.split('.', 1)[0]  
        code_part = code_part.strip()

        codes.append(int(code_part))
    return codes

def count_violations_in_range(violation_codes, lower_bound, upper_bound):
    return sum(lower_bound <= code <= upper_bound for code in violation_codes)

df_clean['violation_codes'] = df_clean['Violations'].apply(extract_violation_codes)

df_clean['non_critical_violations'] = df_clean['violation_codes'].apply(
    lambda codes: count_violations_in_range(codes, 1, 14)
)

df_clean['serious_violations'] = df_clean['violation_codes'].apply(
    lambda codes: count_violations_in_range(codes, 15, 100)
)

df_clean['violation_count'] = df_clean['violation_codes'].apply(len)

df_clean['risk_violations'] = df_clean['Risk_encoded'] * df_clean['violation_count']

df_clean['days_since_last_inspection'] = df_clean.groupby('License #')['Inspection Date'].diff().dt.days
df_clean['days_since_last_inspection'].fillna(df_clean['days_since_last_inspection'].max(), inplace=True)

df_clean['previous_inspection_result'] = df_clean.groupby('License #')['Results_encoded'].shift(1)
df_clean['previous_inspection_result'].fillna(-1, inplace=True)

df_clean['inspection_year'] = df_clean['Inspection Date'].dt.year
df_clean['inspection_month'] = df_clean['Inspection Date'].dt.month
df_clean['day_of_week'] = df_clean['Inspection Date'].dt.dayofweek

df_clean['risk_facility_interaction'] = df_clean['Risk_encoded'] * df_clean['Facility Type']

kmeans = KMeans(n_clusters=10, random_state=42)
df_clean['geo_cluster'] = kmeans.fit_predict(df_clean[['Latitude', 'Longitude']])

avg_violations_by_facility = df_clean.groupby('Facility Type_numeric')[['serious_violations', 'non_critical_violations']].mean()
avg_violations_by_facility['avg_violations'] = avg_violations_by_facility.mean(axis=1)
df_clean['avg_violations_by_facility'] = df_clean['Facility Type_numeric'].map(avg_violations_by_facility['avg_violations'])

avg_risk_by_inspection_type = df_clean.groupby('Inspection Type_numeric')['Risk_encoded'].mean()
df_clean['avg_risk_by_inspection_type'] = df_clean['Inspection Type_numeric'].map(avg_risk_by_inspection_type)

correlation_matrix = df_clean[[ 'non_critical_violations',
    'serious_violations',
    'violation_count',
    'risk_violations',
    'days_since_last_inspection',
    'previous_inspection_result',
    'avg_violations_by_facility',
    'avg_risk_by_inspection_type',
    'inspection_year',
    'inspection_month',
    'day_of_week',
    'geo_cluster',
    'risk_facility_interaction',
    'Latitude',
    'Longitude']].corr()

print(correlation_matrix)
 
#===========================================================
#FILTERING
#===========================================================

def remove_outliers(df, columns):
    for column in columns:
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    return df

numerical_cols = ['non_critical_violations', 'serious_violations', 'days_since_last_inspection']
df_clean = remove_outliers(df_clean, numerical_cols)



#===========================================================
#MODEL CREATION
#===========================================================

feature_cols = [
    'non_critical_violations',
    'serious_violations',
    'violation_count',
    'risk_violations',
    'days_since_last_inspection',
    'previous_inspection_result',
    'avg_violations_by_facility',
    'avg_risk_by_inspection_type',
    'inspection_year',
    'inspection_month',
    'day_of_week',
    'geo_cluster',
    'risk_facility_interaction',
    'Latitude',
    'Longitude'
] + [col for col in df_clean.columns if 'Inspection Type_' in col or 'Facility Type_' in col]

X = df_clean[feature_cols]
y = df_clean['Results_encoded']

X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.25, random_state=42
)


smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)


param_grid = {
    'n_estimators': [200],
    'max_depth': [None],
    'min_samples_split': [2],
    'max_features': ['sqrt'],
    'criterion': ['entropy']
}


cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

grid_search = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_grid=param_grid,
    cv=cv,
    scoring='f1_macro',
    n_jobs=4,
    verbose=2
)

grid_search.fit(X_train_resampled, y_train_resampled)

print('Best parameters found:', grid_search.best_params_)

best_model = grid_search.best_estimator_

y_val_pred = best_model.predict(X_val)

y_test_pred = best_model.predict(X_test)

print("Validation Set Performance after Hyperparameter Tuning:")
print(classification_report(y_val, y_val_pred))

print("Test Set Performance after Hyperparameter Tuning:")
print(classification_report(y_test, y_test_pred))

scores = cross_val_score(best_model, X_temp, y_temp, cv=5, scoring='f1_macro')

print(f'Cross-Validation F1 Scores: {scores}')
print(f'Average F1 Score: {scores.mean()}')

#=============================================================
    
