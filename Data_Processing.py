import pandas as pd 
import arff
from sklearn.preprocessing import OneHotEncoder

# Load Data
file_path = 'Autism-Adult-Data.arff'

with open(file_path, 'r') as f:
    arff_data = arff.load(f)

# Convert to pandas DataFrame
df = pd.DataFrame(arff_data['data'], columns=[attr[0] for attr in arff_data['attributes']])

# Check for missing values
print(df.isnull().sum())

# Drop rows with any null values
df = df.dropna(how='any',axis=0)
print(df.isnull().sum())

# Unique Values for age
a = df['age'].unique()
print(sorted(a))

# Drop Anomaly 
df = df[df.age != 383.0]

cleaned_csv_file_path = 'Cleaned_Autism_Adult_Data.csv'
df.to_csv(cleaned_csv_file_path, index=False)

# Encode Categorical Columns
categorical_columns = ['gender','ethnicity','jundice','austim','contry_of_res','used_app_before','result','age_desc','relation']

# Initialize OneHotEncoder
encoder = OneHotEncoder(sparse_output=False, drop='first')  # drop='first' to avoid multicollinearity

# Fit and transform the categorical columns
encoded_features = encoder.fit_transform(df[categorical_columns])

# Convert to DataFrame
encoded_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out(categorical_columns))

# Concatenate the encoded columns to the original DataFrame
df = df.drop(categorical_columns, axis=1)
df = pd.concat([df, encoded_df], axis=1)
df = df.dropna(how='any',axis=0)
# Display the DataFrame and export to CSV
print(encoded_df)
encoded_csv_file_path = 'Encoded_and_Cleaned_Autism_Adult_Data.csv'
df.to_csv(encoded_csv_file_path, index=False)