import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Read the CSV file
df = pd.read_csv('heart.csv')

# Display basic information about the dataset
print("Dataset Info:")
print(df.info())

# Display the first few rows
print("\nFirst few rows:")
print(df.head())

# Get basic statistics of numerical columns
print("\nBasic statistics:")
print(df.describe())

# Check for missing values
print("\nMissing values:")
print(df.isnull().sum())

# Display unique values in categorical columns
categorical_cols = ['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope']
print("\nUnique values in categorical columns:")
for col in categorical_cols:
    print(f"\n{col}:", df[col].unique())

# Separate features (X) and target variable (y)
X = df.drop('HeartDisease', axis=1)  # Assuming 'HeartDisease' is your target column
y = df['HeartDisease']

# Encode categorical variables
categorical_cols = ['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope']
le = LabelEncoder()
for col in categorical_cols:
    X[col] = le.fit_transform(X[col])

# Split the data into training and testing sets
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\nData split shapes:")
print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"y_test shape: {y_test.shape}")