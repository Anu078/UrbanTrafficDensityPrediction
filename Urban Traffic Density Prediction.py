#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np

# Load the dataset
file_path = r"C:\Users\DELL\Downloads\Compressed\archive_3\traffic.csv"  # Update the path if needed
df = pd.read_csv(file_path, parse_dates=['DateTime'])

# Display first few rows
df.head()


# In[2]:


# Extracting useful components from DateTime
df['Hour'] = df['DateTime'].dt.hour
df['DayOfWeek'] = df['DateTime'].dt.dayofweek  # Monday=0, Sunday=6
df['Month'] = df['DateTime'].dt.month
df['IsWeekend'] = df['DayOfWeek'].apply(lambda x: 1 if x >= 5 else 0)  # Sat/Sun = 1

# Assigning time slots
def assign_time_slot(hour):
    if 0 <= hour < 6:
        return 1  # Late Night
    elif 6 <= hour < 9:
        return 2  # Morning Rush
    elif 9 <= hour < 12:
        return 3  # Late Morning
    elif 12 <= hour < 15:
        return 4  # Afternoon
    elif 15 <= hour < 18:
        return 5  # Evening Rush
    elif 18 <= hour < 21:
        return 6  # Night Traffic
    else:
        return 7  # Late Evening

df['TimeSlot'] = df['Hour'].apply(assign_time_slot)

# Drop original DateTime column (not needed anymore)
df.drop(columns=['DateTime'], inplace=True)

# Display updated DataFrame
df.head()


# In[3]:


# Drop ID column if present
if 'ID' in df.columns:
    df.drop(columns=['ID'], inplace=True)

df.head()


# In[4]:


# Example: Add a 'Weather' column with random data


# Random weather data (for demonstration)
weather_conditions = ['Sunny', 'Rainy', 'Cloudy']
df['Weather'] = np.random.choice(weather_conditions, size=len(df))

# Verify
print(df.head())


# In[5]:


# Convert categorical variables to numeric (One-Hot Encoding)
df = pd.get_dummies(df, columns=['Weather'], drop_first=True)

# Display updated DataFrame
df.head()


# In[6]:


# Check column names to verify the correct target variable name
print(df.columns)


# In[7]:


# Group by Junction and find the max vehicles recorded at each
max_vehicles_per_junction = df.groupby('Junction')['Vehicles'].transform('max')

# Calculate TrafficDensity as a ratio
df['TrafficDensity'] = df['Vehicles'] / max_vehicles_per_junction

# Verify the column
print(df[['Junction', 'Vehicles', 'TrafficDensity']].head())


# In[8]:


print(df.groupby('Junction')['Vehicles'].max())


# In[9]:


import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(8, 5))
sns.histplot(df['TrafficDensity'], bins=20, kde=True)
plt.xlabel("Traffic Density (Normalized)")
plt.ylabel("Frequency")
plt.title("Distribution of Estimated Traffic Density")
plt.show()


# In[10]:


df['TrafficDensity'].describe()


# In[11]:


from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np


# In[12]:


# Define target variable
target = 'TrafficDensity'
X = df.drop(columns=[target])  # Features
y = df[target]  # Target variable

# Splitting the dataset into 80% training and 20% testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)


# In[13]:


# Initialize the XGBoost Regressor model
xgb_model = XGBRegressor(
    n_estimators=50,  # Reduce number of trees (default was 100)
    learning_rate=0.05,  # Reduce learning rate for better generalization
    max_depth=4,  # Limit tree depth (default is usually higher)
    subsample=0.8,  # Use 80% of training data for each boosting round
    colsample_bytree=0.8,  # Use only 80% of features per tree
    random_state=42
)
# Train the model
xgb_model.fit(X_train, y_train)


# In[14]:


# Make predictions on the test set
y_pred = xgb_model.predict(X_test)

# Calculate evaluation metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

# Print the evaluation results
print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
print(f"R-squared (R²): {r2:.4f}")


# In[15]:


print(X_train.corrwith(y_train))


# In[16]:


print(df['TrafficDensity'].describe())


# In[17]:


df['Lag_Vehicles_1H'] = df.groupby('Junction')['Vehicles'].shift(1)
df['Lag_Vehicles_3H'] = df.groupby('Junction')['Vehicles'].shift(3)
df['Lag_Vehicles_6H'] = df.groupby('Junction')['Vehicles'].shift(6)


# In[18]:


df['MovingAvg_2H'] = df.groupby('Junction')['Vehicles'].rolling(2).mean().reset_index(0, drop=True)
df['MovingAvg_6H'] = df.groupby('Junction')['Vehicles'].rolling(6).mean().reset_index(0, drop=True)


# In[19]:


df['Vehicles_TimeSlot'] = df['Vehicles'] * df['TimeSlot']
df['Vehicles_Weekend'] = df['Vehicles'] * df['IsWeekend']


# In[20]:


df.head()


# In[21]:


df.info()


# In[22]:


df[['Lag_Vehicles_1H', 'Lag_Vehicles_3H', 'Lag_Vehicles_6H']] = df.groupby('Junction')[
    ['Lag_Vehicles_1H', 'Lag_Vehicles_3H', 'Lag_Vehicles_6H']
].fillna(method='bfill')


# In[23]:


df[['MovingAvg_2H', 'MovingAvg_6H']] = df.groupby('Junction')[
    ['MovingAvg_2H', 'MovingAvg_6H']
].fillna(method='ffill')


# In[24]:


df.fillna(df.median(), inplace=True)


# In[25]:


df.isnull().sum()


# In[26]:


from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
num_cols = ['Vehicles', 'Lag_Vehicles_1H', 'Lag_Vehicles_3H', 'Lag_Vehicles_6H', 
            'MovingAvg_2H', 'MovingAvg_6H', 'Vehicles_TimeSlot', 'Vehicles_Weekend', 'TrafficDensity']

df[num_cols] = scaler.fit_transform(df[num_cols])


# In[27]:


import xgboost as xgb
import matplotlib.pyplot as plt

X = df.drop(columns=['TrafficDensity'])  # Features
y = df['TrafficDensity']  # Target

model = xgb.XGBRegressor()
model.fit(X, y)

# Plot feature importance
xgb.plot_importance(model)
plt.show()


# In[28]:


drop_cols = ['Weather_Rainy', 'Weather_Sunny', 'Month', 'DayOfWeek']
df = df.drop(columns=drop_cols)


# In[29]:


print(df.corr()['TrafficDensity'].sort_values(ascending=False))


# In[30]:


df = df.drop(columns=['Vehicles_Weekend'])


# In[31]:


from sklearn.model_selection import train_test_split

# Define features (drop 'TrafficDensity' as it's the target)
X = df.drop(columns=['TrafficDensity'])
y = df['TrafficDensity']

# Split into training and testing sets (80-20 split)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[32]:


import xgboost as xgb

# Initialize XGBoost Regressor
xgb_model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42)

# Train the model
xgb_model.fit(X_train, y_train)


# In[33]:


from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

# Predictions
y_pred = xgb_model.predict(X_test)

# Evaluation Metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

# Print results
print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
print(f"R-squared (R²): {r2:.4f}")


# In[34]:


print("Check if target variable is in features:", 'TrafficDensity' in X_train.columns)


# In[35]:


xgb_model = xgb.XGBRegressor(
    objective='reg:squarederror',
    n_estimators=100,        # Reduce trees if needed
    learning_rate=0.1,       # Keep it moderate
    max_depth=4,             # Reduce complexity
    subsample=0.8,           # Use only 80% of training data per tree
    colsample_bytree=0.8,    # Use 80% of features per tree
    reg_alpha=0.1,           # L1 Regularization
    reg_lambda=0.1,          # L2 Regularization
    random_state=42
)

# Retrain the model
xgb_model.fit(X_train, y_train)


# In[36]:


y_pred = xgb_model.predict(X_test)

# Recalculate Metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
print(f"R-squared (R²): {r2:.4f}")


# In[37]:


from sklearn.model_selection import cross_val_score

cv_scores = cross_val_score(xgb_model, X_train, y_train, cv=5, scoring='r2')
print(f"Cross-Validation R² Scores: {cv_scores}")
print(f"Mean CV R²: {cv_scores.mean():.4f}")


# In[38]:


df.head()


# In[39]:


from statsmodels.stats.outliers_influence import variance_inflation_factor
import pandas as pd

X = X_train.copy()  # Use training data only
vif_data = pd.DataFrame()
vif_data["Feature"] = X.columns
vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(len(X.columns))]

print(vif_data.sort_values(by="VIF", ascending=False))


# In[40]:


# Drop features with extremely high VIF
X_filtered = X.drop(columns=['MovingAvg_2H', 'Vehicles', 'Lag_Vehicles_1H'])

# Function to recalculate VIF
def calculate_vif(df):
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    import pandas as pd

    vif_data = pd.DataFrame()
    vif_data["Feature"] = df.columns
    vif_data["VIF"] = [variance_inflation_factor(df.values, i) for i in range(df.shape[1])]
    return vif_data

# Recalculate VIF after dropping
vif_df_new = calculate_vif(X_filtered)
print(vif_df_new)


# In[41]:


print(X_filtered.columns)


# In[42]:


X_filtered['Hour_sin'] = np.sin(2 * np.pi * X_filtered['Hour'] / 24)
X_filtered['Hour_cos'] = np.cos(2 * np.pi * X_filtered['Hour'] / 24)


# In[43]:


X_filtered = X_filtered.drop(columns=['Hour'])


# In[45]:


'''X_filtered = X_filtered.drop(columns=['TimeSlot', 'MovingAvg_6H', 'Lag_Vehicles_3H', 'Lag_Vehicles_6H'])'''


# In[46]:


print(X_filtered.columns)


# In[47]:


from statsmodels.stats.outliers_influence import variance_inflation_factor

# Compute VIF for each feature
vif_data = pd.DataFrame()
vif_data["Feature"] = X_filtered.columns
vif_data["VIF"] = [variance_inflation_factor(X_filtered.values, i) for i in range(X_filtered.shape[1])]

print(vif_data)


# In[49]:


print("Features in X_train:", X_train.columns.tolist())
print("Features in X_test:", X_test.columns.tolist())


# In[51]:


print("Current X_train columns:", X_train.columns.tolist())
print("Current X_test columns:", X_test.columns.tolist())


# In[52]:


import numpy as np

# Ensure 'Hour' exists before creating sin/cos transformations
if 'Hour' in X_train.columns:
    X_train['Hour_sin'] = np.sin(2 * np.pi * X_train['Hour'] / 24)
    X_train['Hour_cos'] = np.cos(2 * np.pi * X_train['Hour'] / 24)
    X_train = X_train.drop(columns=['Hour'])  # Drop original Hour

if 'Hour' in X_test.columns:
    X_test['Hour_sin'] = np.sin(2 * np.pi * X_test['Hour'] / 24)
    X_test['Hour_cos'] = np.cos(2 * np.pi * X_test['Hour'] / 24)
    X_test = X_test.drop(columns=['Hour'])  # Drop original Hour


# In[53]:


print("Updated X_train columns:", X_train.columns.tolist())
print("Updated X_test columns:", X_test.columns.tolist())


# In[54]:


# Drop unwanted features
X_train = X_train.drop(columns=['Lag_Vehicles_1H', 'MovingAvg_2H'])
X_test = X_test.drop(columns=['Lag_Vehicles_1H', 'MovingAvg_2H'])

# Verify again
print("Final X_train columns:", X_train.columns.tolist())
print("Final X_test columns:", X_test.columns.tolist())


# In[55]:


# Drop MovingAvg_6H due to high VIF
X_train = X_train.drop(columns=['MovingAvg_6H'])
X_test = X_test.drop(columns=['MovingAvg_6H'])

# Verify again
print("Final X_train columns:", X_train.columns.tolist())
print("Final X_test columns:", X_test.columns.tolist())


# In[56]:


# Drop 'Vehicles' to match the expected feature set
X_train = X_train.drop(columns=['Vehicles'])
X_test = X_test.drop(columns=['Vehicles'])

# Verify again
print("Final X_train columns:", X_train.columns.tolist())
print("Final X_test columns:", X_test.columns.tolist())


# In[57]:


from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

# Initialize XGBoost Regressor
model = XGBRegressor(
    objective='reg:squarederror',
    n_estimators=100,
    learning_rate=0.1,
    max_depth=6,
    random_state=42
)

# Train the model
model.fit(X_train, y_train)

# Predictions
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# Evaluation Metrics
train_mae = mean_absolute_error(y_train, y_train_pred)
test_mae = mean_absolute_error(y_test, y_test_pred)

train_mse = mean_squared_error(y_train, y_train_pred)
test_mse = mean_squared_error(y_test, y_test_pred)

train_rmse = np.sqrt(train_mse)
test_rmse = np.sqrt(test_mse)

train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)

# Print results
print(f"Training MAE: {train_mae}")
print(f"Testing MAE: {test_mae}")
print(f"Training RMSE: {train_rmse}")
print(f"Testing RMSE: {test_rmse}")
print(f"Training R²: {train_r2}")
print(f"Testing R²: {test_r2}")


# In[58]:


import matplotlib.pyplot as plt
import seaborn as sns

# Get feature importance
feature_importance = model.feature_importances_
features = X_train.columns

# Create DataFrame
importance_df = pd.DataFrame({'Feature': features, 'Importance': feature_importance})
importance_df = importance_df.sort_values(by='Importance', ascending=False)

# Plot feature importance
plt.figure(figsize=(10, 5))
sns.barplot(x='Importance', y='Feature', data=importance_df, palette='viridis')
plt.title('Feature Importance')
plt.show()


# In[59]:


from statsmodels.stats.outliers_influence import variance_inflation_factor
import pandas as pd

# Define the dataframe with selected features
#VIF-Variance Inflation Factor (VIF) is a measure that identifies multicollinearity , lower the score better for the model.
X_vif = X_train.copy()

# Compute VIF for each feature
vif_data = pd.DataFrame()
vif_data["Feature"] = X_vif.columns
vif_data["VIF"] = [variance_inflation_factor(X_vif.values, i) for i in range(X_vif.shape[1])]

# Display results
print(vif_data.sort_values(by="VIF", ascending=False))


# In[61]:


from sklearn.model_selection import cross_val_score, KFold
import numpy as np

# Define cross-validation strategy
# we are using cross validation to test stability of our model
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Perform cross-validation for MAE, RMSE, and R²
mae_scores = cross_val_score(model, X_train, y_train, cv=kf, scoring='neg_mean_absolute_error')
rmse_scores = cross_val_score(model, X_train, y_train, cv=kf, scoring='neg_root_mean_squared_error')
r2_scores = cross_val_score(model, X_train, y_train, cv=kf, scoring='r2')

# Convert negative scores to positive
mae_scores = -mae_scores
rmse_scores = -rmse_scores

# Print results
print(f"Cross-Validation MAE: {np.mean(mae_scores):.5f} ± {np.std(mae_scores):.5f}")
print(f"Cross-Validation RMSE: {np.mean(rmse_scores):.5f} ± {np.std(rmse_scores):.5f}")
print(f"Cross-Validation R²: {np.mean(r2_scores):.5f} ± {np.std(r2_scores):.5f}")


# Final MAE Values for Your Model
# Training MAE: 0.00565
# Testing MAE: 0.00605
# Cross-Validation MAE: 0.00627 ± 0.00009
# 📊 What Does This Mean? (Performance Analysis)
# The Mean Absolute Error (MAE) represents the average absolute difference between the predicted and actual traffic density values.
# 
# 🔍 MAE Benchmarking for Performance Evaluation
# MAE < 0.01 → Excellent (Highly Accurate Predictions ✅)
# MAE between 0.01 - 0.03 → Good (Minor Errors, Still Usable 👍)
# MAE between 0.03 - 0.05 → Moderate (Can be improved, may need tuning 🤔)
# MAE > 0.05 → Poor (Requires significant improvement ❌)
# 🏆 My Model's Performance: ✅
# Since my  MAE is well below 0.01, the model is highly accurate and suitable for real-world deployment.

# In[ ]:




