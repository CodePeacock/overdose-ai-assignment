''''''
# Import libraries
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.tree import DecisionTreeRegressor

# Load the data from Excel file
data = pd.read_excel("filename.xlsb")

# Remove unnecessary columns
data = data.drop(columns=["material_ref", "product_ref", "delivery date"])

# Replace missing values with the mean value of that column
data = data.fillna(data.mean())


# One-hot encode categorical variables
categorical_columns = ["status", "item type"]
enc = OneHotEncoder()
enc.fit(data[categorical_columns])
encoded_data = pd.DataFrame(enc.transform(data[categorical_columns]).toarray())
data = pd.concat([data, encoded_data], axis=1)
data = data.drop(columns=categorical_columns)

# Normalize the numerical variables using MinMaxScaler
scaler = MinMaxScaler()
numerical_columns = ["quantity tons", "thickness", "width"]
data[numerical_columns] = scaler.fit_transform(data[numerical_columns])

# Analyze the relationships between variables using statistical and visualization methods
sns.pairplot(data)
plt.show()

# Analyze the correlation between variables
corr = data.corr()
sns.heatmap(corr, annot=True)
plt.show()


# Analyze the relationship between the target variable 'selling_price' and the numerical variable 'quantity tons'
sns.scatterplot(x="quantity tons", y="selling_price", data=data)
plt.show()

# Split the data into training and testing sets
X = data.drop(columns=["selling_price"])
y = data["selling_price"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)

# Build regression models
models = [
    ("Linear Regression", LinearRegression()),
    ("Ridge Regression", Ridge()),
    ("Lasso Regression", Lasso()),
    ("ElasticNet Regression", ElasticNet()),
    ("Decision Tree Regression", DecisionTreeRegressor()),
    ("Random Forest Regression", RandomForestRegressor()),
    ("Gradient Boosting Regression", GradientBoostingRegressor()),
    ("Neural Network Regression", MLPRegressor()),
]

results = []
for name, model in models:
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    results.append((name, r2, mae, mse))

results_df = pd.DataFrame(results, columns=["Model", "R2 Score", "MAE", "MSE"])
print(results_df)

# Hyperparameter tuning using GridSearchCV
params = {
    "n_estimators": [50, 100, 150],
    "max_depth": [2, 4, 6, 8],
    "min_samples_split": [2, 4, 6],
    "min_samples_leaf": [1, 2, 4],
}


from sklearn.ensemble import RandomForestRegressor

# Create random forest regression model
rf_reg = RandomForestRegressor(n_estimators=100, random_state=42)

# Train random forest regression model on training set
rf_reg.fit(X_train, y_train)

# Evaluate random forest regression model on testing set
rf_reg_y_pred = rf_reg.predict(X_test)
rf_reg_r2_score = r2_score(y_test, rf_reg_y_pred)
rf_reg_mae = mean_absolute_error(y_test, rf_reg_y_pred)
rf_reg_mse = mean_squared_error(y_test, rf_reg_y_pred)

rf_reg_mae, rf_reg_mse, rf_reg_r2_score

# Print evaluation metrics for random forest regression model
print("Random Forest Regression Model Evaluation Metrics:")
print("R2 Score: ", rf_reg_r2_score)
print("Mean Absolute Error: ", rf_reg_mae)
print("Mean Squared Error: ", rf_reg_mse)
