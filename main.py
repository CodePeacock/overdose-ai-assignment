import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# Load data into pandas dataframe
df = pd.read_excel(
    "/content/daily_offers.xlsb",
    sheet_name="Result 1",
    usecols=["quantity tons", "thickness", "width", "selling_price"],
)
df["quantity tons"] = pd.to_numeric(df["quantity tons"], errors="coerce")

df.to_parquet("/content/daily_offers.parquet")
df = pd.read_parquet("/content/daily_offers.parquet")

# Inspect data
print(df.head())

# Visualize the distribution of the target variable
# sns.histplot(data=df, x='selling_price')
# plt.show()

# Visualize the relationships between features and target variable
# sns.pairplot(data=df, x_vars=['quantity tons', 'thickness', 'width'], y_vars=['selling_price'])
# plt.show()

# Split data into training and test sets
X = df[["quantity tons", "thickness", "width"]]
y = df["selling_price"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Convert data to TensorFlow tensors
X_train = tf.convert_to_tensor(X_train.values, dtype=tf.float32)
y_train = tf.convert_to_tensor(y_train.values, dtype=tf.float32)
X_test = tf.convert_to_tensor(X_test.values, dtype=tf.float32)
y_test = tf.convert_to_tensor(y_test.values, dtype=tf.float32)

# Create a simple neural network model
model = tf.keras.models.Sequential(
    [tf.keras.layers.Dense(32, activation="relu"), tf.keras.layers.Dense(1)]
)

# Compile the model
model.compile(optimizer="adam", loss="mean_squared_error")

# Train the model on GPU
with tf.device("/GPU:0"):
    model.fit(
        X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test)
    )

# Evaluate the model
y_pred = model.predict(X_test)
print(f"R-squared: {r2_score(y_test, y_pred)}")
print(f"Mean squared error: {mean_squared_error(y_test, y_pred)}")
print(f"Mean absolute error: {mean_absolute_error(y_test, y_pred)}")

# Visualize residuals
sns.scatterplot(x=y_pred.ravel(), y=y_test.numpy() - y_pred.ravel())
plt.show()
