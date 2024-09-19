import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

# Sample data
data = {
    'Year': [2017, 2018, 2019, 2020, 2021, 2022, 2023],
    'FBWA': [126193, 165339, 192991, 266321, 259705, 227825, 189883],
    'FFTx': [27123, 74054, 132730, 217468, 285572, 399637, 503460],
    'xDSL': [195135, 152042, 131775, 136352, 138448, 128753, 113174],
    'Registered_Digital_Companies': [510, 601, 614, 580, 748, 920, 1039]
}

df = pd.DataFrame(data)

# Independent variables
X = df[['FBWA', 'FFTx', 'xDSL']]
X = sm.add_constant(X)  # adding a constant

# Dependent variable
y = df['Registered_Digital_Companies']

# Fit the regression model
model = sm.OLS(y, X).fit()

# Get predictions and residuals
predictions = model.predict()
residuals = model.resid

# Print out the statistics
print(model.summary())

# Plotting the relationship
plt.figure(figsize=(10, 6))
plt.plot(df['Year'], df['Registered_Digital_Companies'], label='Actual Registered Digital Companies', marker='o')
plt.plot(df['Year'], predictions, label='Predicted', linestyle='--', marker='x')
plt.xlabel('Year')
plt.ylabel('Number of Companies')
plt.title('Registered Digital Companies and FIT Predictions Over the Years')
plt.legend()
plt.grid(True)
plt.show()

# Plotting residuals
plt.figure(figsize=(10, 6))
plt.stem(df['Year'], residuals)
plt.xlabel('Year')
plt.ylabel('Residuals')
plt.title('Residual Plot')
plt.grid(True)
plt.show()

# Regression equation
coefficients = model.params
equation = f'Regression Equation: y = {coefficients[0]:.4f} '
for i, col in enumerate(X.columns[1:]):
    equation += f'+ ({coefficients[i+1]:.4f} * {col}) '

print(equation)

