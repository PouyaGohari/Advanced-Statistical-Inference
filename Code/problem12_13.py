import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.model_selection import train_test_split 
from sklearn.metrics import r2_score, mean_squared_error

print("\n")
print("Question 12:")
print("\n")
fig_count = 1
path = "/media/pouya/New Volume1/Master SE/Term1/Statistical inference/HW/HW4/prostate_analysis_results.csv"

data = pd.read_csv(path)
x = data['lcavol']
y = data['lpsa']

x = sm.add_constant(x)
model = sm.OLS(y, x, missing='drop')
model_result = model.fit()
print(model_result.summary())
error_variance = model_result.scale
print(f'The residula erros is: {error_variance}')
Fisher_information = np.linalg.inv(np.dot(x.T, x))
CRLB = np.diag(Fisher_information) * error_variance
model_se = model_result.bse

print("Cramer-Rao Lower Bounds for the variances of the estimators:", CRLB)
print("Empirical standard errors of the estimates:", model_se)

sum_squared = ((data['lcavol']-data['lcavol'].mean())**2).sum()
fissher_information = error_variance/sum_squared
print('the cramer rao lower bounds with formula:',fissher_information)
print('the epmprical is:', model_result.bse['lcavol']**2)

print("\n")
print("Question 13:")
print("\n")
#part 13
print("\n")
print("Question 13: part 1")
print("\n")
# 1)
x = data[['age', 'lpsa']]
y = data['lweight']
x = sm.add_constant(x)
model = sm.OLS(y, x).fit()
print(model.summary())

intercept, beta_age, beta_lpsa = model.params

# 2.a)
print("\n")
print("Question 13: part 2.a seeing both explanatory variables once")
print("\n")
residuals = model.resid
plt.figure(fig_count)
fig_count += 1
plt.scatter(data['age'], residuals)
plt.axhline(y = 0, color='r', linestyle='--')
plt.xlabel('age')
plt.ylabel('residulas')
plt.title('residulas and age')

plt.figure(fig_count)
fig_count += 1
plt.scatter(data['lpsa'], residuals)
plt.axhline(y = 0, color='r', linestyle='--')
plt.xlabel('lpsa')
plt.ylabel('residulas')
plt.title('residulas and lpsa')


# 2.b)
print("\n")
print("Question 13: part 2.b seeing both explanatory variables once")
print("\n")
my_x = data[['age', 'lpsa']]
my_x = np.array(my_x)
ones_cols = np.ones((my_x.shape[0], 1), dtype=my_x.dtype)
my_x = np.hstack((ones_cols, my_x))
my_y = np.array(y)
estimate_bethas = np.dot(np.linalg.inv(np.dot(my_x.T, my_x)),np.dot(my_x.T,y))
print(estimate_bethas)
# 2.c)
print("\n")
print("Question 13: part 2.c seeing both explanatory variables once")
print("\n")
print(f"The predictive equation is: weight = {intercept:.2f} + {beta_age:.2f}*age + {beta_lpsa:.2f}*lpsa")

# 2.d) Scatter plot with the regression line
plt.figure(fig_count)
fig_count += 1
plt.scatter(data['age'], y, label='Age vs Weight')
predicted_values = model.predict(x)
plt.plot(data['age'], predicted_values, 'r--', label='Regression Line')
plt.xlabel('Age')
plt.ylabel('Weight')
plt.title('Age vs Weight with Regression Line')
plt.legend()

plt.figure(fig_count)
fig_count += 1
plt.scatter(data['lpsa'], y, label='Lpsa vs Weight')
plt.plot(data['lpsa'], predicted_values, 'r--', label='Regression Line')
plt.xlabel('Lpsa')
plt.ylabel('Weight')
plt.title('Lpsa vs Weight with Regression Line')
plt.legend()



print("\n")
print("Question 13: part 2.a seeing each explanatory variables once")
print("\n")
## seeing each explantory variable at a time
 # part a
x_age = data['age']
y = data['lweight']
x_age = sm.add_constant(x_age)
model_age = sm.OLS(y, x_age).fit()
print(model_age.summary())
intercept_age , beta_age = model_age.params
residuals = model_age.resid

x_lpsa = data['lpsa']
y = data['lweight']
x_lpsa = sm.add_constant(x_lpsa)
model_lpsa = sm.OLS(y, x_lpsa).fit()
print(model_lpsa.summary())
intercept_lpsa, beta_lpsa = model_lpsa.params

print("\n")
print("Question 13: part 2.b seeing each explanatory variables once")
print("\n")

 # part b
my_x = np.array(data[['age']])
my_y = np.array(data[['lweight']])
ones_cols = np.ones((my_x.shape[0], 1), dtype=my_x.dtype)
my_x = np.hstack((ones_cols, my_x))
estimate_bethas = np.dot(np.linalg.inv(np.dot(my_x.T, my_x)),np.dot(my_x.T,y))
print(f'The esimated const and slop for age and weight is:{estimate_bethas}')

my_x = np.array(data[['lpsa']])
my_y = np.array(data[['lweight']])
ones_cols = np.ones((my_x.shape[0], 1), dtype=my_x.dtype)
my_x = np.hstack((ones_cols, my_x))
estimate_bethas = np.dot(np.linalg.inv(np.dot(my_x.T, my_x)),np.dot(my_x.T,y))
print(f'The esimated const and slop for lpsa and weight is:{estimate_bethas}')

print("\n")
print("Question 13: part 2.c seeing each explanatory variables once")
print("\n")
 # part c
print(f"The predictive equation(age,weight) is: weight = {intercept_age:.2f} + {beta_age:.2f}*age")
print(f"The predictive equation(lpsa,weight) is: weight = {intercept_lpsa:.2f} + {beta_lpsa:.2f}*lpsa")

print("\n")
print("Question 13: part 2.d seeing each explanatory variables once")
print("\n")

plt.figure(fig_count)
fig_count += 1
plt.scatter(data['age'], y, label='Age vs Weight')
predicted_values = model_age.predict(x_age)
plt.plot(data['age'], predicted_values, 'r--', label='Regression Line')
plt.xlabel('Age')
plt.ylabel('Weight')
plt.title('Age vs Weight with Regression Line')
plt.legend()

plt.figure(fig_count)
fig_count += 1
predicted_values = model_lpsa.predict(x_lpsa)
plt.scatter(data['lpsa'], y, label='Lpsa vs Weight')
plt.plot(data['lpsa'], predicted_values, 'r--', label='Regression Line')
plt.xlabel('Lpsa')
plt.ylabel('Weight')
plt.title('Lpsa vs Weight with Regression Line')
plt.legend()
# part 4
print("\n")
print("Question 13: part 4")
print("\n")
sample = data.sample(n=50, random_state=1, replace=True)
train_sample, test_sample = train_test_split(sample, test_size=0.1, random_state=1)
# print(sample)

print("\n")
print("Question 13: part 4.a")
print("\n")
## part a

x = train_sample['age']
y = train_sample['lweight']
x = sm.add_constant(x)
model_age = sm.OLS(y, x, missing='drop')
model_age_result = model_age.fit()
print(model_age_result.summary())

x = train_sample['lpsa']
y = train_sample['lweight']
x = sm.add_constant(x)
model_lpsa = sm.OLS(y, x, missing='drop')
model_lpsa_result = model_lpsa.fit()
print(model_lpsa_result.summary())

print("\n")
print("Question 13: part 4.b")
print("\n")

## part b
CI_age = model_age_result.conf_int(alpha=0.05).loc['age']
print("Confidence interval for age:", CI_age[0], CI_age[1])
CI_lpsa = model_lpsa_result.conf_int(alpha=0.05).loc['lpsa']
print("Confidence interval for lpsa:",CI_lpsa[0], CI_lpsa[1])

print("\n")
print("Question 13: part 4.c")
print("\n")
## part c
test_sample_age = sm.add_constant(test_sample['age'])
prediction_age_based = model_age_result.predict(test_sample_age)
test_sample_lpsa = sm.add_constant(test_sample['lpsa'])
prediction_lpsa_based = model_lpsa_result.predict(test_sample_lpsa)
generete_pd = pd.DataFrame({
    "Actual values": test_sample['lweight'],
    "Predict with age": prediction_age_based,
    "predict with lpsa": prediction_lpsa_based
})
print(generete_pd)

## part d
print("\n")
print("Question 13: part 4.d")
print("\n")

r2_age = r2_score(test_sample['lweight'], prediction_age_based)
r2_lpsa = r2_score(test_sample['lweight'], prediction_lpsa_based)
MSE_age = mean_squared_error(test_sample['lweight'], prediction_age_based)
MSE_lpsa = mean_squared_error(test_sample['lweight'], prediction_lpsa_based)

print(f'r2 score for age is: {r2_age}')
print(f'r2 score for lpsa is: {r2_lpsa}')
print(f'mean square error for age: {MSE_age}')
print(f'mean square error for lpsa: {MSE_lpsa}')
plt.show()