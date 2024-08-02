import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

fig_count = 1
path = "/media/pouya/New Volume1/Master SE/Term1/Statistical inference/HW/HW4/diabetes.csv"

data = pd.read_csv(path)
print(data.head())

population = data[data["Outcome"] == 1]["BMI"]

def generate_samples(df, size=100):
    return df.iloc[np.random.randint(0, len(df), size)]

sample = generate_samples(population, 100)

sns.histplot(sample, bins='auto', kde=True, alpha=0.7)
plt.figure(fig_count)
fig_count += 1
plt.xlabel("number of samples")
plt.ylabel("frequency")
plt.title("histogram of generated sample")

def mean_bootstrap(df, simulations=1000, size=100):
    means = np.zeros(shape=simulations)
    for i in range(simulations):
        means[i] = generate_samples(df, size=size).mean()
    return means

bootstrap_means = mean_bootstrap(population)
lower_bound = np.percentile(bootstrap_means, 5)
upper_bound = np.percentile(bootstrap_means, 95)
print(f'The CI for mean using bootstrap and percentile method is: ({lower_bound},{upper_bound})')

print(f'The bootstrap mean is: {bootstrap_means.mean()} and population mean is: {population.mean()}')

plt.figure(fig_count)
sns.histplot(bootstrap_means, kde=True, alpha=0.7, bins='auto')
fig_count += 1
plt.xlabel("means for each simulation")
plt.ylabel("frequency")
plt.title("historgram of means with bootstrap method")
plt.axvline(lower_bound, color='red', linestyle='dashed', linewidth=2, label=f'2.5th percentile (lower CI) = {lower_bound:.2f}')
plt.axvline(upper_bound, color='red', linestyle='dashed', linewidth=2, label=f'97.5th percentile (upper CI) = {upper_bound:.2f}')


sample = generate_samples(population, 10)

plt.figure(fig_count)
sns.histplot(sample, bins='auto', kde=True, alpha=0.7)
fig_count += 1
plt.xlabel("number of samples")
plt.ylabel("frequency")
plt.title("histogram of generated sample")

bootstrap_means = mean_bootstrap(population, size=10)
lower_bound = np.percentile(bootstrap_means, 5)
upper_bound = np.percentile(bootstrap_means, 95)
print(f'The CI for mean using bootstrap and percentile method is: ({lower_bound},{upper_bound})')

print(f'The bootstrap mean is: {bootstrap_means.mean()} and population mean is: {population.mean()}')

plt.figure(fig_count)
sns.histplot(bootstrap_means, kde=True, alpha=0.7, bins='auto')
fig_count += 1
plt.xlabel("means for each simulation")
plt.ylabel("frequency")
plt.title("historgram of means with bootstrap method")
plt.axvline(lower_bound, color='red', linestyle='dashed', linewidth=2, label=f'2.5th percentile (lower CI) = {lower_bound:.2f}')
plt.axvline(upper_bound, color='red', linestyle='dashed', linewidth=2, label=f'97.5th percentile (upper CI) = {upper_bound:.2f}')
plt.show()
