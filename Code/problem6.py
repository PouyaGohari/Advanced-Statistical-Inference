import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats


Lemon_group = [11, 10, 12]
Flora_group = [11, 14, 11]
Friedfood_group = [5, 5, 8]
None_group = [8, 7, 6]

f_stats, p_value = stats.f_oneway(Lemon_group, Flora_group, Friedfood_group, None_group)

print(f'F-statistic is: {f_stats}')
print(f'p_value is : {p_value}')