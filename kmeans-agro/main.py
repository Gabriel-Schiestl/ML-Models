# %%
import pandas as pd
from IPython.display import display
from sklearn.preprocessing import StandardScaler

df = pd.read_excel("SOIL DATA GR.xlsx")
display(df.head())

display(df.describe())
print(df.columns)
# %%
true_data = df[['pH', 'N_NO3 ppm', 'P ppm', 'K ppm ', 'O.M. %', 'CACO3 %', 'EC mS/cm']]

display(true_data.head())

std_data = StandardScaler().fit_transform(true_data)
display(std_data)