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

# %%
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

inertia = []
for k in range(1, 6):
    km = KMeans(n_clusters=k, random_state=42)
    km.fit(std_data)
    inertia.append(km.inertia_)

# Plot do Elbow
plt.plot(range(1, 6), inertia, marker='o')
plt.title('Elbow Method')
plt.xlabel('Número de Clusters')
plt.ylabel('Inércia')
plt.grid(True)
plt.show()

# %%
km = KMeans(n_clusters=3, random_state=42)

df['Cluster'] = km.fit_predict(std_data)

display(df.groupby('Cluster').mean())