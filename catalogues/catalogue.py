import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv("Below_6.0_SAO.csv", usecols=["Star ID", "RA", "DE", "Magnitude"])
df = df[df["Magnitude"] < 5.7]
df.to_csv("Below_5.7_SAO.csv", index=False)

ra, de = df['RA'], df['DE']
plt.scatter(ra, de, s=1)
plt.show()