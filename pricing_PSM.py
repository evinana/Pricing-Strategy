import pandas as pd
import matplotlib.pyplot as plt

df=pd.read_csv("psm.csv")
df1=df\
    .unstack()\
    .reset_index()\
    .drop(columns='level_1')\
    .rename(columns={0:"Price","level_0": "Label"})\
    .groupby(["Label","Price"])\
    .size()\
    .reset_index()\
    .rename(columns={0:"Frequency"})

df1["Sum"]=df1.groupby("Label")["Frequency"].transform("sum")
df1["CumSum"]=df1.groupby("Label")["Frequency"].cumsum()
df1["Percentage"]=df1["CumSum"]/df1["Sum"]*100
print(df1)

df2=df1.pivot_table(values="Percentage",index="Price",columns="Label")

df2=df2.ffill().fillna(0)
#df2=df2.interpolate().fillna(0)
print(df2)
df2["Cheap"]=100-df2["Cheap"]
df2["Too Cheap"]=100-df2["Too Cheap"]
df2.plot()
plt.show()
