import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import numpy as np
# read dataset as a dataframe (each record represents a combination of features and the result-> if it has been selected or not

df=pd.read_csv("candidate_1.tab.txt" , delimiter='\t')
print(df.head())
print(df.columns)
df_input=df[['education', 'religion', 'research_area', 'professional','pricing_group', 'race', 'age_group', 'gender']]

# Step1: perform EDA to see distribution of each attribute
#sns.countplot(data=df,x="gender",hue="religion")
#plt.show()

# Step2: Fit a linear regression model on input features and output(if it is selected)
# generate a regression report
# --- we can use sm.OLS or we can use label encoding / one-hot encoding
model = sm.OLS(df["selected"],pd.get_dummies(data=df_input,columns=df_input.columns))
res = model.fit()
#print(res.summary())
print(res.params)
print(res.params.keys())
print(res.params.values)
print(res.pvalues)

#Step3: Calculate some important measures, including part-worths, attribution importance and utility for each record

#create a dataframe that includes all part-worth for each level of each attribute wtih its p-value
df_res=pd.DataFrame({"name":res.params.keys(),
                     "coeff":res.params.values,
                     "p-value":res.pvalues})
print(df_res.head())
# plot the part-worth for each level in each attribute
fig,axe=plt.subplots(figsize=(30,15))
sns.barplot(data=df_res, x="coeff",y="name",orient='h',order=df_res.sort_values("coeff",ascending=False,key=abs).name)
plt.xticks(rotation=90)
plt.show()


# calculate feature importance for each attribute and plot feature importance
# feature importance = difference between the largest coeff and smallest coeff of each attribute

feature_range={}
for key,coeff in res.params.items():
    feature=key.split("_")[0]
    if feature not in feature_range:
        feature_range[feature]=list()
    feature_range[feature].append(coeff)
print(feature_range)


feature_importance={}
for key in feature_range:
    list=feature_range[key]
    feature_importance[key]= max(list)-min(list)
# feature_importance={key: max(feature_range[key])-min(feature_range[key]) for key in feature_range}
print(feature_importance)

df_feature_importance=pd.DataFrame(data=feature_importance.values(),index=feature_importance.keys(),columns=["feature_importance"])
print(df_feature_importance)
df_feature_importance.plot(kind="bar",rot=90,title="feature_importance",legend=False)
plt.show()

#calculate relative feature importance
total_importance=sum(feature_importance.values())
rel_feature_importance={key: round(value/total_importance,2)*100  for key,value in feature_importance.items()}
print(rel_feature_importance)

# calculate utility for each record, and find the record with the largest utility

df_res["abs_coeff"] = np.abs(df_res["coeff"])
df_res["sig_95"]= [True if x < 0.05 else False for x in df_res["p-value"]]
df_res=df_res.sort_values(by="abs_coeff", ascending=True)
print(df_res)
df_attr_level=pd.get_dummies(data=df_input,columns=df_input.columns)
print(df_attr_level)

for col in df_attr_level:
    df_attr_level.loc[:,col]=df_attr_level.loc[:,col] * df_res.loc[col,"coeff"]
print(df_attr_level)

utility_score=df_attr_level.sum(axis=1)
print(utility_score)

#choose the record with the largest utility
print(df.loc[np.argmax(utility_score)])

