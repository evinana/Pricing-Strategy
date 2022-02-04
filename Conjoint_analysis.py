import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

df=pd.read_csv("candidate_1.tab.txt" , delimiter='\t')
print(df.head())
print(df.columns)
df_input=df[['education', 'religion', 'research_area', 'professional','pricing_group', 'race', 'age_group', 'gender']]

# perform EDA to see distribution of each attribute
#sns.countplot(data=df,x="gender",hue="religion")
#plt.show()

# Fit a linear regression model on input features and output(if it is selected)
# --- we can use sm.OLS or we can use label encoding / one-hot encoding
model = sm.OLS(df["selected"],pd.get_dummies(data=df_input,columns=df_input.columns))
res = model.fit()
#print(res.summary())
print(res.params)
print(res.params.keys())
print(res.params.values)
print(res.pvalues)

#create a dataframe that includes all part-worth for each level of each attribute wtih its p-value
df_res=pd.DataFrame({"name":res.params.keys(),
                     "coeff":res.params.values,
                     "p-value":res.pvalues})
print(df_res.head())


# plot the part-worth for each level in each attribute
#plt.figure(figsize=(50, 25))
sns.barplot(data=df_res, x="coeff",y="name",orient='h',order=df_res.sort_values("coeff",ascending=False,key=abs).name)
plt.xticks(rotation=90)
plt.show()


# calculate feature importance for each attribute
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
print(feature_importance)
total_importance=sum(feature_importance.values())

#calculate relative feature importance 
rel_feature_importance={key: round(value/total_importance,2)*100  for key,value in feature_importance.items()}
print(rel_feature_importance)
