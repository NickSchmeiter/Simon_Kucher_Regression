import pandas as pd


from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression





#read excel
df = pd.read_excel(r"C:\Users\nicks\Downloads\PromoEx_HTW_anonymized_data.xlsx")
df['mechanism_detailed'] = df.apply(
    lambda x: x.mechanism.replace('M', str(int(x.M))).replace('N', str(int(x.N))) \
        if x.mechanism in ['Dto NxM', 'Dto N+M'] \
        else x.mechanism, axis=1)

df['Discount'] = 1 - df['PN_old'] / df['PN_new']
df['month'] = df['start_date'].dt.month
df = df.dropna(axis=0, how="any")


X = df[['customer_lv_1', 'region_desc', 'canal_group', 'sku', 'mechanism_detailed', 'month', 'duration_consumer',
 'Discount', 'discount_so']]
# Features
#didnt work bc off wrong datatypes 
#X=df
#X.drop('ROI',axis = 1)
#X.drop('uplift',axis = 1)
#X.drop('class',axis = 1)
y = df['ROI']  # Labels

# One hot encode
X_oneh = pd.get_dummies(X)

# train-test-split
X_train, X_test, y_train, y_test = train_test_split(X_oneh, y, test_size=0.8, random_state=42)

#Grid Building
grid_param = {
    'fit_intercept': [True, False],
    'copy_X': [True, False],
    'normalize': [True, False],
    'positive': [True, False]
}
#Grid Search!!!
gd_sr = GridSearchCV(estimator=LinearRegression(),
                     param_grid=grid_param,
                     cv=5,
                     n_jobs=-1)
gd_sr.fit(X_train, y_train)

# accuracy_score
print('Best parameters: {}'.format(gd_sr.best_params_))
print('Best cross-validation score: {:.2f}'.format(gd_sr.best_score_))
print('Final Test Score with new data: {:.2f}'.format(gd_sr.score(X_test,y_test)))