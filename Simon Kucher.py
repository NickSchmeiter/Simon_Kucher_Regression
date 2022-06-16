import pandas as pd


from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split





#read excel
db_pr_rf = pd.read_excel(r"C:\Users\nicks\Downloads\PromoEx_HTW_anonymized_data.xlsx")
db_pr_rf['mechanism_detailed'] = db_pr_rf.apply(
    lambda x: x.mechanism.replace('M', str(int(x.M))).replace('N', str(int(x.N))) \
        if x.mechanism in ['Dto NxM', 'Dto N+M'] \
        else x.mechanism, axis=1)

db_pr_rf['Discount'] = 1 - db_pr_rf['PN_old'] / db_pr_rf['PN_new']
db_pr_rf['month'] = db_pr_rf['start_date'].dt.month


# Drop na
db_pr = db_pr_rf.dropna(axis=0, how="any")


X = db_pr[['customer_lv_1', 'region_desc', 'canal_group', 'sku', 'mechanism_detailed', 'month', 'duration_consumer',
           'Discount', 'discount_so']]  # Features
y = db_pr['ROI']  # Labels

# One hot encode
X_oneh = pd.get_dummies(X)

# train-test-split
X_train, X_test, y_train, y_test = train_test_split(X_oneh, y, test_size=0.7, random_state=42)

print('Training Features Shape:', X_train.shape)
print('Training Labels Shape:', y_train.shape)
print('Testing Features Shape:', X_test.shape)
print('Testing Labels Shape:', y_test.shape)

# Random Forest
random_forest = RandomForestRegressor(n_estimators=200)
random_forest.fit(X_train, y_train)
predicted = random_forest.predict(X_test)

# accuracy_score
accuracy = random_forest.score(X_test, y_test)
print(accuracy)