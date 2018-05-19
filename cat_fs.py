import _data
import cat
from sklearn.metrics import roc_auc_score
import catboost

train_data, test_data = _data.data()
feature_score = cat.feature_score()
new_cols = feature_score.iloc[0:18, :]["Feature"]
cat_feature_inds = []
for i, c in enumerate(train_data[new_cols].columns.values):
    num_uniques = len(train_data[new_cols][c].unique())
    if num_uniques < 5:
        cat_feature_inds.append(i)

cat_model = catboost.CatBoostClassifier(
            iterations=400,
            learning_rate=0.03,
            depth=6,
            l2_leaf_reg=1,
            eval_metric='F1',
            random_seed=4 * 100 + 6)

cat_model.fit(train_data[new_cols], train_data.diabetes, cat_features=cat_feature_inds)

print("The test auc is %.4f"% roc_auc_score(test_data.diabetes, cat_model.predict_proba(test_data[new_cols])[:, 1]))