import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from explainerdashboard import ClassifierExplainer, ExplainerDashboard


df = pd.read_csv("/Users/navib/Desktop/Task 1/bank.csv")


df["deposit"] = df["deposit"].map({"no": 0, "yes": 1})


X = df.drop("deposit", axis=1)
y = df["deposit"]


X_encoded = pd.get_dummies(X, drop_first=True)


X_encoded = X_encoded.astype(float)


X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y, test_size=0.2, random_state=42
)


model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
model.fit(X_train, y_train)


explainer = ClassifierExplainer(
    model,
    X_test,
    y_test,
    labels=["No Deposit", "Yes Deposit"]
)


dashboard = ExplainerDashboard(
    explainer,
    title="Bank Deposit Prediction Dashboard",
    whatif=True,
    shap_interaction=False,
    decision_trees=True
)

dashboard.run(port=8050, use_waitress=False)

