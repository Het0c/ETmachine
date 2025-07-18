import pandas as pd
import joblib
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.pipeline       import Pipeline
from sklearn.compose        import ColumnTransformer
from sklearn.preprocessing  import OneHotEncoder, StandardScaler

from sklearn.svm            import SVC
from sklearn.ensemble       import RandomForestRegressor

from sklearn.metrics        import (
    accuracy_score, roc_auc_score,
    mean_squared_error, r2_score
)

def load_and_clean(path="csgo_Final.csv"):
    df = pd.read_csv(path, low_memory=False)

    # Columnas numéricas
    num_feats = ['RoundKills', 'RoundDeaths', 'KDR', 'TeamStartingEquipmentValue']
    for col in num_feats:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df.dropna(subset=num_feats, inplace=True)

    # Normalizar RoundWinner
    df['RoundWinner'] = df['RoundWinner'].map({
        True: 1, False: 0,
        'True': 1, 'False': 0,
        'T': 1, 'CT': 0,
        1: 1, 0: 0
    })
    df.dropna(subset=['RoundWinner'], inplace=True)
    df['RoundWinner'] = df['RoundWinner'].astype(int)

    return df

def build_pipelines(cat_features, num_features):
    # Clasificación (SVM): escala num y codifica cat
    preproc_clf = ColumnTransformer([
        ("num", StandardScaler(),        num_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features)
    ])

    clf_pipeline = Pipeline([
        ("preproc", preproc_clf),
        ("svc", SVC(kernel="rbf", C=1.0, probability=True, random_state=42))
    ])

    # Regresión (RF): codifica cat, deja num tal cual
    preproc_reg = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features)
    ], remainder="passthrough")

    reg_pipeline = Pipeline([
        ("preproc", preproc_reg),
        ("rf", RandomForestRegressor(n_estimators=300, max_depth=20, random_state=42))
    ])

    return clf_pipeline, reg_pipeline

def train_and_evaluate(df):
    features  = ['Map', 'Team', 'RoundKills', 'RoundDeaths', 'KDR', 'TeamStartingEquipmentValue']
    cat_feats = ['Map', 'Team']
    num_feats = ['RoundKills', 'RoundDeaths', 'KDR', 'TeamStartingEquipmentValue']

    # Targets
    y_clf = df['RoundWinner']
    y_reg = df['MatchKills']
    X     = df[features]

    # Split stratificado para clasificación y aleatorio para regresión
    X_tr_c, X_te_c, y_tr_c, y_te_c = train_test_split(
        X, y_clf, test_size=0.2, random_state=42, stratify=y_clf
    )
    X_tr_r, X_te_r, y_tr_r, y_te_r = train_test_split(
        X, y_reg, test_size=0.2, random_state=42
    )

    clf_pipe, reg_pipe = build_pipelines(cat_feats, num_feats)

    # Entrenamiento
    clf_pipe.fit(X_tr_c, y_tr_c)
    reg_pipe.fit(X_tr_r, y_tr_r)

    # Evaluación Clasificación
    y_pred_c  = clf_pipe.predict(X_te_c)
    y_proba_c = clf_pipe.predict_proba(X_te_c)[:,1]
    print("Clasificación [RoundWinner]")
    print("Accuracy:", accuracy_score(y_te_c, y_pred_c))
    print("ROC AUC :", roc_auc_score(y_te_c, y_proba_c))

    # Evaluación Regresión
    y_pred_r = reg_pipe.predict(X_te_r)
    print("\nRegresión [MatchKills]")
    print("RMSE:", mean_squared_error(y_te_r, y_pred_r, squared=False))
    print("R²   :", r2_score(y_te_r, y_pred_r))

    # Guardar modelos
    joblib.dump(clf_pipe, "model_classifier.pkl")
    print("\nGuardado: model_classifier.pkl")
    joblib.dump(reg_pipe, "model_regressor.pkl")
    print("Guardado: model_regressor.pkl")

if __name__ == "__main__":
    df = load_and_clean("csgo_Final.csv")
    train_and_evaluate(df)
