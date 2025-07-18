from flask import Flask, render_template, request
import joblib
import pandas as pd
import math

app = Flask(__name__)

# 1. Carga de ambos modelos
clf_model = joblib.load("model_classifier.pkl")    # Clasificación: RoundWinner
reg_model = joblib.load("model_regressor.pkl")     # Regresión: MatchKills

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    # 2. Leer inputs del formulario
    mapa      = request.form["mapa"]
    equipo    = request.form["equipo"]
    kills     = float(request.form["kills"])
    deaths    = float(request.form["deaths"])
    kdr       = float(request.form["kdr"])
    equip_val = float(request.form["equip_val"])
    
    # 3. Construir DataFrame de entrada
    df_in = pd.DataFrame([[
        mapa, equipo, kills, deaths, kdr, equip_val
    ]], columns=[
        "Map", "Team", "RoundKills", "RoundDeaths", "KDR", "TeamStartingEquipmentValue"
    ])

    # 4. Predicción de clasificación
    clf_pred      = clf_model.predict(df_in)[0]              # 0 ó 1
    clf_probas    = clf_model.predict_proba(df_in)[0]        # [p(0), p(1)]
    # Interpretar label y probabilidad
    winner_label  = "Terrorista" if clf_pred == 1 else "Antiterrorista"
    winner_conf   = round(clf_probas[clf_pred] * 100, 2)     # porcentaje

    # 5. Predicción de regresión
    reg_pred      = reg_model.predict(df_in)[0]
    matchkills    = round(reg_pred)                          # redondeo al entero más cercano

    # 6. Devolver ambos resultados a la plantilla
    return render_template(
        "index.html",
        winner=winner_label,
        confidence=winner_conf,
        matchkills=matchkills
    )

if __name__ == "__main__":
    app.run(debug=True)
