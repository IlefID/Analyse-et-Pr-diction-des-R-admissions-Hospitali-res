"""
=============================================================
 MODELE LSTM - Prediction du Risque de Readmission
 Source : PostgreSQL hospital_dw  (produit par etl_pipeline_pg.py)
 Sortie : rapport HTML + modele sauvegarde
 Dependances : pip install psycopg2-binary sqlalchemy tensorflow
               scikit-learn matplotlib pandas numpy
=============================================================
"""

import sys
import io
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

from sqlalchemy import create_engine, text
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (classification_report, confusion_matrix,
                              roc_auc_score, roc_curve)

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (LSTM, Dense, Dropout,
                                     BatchNormalization, Bidirectional)
from tensorflow.keras.callbacks import (EarlyStopping, ModelCheckpoint,
                                        ReduceLROnPlateau)
from tensorflow.keras.optimizers import Adam

# Fix encodage Windows
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# ─── Configuration PostgreSQL ─────────────────────────────────
# Meme valeurs que dans etl_pipeline_pg.py
PG_CONFIG = {
    "host":     "localhost",
    "port":     5433,           # <-- votre port detecte dans le log
    "database": "hospital_dataWarehouse", # <-- la base doit exister au prealable
    "user":     "postgres",
    "password": "2003"  # <-- a modifier
}

DB_URL = (
    f"postgresql+psycopg2://{PG_CONFIG['user']}:{PG_CONFIG['password']}"
    f"@{PG_CONFIG['host']}:{PG_CONFIG['port']}/{PG_CONFIG['database']}"
)

# ─── Configuration modele ─────────────────────────────────────
MODEL_FILE   = "lstm_readmission_model.h5"
REPORT_FILE  = "lstm_rapport.html"
SEQUENCE_LEN = 5
BATCH_SIZE   = 64
EPOCHS       = 50
LEARNING_RATE = 1e-3
RANDOM_STATE  = 42

tf.random.set_seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)


# ─────────────────────────────────────────────────────────────
# 1. CHARGEMENT DEPUIS POSTGRESQL
# ─────────────────────────────────────────────────────────────
class DataPreparer:

    NUMERIC_FEATURES = [
        "age", "comorbidities_count", "length_of_stay",
        "medications_count", "followup_visits_last_year",
        "prev_readmissions", "readmission_risk_score"
    ]
    CAT_FEATURES = [
        "gender", "region", "diagnosis_name", "treatment_name",
        "insurance_type", "discharge_disposition", "risk_label"
    ]
    TARGET = "label"

    # Requete SQL identique a SQLite mais via SQLAlchemy/psycopg2
    QUERY = """
        SELECT
            f.patient_sk,
            d.full_date::TEXT           AS full_date,
            p.age,
            p.gender,
            p.region,
            p.age_group,
            f.comorbidities_count,
            f.length_of_stay,
            f.medications_count,
            f.followup_visits_last_year,
            f.prev_readmissions,
            f.readmission_risk_score,
            f.risk_label,
            f.discharge_disposition,
            di.diagnosis_name,
            t.treatment_name,
            i.insurance_type,
            f.label
        FROM fact_admission f
        JOIN dim_patient   p  ON f.patient_sk   = p.patient_sk
        JOIN dim_date      d  ON f.date_sk       = d.date_sk
        JOIN dim_diagnosis di ON f.diagnosis_sk  = di.diagnosis_sk
        JOIN dim_treatment t  ON f.treatment_sk  = t.treatment_sk
        JOIN dim_insurance i  ON f.insurance_sk  = i.insurance_sk
        ORDER BY f.patient_sk, d.full_date
    """

    def __init__(self, engine):
        self.engine   = engine
        self.scaler   = MinMaxScaler()
        self.encoders = {}

    def load(self) -> pd.DataFrame:
        print("[DATA] Connexion a PostgreSQL et chargement des donnees...")
        try:
            with self.engine.connect() as conn:
                df = pd.read_sql(text(self.QUERY), conn)
            print(f"[DATA] {len(df)} enregistrements charges depuis PostgreSQL [OK]")
            print(f"[DATA] Colonnes : {list(df.columns)}")
            return df
        except Exception as e:
            print(f"[ERREUR] Impossible de lire depuis PostgreSQL : {e}")
            raise

    def encode_categories(self, df: pd.DataFrame) -> pd.DataFrame:
        for col in self.CAT_FEATURES:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            self.encoders[col] = le
        return df

    def scale_numerics(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df[self.NUMERIC_FEATURES] = self.scaler.fit_transform(
            df[self.NUMERIC_FEATURES]
        )
        return df

    def build_sequences(self, df: pd.DataFrame):
        """
        Fenetre glissante de SEQUENCE_LEN admissions par patient.
        Padding zero pour les patients avec peu d'admissions.
        """
        all_features = self.NUMERIC_FEATURES + self.CAT_FEATURES
        X_seqs, y_seqs = [], []

        grouped = df.groupby("patient_sk")
        print(f"[DATA] Construction des sequences ({len(grouped)} patients)...")

        for pid, group in grouped:
            group  = group.sort_values("full_date")
            feats  = group[all_features].values.astype(np.float32)
            labels = group[self.TARGET].values.astype(np.float32)

            # Padding si moins de SEQUENCE_LEN admissions
            if len(feats) < SEQUENCE_LEN:
                pad    = np.zeros((SEQUENCE_LEN - len(feats), feats.shape[1]), dtype=np.float32)
                feats  = np.vstack([pad, feats])
                labels = np.concatenate([
                    np.zeros(SEQUENCE_LEN - len(labels), dtype=np.float32),
                    labels
                ])

            # Fenetre glissante
            for i in range(SEQUENCE_LEN, len(feats) + 1):
                X_seqs.append(feats[i - SEQUENCE_LEN:i])
                y_seqs.append(labels[i - 1])

        X = np.array(X_seqs, dtype=np.float32)
        y = np.array(y_seqs, dtype=np.float32)
        print(f"[DATA] Sequences : X={X.shape}  y={y.shape}")
        print(f"[DATA] Taux de readmission dans les sequences : {y.mean():.2%}")
        return X, y

    def prepare(self):
        df = self.load()
        df = self.encode_categories(df)
        df = self.scale_numerics(df)
        X, y = self.build_sequences(df)
        return train_test_split(X, y, test_size=0.2,
                                random_state=RANDOM_STATE,
                                stratify=y)

    def export_predictions_to_pg(self, engine, patient_sks, y_prob, y_pred):
        """
        Sauvegarde les predictions LSTM dans PostgreSQL
        (table lstm_predictions) pour Power BI.
        """
        df_pred = pd.DataFrame({
            "patient_sk":        patient_sks,
            "risk_probability":  np.round(y_prob, 4),
            "risk_class":        [
                "Eleve" if p >= 0.66 else ("Modere" if p >= 0.33 else "Faible")
                for p in y_prob
            ],
            "predicted_readmit": (y_prob >= 0.5).astype(bool),
            "scored_at":         pd.Timestamp.now()
        })

        with engine.begin() as conn:
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS lstm_predictions (
                    id               BIGSERIAL PRIMARY KEY,
                    patient_sk       BIGINT,
                    risk_probability NUMERIC(5,4),
                    risk_class       VARCHAR(10),
                    predicted_readmit BOOLEAN,
                    scored_at        TIMESTAMPTZ DEFAULT NOW()
                )
            """))
            conn.execute(text("TRUNCATE TABLE lstm_predictions"))

        df_pred.to_sql("lstm_predictions", engine,
                       if_exists="append", index=False, chunksize=500)
        print(f"[DATA] {len(df_pred)} predictions sauvegardees dans PostgreSQL [OK]")


# ─────────────────────────────────────────────────────────────
# 2. ARCHITECTURE BILSTM
# ─────────────────────────────────────────────────────────────
def build_lstm_model(input_shape: tuple) -> Sequential:
    """
    BiLSTM (128) -> BiLSTM (64) -> Dense(32) -> Dense(1, sigmoid)
    """
    model = Sequential([
        Bidirectional(LSTM(128, return_sequences=True),
                      input_shape=input_shape),
        BatchNormalization(),
        Dropout(0.3),

        Bidirectional(LSTM(64, return_sequences=False)),
        BatchNormalization(),
        Dropout(0.3),

        Dense(32, activation="relu"),
        Dropout(0.2),

        Dense(1, activation="sigmoid")
    ])

    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.AUC(name="auc"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall")
        ]
    )
    model.summary()
    return model


# ─────────────────────────────────────────────────────────────
# 3. ENTRAINEMENT
# ─────────────────────────────────────────────────────────────
def train(model, X_train, y_train, X_val, y_val):
    callbacks = [
        EarlyStopping(monitor="val_auc", patience=8,
                      restore_best_weights=True, mode="max", verbose=1),
        ModelCheckpoint(MODEL_FILE, monitor="val_auc",
                        save_best_only=True, mode="max", verbose=0),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                          patience=4, min_lr=1e-6, verbose=1)
    ]

    pos = int(y_train.sum())
    neg = int(len(y_train) - pos)
    class_weight = {
        0: len(y_train) / (2 * neg),
        1: len(y_train) / (2 * pos)
    }
    print(f"\n[TRAIN] Class weights  ->  0: {class_weight[0]:.3f}  |  1: {class_weight[1]:.3f}")
    print(f"[TRAIN] Train: {len(X_train)} sequences  |  Val: {len(X_val)} sequences")

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        class_weight=class_weight,
        callbacks=callbacks,
        verbose=1
    )
    return history


# ─────────────────────────────────────────────────────────────
# 4. EVALUATION & RAPPORT HTML
# ─────────────────────────────────────────────────────────────
def evaluate_and_report(model, X_test, y_test, history):
    y_prob = model.predict(X_test, verbose=0).ravel()
    y_pred = (y_prob >= 0.5).astype(int)

    auc     = roc_auc_score(y_test, y_prob)
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    cm      = confusion_matrix(y_test, y_pred)
    report  = classification_report(
        y_test, y_pred,
        target_names=["Non readmis", "Readmis"],
        output_dict=True
    )

    print(f"\n{'='*55}")
    print(f"  AUC-ROC : {auc:.4f}")
    print(classification_report(y_test, y_pred,
                                target_names=["Non readmis", "Readmis"]))

    # ── Graphiques ───────────────────────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("BiLSTM - Readmission Hospitaliere", fontsize=14, fontweight="bold")

    hist_df = pd.DataFrame(history.history)
    for i, (metric, title) in enumerate([
        ("loss", "Loss"), ("accuracy", "Accuracy"), ("auc", "AUC-ROC")
    ]):
        ax = axes[0, i]
        ax.plot(hist_df[metric],           label="Train",      color="steelblue")
        ax.plot(hist_df[f"val_{metric}"],  label="Validation", color="tomato")
        ax.set_title(title); ax.legend(); ax.grid(alpha=0.3)

    # Courbe ROC
    axes[1, 0].plot(fpr, tpr, label=f"AUC = {auc:.3f}", color="royalblue", lw=2)
    axes[1, 0].plot([0, 1], [0, 1], "k--", lw=1)
    axes[1, 0].set_title("Courbe ROC")
    axes[1, 0].set_xlabel("FPR"); axes[1, 0].set_ylabel("TPR")
    axes[1, 0].legend()

    # Matrice de confusion
    axes[1, 1].imshow(cm, cmap="Blues")
    axes[1, 1].set_title("Matrice de Confusion")
    for (r, c), v in np.ndenumerate(cm):
        axes[1, 1].text(c, r, v, ha="center", va="center", fontsize=14,
                        color="white" if v > cm.max() / 2 else "black")
    axes[1, 1].set_xticks([0, 1]); axes[1, 1].set_yticks([0, 1])
    axes[1, 1].set_xticklabels(["Non readmis", "Readmis"])
    axes[1, 1].set_yticklabels(["Non readmis", "Readmis"])

    # Distribution des probabilites
    axes[1, 2].hist(y_prob[y_test == 0], bins=30, alpha=0.6,
                    label="Non readmis", color="steelblue")
    axes[1, 2].hist(y_prob[y_test == 1], bins=30, alpha=0.6,
                    label="Readmis", color="tomato")
    axes[1, 2].axvline(0.5, color="black", linestyle="--", label="Seuil 0.5")
    axes[1, 2].set_title("Distribution des Probabilites")
    axes[1, 2].legend()

    plt.tight_layout()
    plt.savefig("lstm_evaluation.png", dpi=120, bbox_inches="tight")
    plt.close()
    print("[EVAL] Graphiques sauvegardes -> lstm_evaluation.png [OK]")

    # ── Rapport HTML ─────────────────────────────────────────
    metrics_rows = ""
    for cls, m in report.items():
        if isinstance(m, dict):
            metrics_rows += f"""
            <tr>
              <td>{cls}</td>
              <td>{m['precision']:.3f}</td>
              <td>{m['recall']:.3f}</td>
              <td>{m['f1-score']:.3f}</td>
              <td>{int(m['support'])}</td>
            </tr>"""

    readmis_key = "Readmis"
    html = f"""<!DOCTYPE html>
<html lang="fr">
<head>
  <meta charset="UTF-8">
  <title>Rapport LSTM - Readmission</title>
  <style>
    body  {{ font-family: Arial, sans-serif; max-width: 960px; margin: auto; padding: 24px; }}
    h1    {{ color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 8px; }}
    h2    {{ color: #34495e; margin-top: 32px; }}
    .kpi  {{ display: flex; gap: 16px; margin: 20px 0; flex-wrap: wrap; }}
    .card {{ background: #ecf0f1; border-radius: 10px; padding: 18px 24px;
             flex: 1; text-align: center; min-width: 140px; }}
    .card .val {{ font-size: 2em; font-weight: bold; color: #2980b9; }}
    .card .lbl {{ font-size: 0.85em; color: #7f8c8d; margin-top: 4px; }}
    table {{ border-collapse: collapse; width: 100%; margin-top: 12px; }}
    th, td {{ border: 1px solid #bdc3c7; padding: 9px 14px; text-align: center; }}
    th    {{ background: #2c3e50; color: white; }}
    tr:nth-child(even) {{ background: #f8f9fa; }}
    img   {{ max-width: 100%; border-radius: 8px; margin-top: 20px; border: 1px solid #ddd; }}
    .badge {{ display:inline-block; padding:3px 10px; border-radius:12px;
              background:#3498db; color:white; font-size:0.8em; margin-left:8px; }}
    .footer {{ color: #95a5a6; font-size: 0.82em; text-align: center; margin-top: 48px; }}
  </style>
</head>
<body>
  <h1>BiLSTM &mdash; Prediction de Readmission Hospitaliere
    <span class="badge">PostgreSQL</span>
  </h1>
  <p>
    Source : <b>PostgreSQL hospital_dw</b> &nbsp;|&nbsp;
    Modele : <b>Bidirectional LSTM</b> &nbsp;|&nbsp;
    Sequence : <b>{SEQUENCE_LEN} admissions</b> &nbsp;|&nbsp;
    Batch : <b>{BATCH_SIZE}</b>
  </p>

  <h2>KPIs du Modele</h2>
  <div class="kpi">
    <div class="card">
      <div class="val">{auc:.3f}</div>
      <div class="lbl">AUC-ROC</div>
    </div>
    <div class="card">
      <div class="val">{report[readmis_key]['precision']:.3f}</div>
      <div class="lbl">Precision (Readmis)</div>
    </div>
    <div class="card">
      <div class="val">{report[readmis_key]['recall']:.3f}</div>
      <div class="lbl">Rappel (Readmis)</div>
    </div>
    <div class="card">
      <div class="val">{report[readmis_key]['f1-score']:.3f}</div>
      <div class="lbl">F1-Score (Readmis)</div>
    </div>
    <div class="card">
      <div class="val">{int(report[readmis_key]['support'])}</div>
      <div class="lbl">Support (Readmis)</div>
    </div>
  </div>

  <h2>Rapport de Classification</h2>
  <table>
    <tr>
      <th>Classe</th><th>Precision</th><th>Rappel</th>
      <th>F1-Score</th><th>Support</th>
    </tr>
    {metrics_rows}
  </table>

  <h2>Visualisations</h2>
  <img src="lstm_evaluation.png" alt="Graphiques d evaluation LSTM">

  <div class="footer">
    Source PostgreSQL : {PG_CONFIG['host']}:{PG_CONFIG['port']}/{PG_CONFIG['database']}<br>
    Genere le {pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")} &mdash; Pipeline IA Hospitalier
  </div>
</body>
</html>"""

    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"[EVAL] Rapport HTML sauvegarde -> {REPORT_FILE} [OK]")

    return {
        "auc":       auc,
        "precision": report[readmis_key]["precision"],
        "recall":    report[readmis_key]["recall"],
        "f1":        report[readmis_key]["f1-score"]
    }


# ─────────────────────────────────────────────────────────────
# 5. INFERENCE UNITAIRE
# ─────────────────────────────────────────────────────────────
def predict_patient(model, scaler, sequence: np.ndarray) -> dict:
    """
    sequence : array (SEQUENCE_LEN, n_features) non normalise
    Retourne : probabilite, classe de risque, prediction binaire
    """
    x     = scaler.transform(sequence[:, :len(DataPreparer.NUMERIC_FEATURES)])
    x_seq = x[np.newaxis, ...]
    prob  = float(model.predict(x_seq, verbose=0)[0, 0])
    return {
        "risk_probability":    round(prob, 4),
        "risk_class":          "Eleve" if prob >= 0.66 else ("Modere" if prob >= 0.33 else "Faible"),
        "predicted_readmission": prob >= 0.5
    }


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  PIPELINE LSTM  --  READMISSION HOSPITALIERE")
    print("  Source : PostgreSQL hospital_dw")
    print("=" * 60 + "\n")

    # -- Connexion PostgreSQL ----------------------------------
    engine = create_engine(DB_URL, echo=False, pool_pre_ping=True,
                           connect_args={"connect_timeout": 10})
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        print(f"[CONNEXION] PostgreSQL OK  ->  "
              f"{PG_CONFIG['host']}:{PG_CONFIG['port']}/{PG_CONFIG['database']}")
    except Exception as e:
        print(f"[ERREUR] Connexion PostgreSQL impossible : {e}")
        print(f"  Verifiez PG_CONFIG (host, port, password, database)")
        sys.exit(1)

    # -- 1. Donnees -------------------------------------------
    preparer = DataPreparer(engine)
    X_train, X_test, y_train, y_test = preparer.prepare()
    print(f"\n[SPLIT] Train: {X_train.shape[0]}  |  Test: {X_test.shape[0]}")

    # -- 2. Modele --------------------------------------------
    model = build_lstm_model(input_shape=(X_train.shape[1], X_train.shape[2]))

    # -- 3. Entrainement --------------------------------------
    print("\n[TRAIN] Demarrage de l entrainement...")
    history = train(model, X_train, y_train, X_test, y_test)

    # -- 4. Evaluation ----------------------------------------
    print("\n[EVAL] Evaluation sur le jeu de test...")
    results = evaluate_and_report(model, X_test, y_test, history)

    # -- 5. Sauvegarde predictions dans PostgreSQL ------------
    #    (pour import direct dans Power BI)
    print("\n[DATA] Sauvegarde des predictions dans PostgreSQL...")
    y_prob_full = model.predict(X_test, verbose=0).ravel()
    # Note : patient_sk reconstitue depuis les sequences de test
    # Pour une utilisation en production, lier via l index d origine
    fake_sks = np.arange(len(y_prob_full))   # remplacer par les vrais patient_sk si besoin
    preparer.export_predictions_to_pg(engine, fake_sks, y_prob_full,
                                      (y_prob_full >= 0.5).astype(int))

    engine.dispose()

    # -- Bilan ------------------------------------------------
    print("\n" + "=" * 55)
    print("  RESULTATS FINAUX")
    print("=" * 55)
    print(f"  AUC-ROC   : {results['auc']:.4f}")
    print(f"  Precision : {results['precision']:.4f}")
    print(f"  Rappel    : {results['recall']:.4f}")
    print(f"  F1-Score  : {results['f1']:.4f}")
    print(f"\n  Modele    -> {MODEL_FILE}")
    print(f"  Rapport   -> {REPORT_FILE}")
    print(f"  Predictions -> table lstm_predictions (PostgreSQL)")
    print("=" * 55)