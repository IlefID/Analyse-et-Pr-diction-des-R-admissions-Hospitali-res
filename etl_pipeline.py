"""
=============================================================
 ETL PIPELINE - Data Warehouse Parcours Patient
 Source : hospital_readmission_dataset.csv
 Target : PostgreSQL (schéma en étoile)
 Dépendances : pip install psycopg2-binary sqlalchemy pandas
=============================================================
"""

import pandas as pd
import numpy as np
import hashlib
import logging
import sys
import io
from pathlib import Path
from datetime import datetime

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

# ─── Fix encodage Windows (cp1252 → utf-8) ───────────────────
# Remplace les caracteres speciaux par des equivalents ASCII
# pour eviter UnicodeEncodeError sur le terminal Windows
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# ─── Configuration logging (sans emojis, compatible Windows) ─
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("etl_pipeline.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout)
    ]
)
log = logging.getLogger("ETL")

# ── Parametres de connexion PostgreSQL ───────────────────────
# IMPORTANT : verifiez ces valeurs dans pgAdmin :
#   - port 5433 detecte dans votre log (pas 5432 par defaut)
#   - database : creez la base si elle n'existe pas encore
#     (voir instructions plus bas)
PG_CONFIG = {
    "host":     "localhost",
    "port":     5433,           # <-- votre port detecte dans le log
    "database": "hospital_dataWarehouse", # <-- la base doit exister au prealable
    "user":     "postgres",
    "password": "2003"  # <-- a modifier
}

SRC_FILE = "hospital_readmission_dataset.csv"

# URL SQLAlchemy
DB_URL = (
    f"postgresql+psycopg2://{PG_CONFIG['user']}:{PG_CONFIG['password']}"
    f"@{PG_CONFIG['host']}:{PG_CONFIG['port']}/{PG_CONFIG['database']}"
)


# ─────────────────────────────────────────────────────────────
# LAYER 1 : EXTRACT
# ─────────────────────────────────────────────────────────────
class Extractor:
    REQUIRED_COLUMNS = {
        "patient_id", "admission_date", "season", "age", "gender",
        "region", "primary_diagnosis", "comorbidities_count",
        "length_of_stay", "treatment_type", "medications_count",
        "followup_visits_last_year", "prev_readmissions",
        "insurance_type", "discharge_disposition",
        "readmission_risk_score", "label"
    }

    def __init__(self, filepath: str):
        self.filepath = Path(filepath)

    def run(self) -> pd.DataFrame:
        log.info(f"[EXTRACT] Lecture de {self.filepath}")
        if not self.filepath.exists():
            raise FileNotFoundError(f"Fichier introuvable : {self.filepath}")

        df = pd.read_csv(self.filepath)
        log.info(f"[EXTRACT] {len(df)} lignes, {len(df.columns)} colonnes chargees")

        missing = self.REQUIRED_COLUMNS - set(df.columns)
        if missing:
            raise ValueError(f"Colonnes manquantes : {missing}")

        log.info("[EXTRACT] Schema valide [OK]")
        return df


# ─────────────────────────────────────────────────────────────
# LAYER 2 : TRANSFORM
# ─────────────────────────────────────────────────────────────
class Transformer:

    AGE_GROUPS = {
        (18, 30): "18-30", (31, 45): "31-45", (46, 60): "46-60",
        (61, 75): "61-75", (76, 200): "76+"
    }
    RISK_LABELS = {
        (0.00, 0.33): "Faible",
        (0.33, 0.66): "Modéré",
        (0.66, 1.01): "Élevé"
    }

    def run(self, df: pd.DataFrame) -> dict[str, pd.DataFrame]:
        log.info("[TRANSFORM] Debut de la transformation")
        df = self._clean_base(df.copy())

        dim_patient   = self._build_dim_patient(df)
        dim_date      = self._build_dim_date(df)
        dim_diagnosis = self._build_dim_diagnosis(df)
        dim_treatment = self._build_dim_treatment(df)
        dim_insurance = self._build_dim_insurance(df)
        fact          = self._build_fact(df, dim_date, dim_diagnosis,
                                         dim_treatment, dim_insurance)

        log.info(f"[TRANSFORM] fact_admission : {len(fact)} lignes [OK]")
        return {
            "dim_patient":    dim_patient,
            "dim_date":       dim_date,
            "dim_diagnosis":  dim_diagnosis,
            "dim_treatment":  dim_treatment,
            "dim_insurance":  dim_insurance,
            "fact_admission": fact
        }

    def _clean_base(self, df):
        df["admission_date"] = pd.to_datetime(df["admission_date"])
        for col in ["gender", "region", "primary_diagnosis",
                    "treatment_type", "insurance_type",
                    "discharge_disposition", "season"]:
            df[col] = df[col].str.strip().str.title()

        df["age"]                   = df["age"].clip(0, 120)
        df["length_of_stay"]        = df["length_of_stay"].clip(0, 365)
        df["readmission_risk_score"]= df["readmission_risk_score"].clip(0, 1)
        df["age_group"]  = df["age"].apply(self._age_group)
        df["risk_label"] = df["readmission_risk_score"].apply(self._risk_label)
        df["readmitted"] = df["label"].map({1: "Oui", 0: "Non"})
        df["patient_sk"] = df["patient_id"].apply(
            lambda x: int(hashlib.md5(x.encode()).hexdigest()[:8], 16)
        )
        log.info("[TRANSFORM] Nettoyage termine [OK]")
        return df

    def _build_dim_patient(self, df):
        dim = (df[["patient_sk","patient_id","age","age_group","gender","region"]]
               .drop_duplicates("patient_sk").reset_index(drop=True))
        log.info(f"[TRANSFORM] dim_patient : {len(dim)} lignes")
        return dim

    def _build_dim_date(self, df):
        dates = df["admission_date"].drop_duplicates().reset_index(drop=True)
        seasons = (df.drop_duplicates("admission_date")
                   .set_index("admission_date")["season"])
        dim = pd.DataFrame({
            "date_sk":     dates.dt.strftime("%Y%m%d").astype(int),
            "full_date":   dates.dt.date.astype(str),
            "year":        dates.dt.year,
            "quarter":     dates.dt.quarter,
            "month":       dates.dt.month,
            "month_name":  dates.dt.month_name(),
            "week":        dates.dt.isocalendar().week.astype(int),
            "day_of_week": dates.dt.day_name(),
            "season":      dates.map(seasons),
            "is_weekend":  (dates.dt.dayofweek >= 5).astype(bool)
        })
        log.info(f"[TRANSFORM] dim_date : {len(dim)} lignes")
        return dim

    def _build_dim_diagnosis(self, df):
        diag = df["primary_diagnosis"].drop_duplicates().reset_index(drop=True)
        return pd.DataFrame({
            "diagnosis_sk":   range(1, len(diag) + 1),
            "diagnosis_name": diag.values,
            "diagnosis_code": diag.str[:3].str.upper().values
        })

    def _build_dim_treatment(self, df):
        treat = df["treatment_type"].drop_duplicates().reset_index(drop=True)
        return pd.DataFrame({
            "treatment_sk":   range(1, len(treat) + 1),
            "treatment_name": treat.values
        })

    def _build_dim_insurance(self, df):
        ins = df["insurance_type"].drop_duplicates().reset_index(drop=True)
        return pd.DataFrame({
            "insurance_sk":   range(1, len(ins) + 1),
            "insurance_type": ins.values
        })

    def _build_fact(self, df, d_date, d_diag, d_treat, d_ins):
        df = df.merge(
            d_date[["date_sk","full_date"]],
            left_on=df["admission_date"].dt.date.astype(str),
            right_on="full_date", how="left"
        ).drop(columns=["full_date","key_0"], errors="ignore")

        df = df.merge(d_diag, left_on="primary_diagnosis",
                      right_on="diagnosis_name", how="left")
        df = df.merge(d_treat, left_on="treatment_type",
                      right_on="treatment_name", how="left")
        df = df.merge(d_ins,  on="insurance_type", how="left")

        return pd.DataFrame({
            "patient_sk":               df["patient_sk"],
            "date_sk":                  df["date_sk"],
            "diagnosis_sk":             df["diagnosis_sk"],
            "treatment_sk":             df["treatment_sk"],
            "insurance_sk":             df["insurance_sk"],
            "age":                      df["age"],
            "comorbidities_count":      df["comorbidities_count"],
            "length_of_stay":           df["length_of_stay"],
            "medications_count":        df["medications_count"],
            "followup_visits_last_year":df["followup_visits_last_year"],
            "prev_readmissions":        df["prev_readmissions"],
            "readmission_risk_score":   df["readmission_risk_score"],
            "risk_label":               df["risk_label"],
            "discharge_disposition":    df["discharge_disposition"],
            "readmitted":               df["readmitted"],
            "label":                    df["label"],
            "etl_timestamp":            datetime.now().isoformat()
        })

    def _age_group(self, age):
        for (lo, hi), label in self.AGE_GROUPS.items():
            if lo <= age <= hi:
                return label
        return "Inconnu"

    def _risk_label(self, score):
        for (lo, hi), label in self.RISK_LABELS.items():
            if lo <= score < hi:
                return label
        return "Inconnu"


# ─────────────────────────────────────────────────────────────
# LAYER 3 : LOAD  (PostgreSQL via SQLAlchemy)
# ─────────────────────────────────────────────────────────────
class Loader:

    # DDL PostgreSQL — types natifs (SERIAL, DATE, BOOLEAN, NUMERIC)
    DDL = {
        "dim_patient": """
            CREATE TABLE IF NOT EXISTS dim_patient (
                patient_sk   BIGINT PRIMARY KEY,
                patient_id   VARCHAR(20)  NOT NULL,
                age          SMALLINT,
                age_group    VARCHAR(10),
                gender       VARCHAR(20),
                region       VARCHAR(50)
            )""",

        "dim_date": """
            CREATE TABLE IF NOT EXISTS dim_date (
                date_sk      INTEGER PRIMARY KEY,
                full_date    DATE,
                year         SMALLINT,
                quarter      SMALLINT,
                month        SMALLINT,
                month_name   VARCHAR(15),
                week         SMALLINT,
                day_of_week  VARCHAR(12),
                season       VARCHAR(10),
                is_weekend   BOOLEAN
            )""",

        "dim_diagnosis": """
            CREATE TABLE IF NOT EXISTS dim_diagnosis (
                diagnosis_sk   SERIAL PRIMARY KEY,
                diagnosis_name VARCHAR(100),
                diagnosis_code VARCHAR(10)
            )""",

        "dim_treatment": """
            CREATE TABLE IF NOT EXISTS dim_treatment (
                treatment_sk   SERIAL PRIMARY KEY,
                treatment_name VARCHAR(100)
            )""",

        "dim_insurance": """
            CREATE TABLE IF NOT EXISTS dim_insurance (
                insurance_sk   SERIAL PRIMARY KEY,
                insurance_type VARCHAR(50)
            )""",

        "fact_admission": """
            CREATE TABLE IF NOT EXISTS fact_admission (
                fact_id                    BIGSERIAL PRIMARY KEY,
                patient_sk                 BIGINT     REFERENCES dim_patient(patient_sk),
                date_sk                    INTEGER    REFERENCES dim_date(date_sk),
                diagnosis_sk               INTEGER    REFERENCES dim_diagnosis(diagnosis_sk),
                treatment_sk               INTEGER    REFERENCES dim_treatment(treatment_sk),
                insurance_sk               INTEGER    REFERENCES dim_insurance(insurance_sk),
                age                        SMALLINT,
                comorbidities_count        SMALLINT,
                length_of_stay             SMALLINT,
                medications_count          SMALLINT,
                followup_visits_last_year  SMALLINT,
                prev_readmissions          SMALLINT,
                readmission_risk_score     NUMERIC(5,3),
                risk_label                 VARCHAR(10),
                discharge_disposition      VARCHAR(50),
                readmitted                 VARCHAR(3),
                label                      SMALLINT,
                etl_timestamp              TIMESTAMPTZ DEFAULT NOW()
            )"""
    }

    # Index pour accélérer les requêtes Power BI / analytiques
    INDEXES = [
        "CREATE INDEX IF NOT EXISTS idx_fact_patient  ON fact_admission(patient_sk)",
        "CREATE INDEX IF NOT EXISTS idx_fact_date     ON fact_admission(date_sk)",
        "CREATE INDEX IF NOT EXISTS idx_fact_diag     ON fact_admission(diagnosis_sk)",
        "CREATE INDEX IF NOT EXISTS idx_fact_risk     ON fact_admission(risk_label)",
        "CREATE INDEX IF NOT EXISTS idx_fact_label    ON fact_admission(label)",
        "CREATE INDEX IF NOT EXISTS idx_date_year     ON dim_date(year, month)",
    ]

    VIEW_SQL = """
        CREATE OR REPLACE VIEW v_readmission_kpis AS
        SELECT
            d.year,
            d.quarter,
            d.month_name,
            d.season,
            p.region,
            p.gender,
            p.age_group,
            di.diagnosis_name,
            t.treatment_name,
            i.insurance_type,
            f.discharge_disposition,
            f.risk_label,
            COUNT(*)                                  AS total_admissions,
            SUM(f.label)                              AS total_readmissions,
            ROUND(AVG(f.readmission_risk_score)::NUMERIC, 3) AS avg_risk_score,
            ROUND(AVG(f.length_of_stay)::NUMERIC, 1)         AS avg_los,
            ROUND(AVG(f.comorbidities_count)::NUMERIC, 1)    AS avg_comorbidities,
            ROUND(AVG(f.medications_count)::NUMERIC, 1)      AS avg_medications,
            ROUND(
                100.0 * SUM(f.label) / NULLIF(COUNT(*), 0),
                2
            )                                         AS readmission_rate_pct
        FROM fact_admission f
        JOIN dim_date      d  ON f.date_sk      = d.date_sk
        JOIN dim_patient   p  ON f.patient_sk   = p.patient_sk
        JOIN dim_diagnosis di ON f.diagnosis_sk = di.diagnosis_sk
        JOIN dim_treatment t  ON f.treatment_sk  = t.treatment_sk
        JOIN dim_insurance i  ON f.insurance_sk  = i.insurance_sk
        GROUP BY 1,2,3,4,5,6,7,8,9,10,11,12
    """

    def __init__(self, engine: Engine):
        self.engine = engine

    def run(self, tables: dict[str, pd.DataFrame]) -> None:
        order = ["dim_patient", "dim_date", "dim_diagnosis",
                 "dim_treatment", "dim_insurance", "fact_admission"]

        with self.engine.begin() as conn:
            # Créer les tables
            for tbl in order:
                conn.execute(text(self.DDL[tbl]))
                log.info(f"[LOAD] Table {tbl} creee / verifiee [OK]")

            # TRUNCATE en cascade pour full-refresh propre
            conn.execute(text(
                "TRUNCATE TABLE fact_admission, dim_patient, dim_date, "
                "dim_diagnosis, dim_treatment, dim_insurance RESTART IDENTITY CASCADE"
            ))
            log.info("[LOAD] Tables videes (TRUNCATE CASCADE) [OK]")

        # Chargement par chunks avec pandas + SQLAlchemy
        for tbl in order:
            df = tables[tbl]
            df.to_sql(tbl, self.engine, if_exists="append",
                      index=False, chunksize=500, method="multi")
            log.info(f"[LOAD] {tbl:<20} : {len(df):>6} lignes inserees [OK]")

        # Index + vue
        with self.engine.begin() as conn:
            for idx in self.INDEXES:
                conn.execute(text(idx))
            conn.execute(text(self.VIEW_SQL))
            log.info("[LOAD] Index + vue v_readmission_kpis crees [OK]")


# ─────────────────────────────────────────────────────────────
# ORCHESTRATEUR
# ─────────────────────────────────────────────────────────────
def test_connexion(engine) -> bool:
    """Teste la connexion avant de lancer l'ETL. Affiche un message clair si ca echoue."""
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        log.info(f"[CONNEXION] OK  ->  {PG_CONFIG['host']}:{PG_CONFIG['port']} / {PG_CONFIG['database']}")
        return True
    except Exception as e:
        log.error("=" * 60)
        log.error("[CONNEXION] ECHEC de connexion a PostgreSQL !")
        log.error(f"  Host     : {PG_CONFIG['host']}")
        log.error(f"  Port     : {PG_CONFIG['port']}")
        log.error(f"  Database : {PG_CONFIG['database']}")
        log.error(f"  User     : {PG_CONFIG['user']}")
        log.error("")
        log.error("Solutions possibles :")
        log.error("  1. La base n'existe pas -> creez-la dans pgAdmin :")
        log.error(f"       CREATE DATABASE {PG_CONFIG['database']};")
        log.error("  2. Mauvais mot de passe -> corrigez PG_CONFIG['password']")
        log.error("  3. Mauvais port -> verifiez dans pgAdmin > Properties")
        log.error(f"     (votre PostgreSQL tourne sur le port {PG_CONFIG['port']})")
        log.error(f"  Erreur brute : {e}")
        log.error("=" * 60)
        return False


def run_etl(src: str = SRC_FILE, db_url: str = DB_URL):
    start = datetime.now()
    log.info("=" * 60)
    log.info("  PIPELINE ETL Hospital DW  --  PostgreSQL")
    log.info("=" * 60)

    # -- 1. Connexion (teste avant tout) --------------------------
    engine = create_engine(
        db_url, echo=False,
        pool_pre_ping=True,
        connect_args={"connect_timeout": 10}
    )
    if not test_connexion(engine):
        return   # arret propre, message deja affiche

    # -- 2. ETL ---------------------------------------------------
    try:
        raw    = Extractor(src).run()
        tables = Transformer().run(raw)
        Loader(engine).run(tables)

        elapsed = (datetime.now() - start).total_seconds()
        log.info("")
        log.info(f"  ETL termine en {elapsed:.1f}s  [OK]")
        log.info("")

        for tbl in ["dim_patient", "dim_date", "dim_diagnosis",
                    "dim_treatment", "dim_insurance", "fact_admission"]:
            n = pd.read_sql(f"SELECT COUNT(*) AS n FROM {tbl}", engine).iloc[0, 0]
            log.info(f"   {tbl:<22} : {n:>6} lignes")

        engine.dispose()

    except Exception as e:
        log.error(f"[ERREUR] {e}", exc_info=True)
        raise


if __name__ == "__main__":
    run_etl()