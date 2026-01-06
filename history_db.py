# history_db.py
from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional


def db_path() -> Path:
    """
    Renvoie le chemin du fichier SQLite utilisé pour historiser les runs.

    Choix volontaire :
    - DB locale, simple, sans dépendance externe (pas besoin de serveur).
    - Placée à côté de app.py / ui_common.py pour rester "portable" (copier-coller du projet).
    """
    return Path(__file__).resolve().parent / "frtb_history.sqlite3"


def _connect() -> sqlite3.Connection:
    """
    Ouvre une connexion SQLite configurée pour l'usage Streamlit.

    - check_same_thread=False :
        Streamlit peut recharger/rerun le script et manipuler la DB depuis différents contextes.
        Cela évite certaines erreurs "SQLite objects created in a thread...".
        (On reste prudent : on ouvre/ferme vite les connexions.)

    - row_factory=sqlite3.Row :
        Permet de récupérer des lignes sous forme "dict-like" (accès par nom de colonne),
        très pratique pour transformer ensuite en dict/JSON/DataFrame.
    """
    p = db_path()
    con = sqlite3.connect(str(p), check_same_thread=False)
    con.row_factory = sqlite3.Row
    return con


def init_db() -> None:
    """
    Initialise le schéma de la base (idempotent).
    """
    con = _connect()
    try:
        # Création de la table si elle n'existe pas déjà
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS runs (
                id                INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at_utc     TEXT NOT NULL,
                created_at_local   TEXT NOT NULL,
                status            TEXT NOT NULL,            -- "ok" / "error"
                k_eq              REAL,
                k_sw              REAL,
                k_bo              REAL,
                k_total           REAL,
                portfolio_csv     TEXT,
                snapshot_json     TEXT,
                results_json      TEXT,
                logs_txt          TEXT,
                meta_json         TEXT,
                error_txt         TEXT
            );
            """
        )

        # Index pour accélérer la navigation dans l'historique
        con.execute("CREATE INDEX IF NOT EXISTS idx_runs_created_at ON runs(created_at_utc);")
        con.execute("CREATE INDEX IF NOT EXISTS idx_runs_status ON runs(status);")

        con.commit()
    finally:
        # Toujours fermer la connexion, même en cas d'erreur
        con.close()


def insert_run(rec: Dict[str, Any]) -> int:
    """
    Insère un run dans la table runs et renvoie l'id créé.
    """
    init_db()
    con = _connect()
    try:
        cols = [
            "created_at_utc",
            "created_at_local",
            "status",
            "k_eq",
            "k_sw",
            "k_bo",
            "k_total",
            "portfolio_csv",
            "snapshot_json",
            "results_json",
            "logs_txt",
            "meta_json",
            "error_txt",
        ]

        vals = [rec.get(c) for c in cols]

        q = f"INSERT INTO runs ({','.join(cols)}) VALUES ({','.join(['?']*len(cols))})"
        cur = con.execute(q, vals)
        con.commit()

        return int(cur.lastrowid)
    finally:
        con.close()


def list_runs(limit: int = 200, status: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Liste les runs (vue "compacte") pour alimenter l'UI Historique.
    - limit : nombre max de runs renvoyés
    - status : None (tous) ou "ok" / "error" pour filtrer
    """
    init_db()
    con = _connect()
    try:
        # Filtre optionnel sur le statut
        if status in ("ok", "error"):
            cur = con.execute(
                """
                SELECT id, created_at_local, created_at_utc, status, k_eq, k_sw, k_bo, k_total
                FROM runs
                WHERE status = ?
                ORDER BY id DESC
                LIMIT ?
                """,
                (status, int(limit)),
            )
        else:
            cur = con.execute(
                """
                SELECT id, created_at_local, created_at_utc, status, k_eq, k_sw, k_bo, k_total
                FROM runs
                ORDER BY id DESC
                LIMIT ?
                """,
                (int(limit),),
            )

        # sqlite3.Row -> dict pour faciliter pandas / json / streamlit
        return [dict(r) for r in cur.fetchall()]
    finally:
        con.close()


def get_run(run_id: int) -> Optional[Dict[str, Any]]:
    """
    Renvoie le détail complet d'un run (toutes les colonnes), ou None si introuvable.
    """
    init_db()
    con = _connect()
    try:
        cur = con.execute("SELECT * FROM runs WHERE id = ?", (int(run_id),))
        row = cur.fetchone()
        return dict(row) if row else None
    finally:
        con.close()


def delete_run(run_id: int) -> None:
    """
    Supprime un run spécifique (par id).
    """
    init_db()
    con = _connect()
    try:
        con.execute("DELETE FROM runs WHERE id = ?", (int(run_id),))
        con.commit()
    finally:
        con.close()


def clear_all() -> None:
    """
    Supprime tout l'historique (tous les runs).
    """
    init_db()
    con = _connect()
    try:
        con.execute("DELETE FROM runs")
        con.commit()
    finally:
        con.close()
