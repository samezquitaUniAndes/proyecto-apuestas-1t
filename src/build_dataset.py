"""
build_dataset.py
------------------------------------------------------------------
Construye el dataset equipo-partido con métricas del PRIMER TIEMPO
a partir de StatsBomb Open Data (datos evento por evento).

Torneos incluidos (fútbol masculino, internacional, recientes,
todos con fase de grupos + eliminación):
    - Mundial Qatar 2022
    - Eurocopa 2024
    - Eurocopa 2020
    - Copa América 2024

Salida: data/processed/team_match_dataset.csv
Una fila por equipo-partido (2 filas por partido).
------------------------------------------------------------------
Nota: la posesión se aproxima por participación de pases en el 1er
tiempo (StatsBomb no entrega % de posesión directo en el evento).
"""
import os, requests
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd

BASE = "https://raw.githubusercontent.com/statsbomb/open-data/master/data"
ON_TARGET = {"Goal", "Saved", "Saved to Post"}          # remate al arco
SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "Mozilla/5.0"})

# (competition_name, gender, season_name)
TORNEOS = [
    ("FIFA World Cup", "male", "2022"),
    ("UEFA Euro",      "male", "2024"),
    ("UEFA Euro",      "male", "2020"),
    ("Copa America",   "male", "2024"),
]
NOMBRE = {
    ("FIFA World Cup", "2022"): "Mundial 2022",
    ("UEFA Euro", "2024"): "Euro 2024",
    ("UEFA Euro", "2020"): "Euro 2020",
    ("Copa America", "2024"): "Copa América 2024",
}


def get(url):
    r = SESSION.get(url, timeout=90)
    r.raise_for_status()
    return r.json()


def derive_match(m, tournament):
    """Devuelve 2 filas (una por equipo) con features del 1er tiempo."""
    mid = m["match_id"]
    home = m["home_team"]["home_team_name"]
    away = m["away_team"]["away_team_name"]
    stage = m.get("competition_stage", {}).get("name", "")
    fase = "Grupos" if stage == "Group Stage" else "Eliminación"

    ev = get(f"{BASE}/events/{mid}.json")
    T = {home: dict(pas=0, sh=0, sot=0, xg=0.0, gh=0, gf=0),
         away: dict(pas=0, sh=0, sot=0, xg=0.0, gh=0, gf=0)}

    for e in ev:
        per = e.get("period")
        tn = e.get("team", {}).get("name")
        ty = e.get("type", {}).get("name")
        if tn not in T:
            continue
        if ty == "Pass" and per == 1:
            T[tn]["pas"] += 1
        if ty == "Shot":
            sh = e.get("shot", {})
            out = sh.get("outcome", {}).get("name")
            xg = sh.get("statsbomb_xg", 0.0) or 0.0
            if per == 1:                       # solo primer tiempo
                T[tn]["sh"] += 1
                T[tn]["xg"] += xg
                if out in ON_TARGET:
                    T[tn]["sot"] += 1
                if out == "Goal":
                    T[tn]["gh"] += 1
            if per in (1, 2) and out == "Goal":  # marcador a 90'
                T[tn]["gf"] += 1

    tot = sum(T[t]["pas"] for t in T) or 1
    for t in T:
        T[t]["pos"] = round(100 * T[t]["pas"] / tot, 1)

    rows = []
    for t, opp in ((home, away), (away, home)):
        a, b = T[t], T[opp]
        res = "W" if a["gf"] > b["gf"] else ("L" if a["gf"] < b["gf"] else "D")
        ht = "Lead" if a["gh"] > b["gh"] else ("Trail" if a["gh"] < b["gh"] else "Level")
        rows.append(dict(
            torneo=tournament, fase=fase, match_id=mid, equipo=t, rival=opp,
            pos_h1=a["pos"], rem_h1=a["sh"], arco_h1=a["sot"],
            xg_h1=round(a["xg"], 3), goles_h1=a["gh"],
            pos_diff=round(a["pos"] - b["pos"], 1),
            rem_diff=a["sh"] - b["sh"], arco_diff=a["sot"] - b["sot"],
            xg_diff=round(a["xg"] - b["xg"], 3),
            estado_ht=ht, resultado_ft=res,
        ))
    return rows


def main():
    comps = get(f"{BASE}/competitions.json")
    sel = []
    for name, gen, seas in TORNEOS:
        for c in comps:
            if (c["competition_name"] == name and c.get("competition_gender") == gen
                    and c["season_name"] == seas):
                sel.append((NOMBRE[(name, seas)], c["competition_id"], c["season_id"]))
                break

    all_matches = []
    for tname, cid, sid in sel:
        ms = get(f"{BASE}/matches/{cid}/{sid}.json")
        for m in ms:
            all_matches.append((tname, m))
        print(f"{tname}: {len(ms)} partidos")
    print(f"Total partidos: {len(all_matches)} | descargando eventos...")

    rows, fail = [], 0
    with ThreadPoolExecutor(max_workers=10) as ex:
        futs = {ex.submit(derive_match, m, tn): tn for tn, m in all_matches}
        done = 0
        for f in as_completed(futs):
            try:
                rows.extend(f.result())
            except Exception:
                fail += 1
            done += 1
            if done % 40 == 0:
                print(f"  {done}/{len(all_matches)}")

    df = pd.DataFrame(rows)
    os.makedirs("data/processed", exist_ok=True)
    df.to_csv("data/processed/team_match_dataset.csv", index=False)
    print(f"OK -> data/processed/team_match_dataset.csv | filas={len(df)} | partidos_fallidos={fail}")


if __name__ == "__main__":
    main()
