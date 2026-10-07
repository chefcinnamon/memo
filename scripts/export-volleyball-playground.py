"""Export exact frequency counts for the real volleyball regression playground.

Run: python3 scripts/export-volleyball-playground.py
Optional: --database /path/to/volleyball.sqlite
Matching (X,Y) pairs are compressed; their counts retain all observations.
"""
import argparse
import collections
import json
import pathlib
import sqlite3


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=pathlib.Path,
                        default=pathlib.Path.home() / "Desktop/volleyball-projections/data/volleyball.sqlite")
    args = parser.parse_args()
    with sqlite3.connect(args.database.resolve().as_uri() + "?mode=ro", uri=True) as con:
        rows = con.execute("""
          SELECT substr(match_date,1,4) AS year, attack_attempts, attack_errors, COUNT(*)
          FROM player_match_stats
          WHERE sport='mvb' AND attack_attempts IS NOT NULL AND attack_errors IS NOT NULL
          GROUP BY year, attack_attempts, attack_errors
          ORDER BY year, attack_attempts, attack_errors
        """).fetchall()
        dates = con.execute("""
          SELECT MIN(match_date), MAX(match_date) FROM player_match_stats
          WHERE sport='mvb' AND attack_attempts IS NOT NULL AND attack_errors IS NOT NULL
        """).fetchone()
    by_year = collections.defaultdict(list)
    for year, x, y, n in rows:
        if x != int(x) or y != int(y):
            raise ValueError("Expected integer counts for attack attempts and attack errors")
        by_year[year].append([int(x), int(y), n])
    if not rows:
        raise ValueError("No complete men's player-match records found")
    data = {
        "description": "NCAA Division I men's volleyball: attack attempts and attack errors in the same match",
        "source": "https://jeffreyrstevens.github.io/ncaavolleyballr/articles/data.html",
        "first_match": dates[0], "last_match": dates[1],
        "years": [{"year": int(year), "points": points} for year, points in sorted(by_year.items())],
    }
    destination = pathlib.Path(__file__).resolve().parents[1] / "static/data/mvb-attack-errors.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(data, separators=(",", ":")) + "\n", encoding="utf-8")
    print(f"Exported {sum(n for _, _, _, n in rows):,} records in {len(rows):,} weighted pairs")


if __name__ == "__main__":
    main()
