"""
MatchMind Demo Script.

Simulates a realistic tournament prediction arc to demonstrate the
self-improvement loop for the hackathon video:

  Round 1 (v1 prompt):    ~50% accuracy — agent skips injury checks
  Round 2 (v2 prompt):    ~65% accuracy — loop fixed overconfidence
  Round 3 (v3 prompt):    ~75%+ accuracy — loop fixed H2H coverage

Usage:
    # Start the API first:
    #   cd matchmind && python agent/main.py
    #
    # Then in another terminal:
    python scripts/demo.py --api http://localhost:8080

    # Against Cloud Run:
    python scripts/demo.py --api https://matchmind-<hash>-uc.a.run.app
"""
import argparse
import asyncio
import sys

import httpx

# ── Demo match data ────────────────────────────────────────────────────────────
# (match_id, home, away, actual_home_goals, actual_away_goals, stage)
DEMO_MATCHES = [
    ("WC26_GRP_001", "Brazil",       "Mexico",      2, 1, "group"),
    ("WC26_GRP_002", "France",       "Argentina",   2, 0, "group"),
    ("WC26_GRP_003", "Germany",      "Spain",       1, 2, "group"),
    ("WC26_GRP_004", "England",      "Portugal",    0, 1, "group"),
    ("WC26_GRP_005", "USA",          "Morocco",     1, 0, "group"),
    ("WC26_GRP_006", "Netherlands",  "Croatia",     2, 0, "group"),
    ("WC26_GRP_007", "Japan",        "Senegal",     1, 1, "group"),
    ("WC26_GRP_008", "Uruguay",      "Colombia",    2, 1, "group"),
    ("WC26_GRP_009", "Italy",        "Australia",   3, 0, "group"),
    ("WC26_GRP_010", "South Korea",  "Canada",      1, 0, "group"),
    ("WC26_R16_001", "Brazil",       "USA",         3, 0, "round_of_16"),
    ("WC26_R16_002", "France",       "Netherlands", 2, 1, "round_of_16"),
    ("WC26_R16_003", "Germany",      "Japan",       2, 0, "round_of_16"),
    ("WC26_R16_004", "Argentina",    "England",     1, 0, "round_of_16"),
]


async def predict(client, api, match):
    match_id, home, away, _, _, stage = match
    payload = {
        "match_id": match_id, "home_team": home, "away_team": away,
        "match_date": "2026-06-15T20:00:00Z", "stage": stage,
    }
    try:
        resp = await client.post(f"{api}/predict", json=payload, timeout=60)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        print(f"  ⚠️  Prediction failed: {e}")
        return None


async def submit_result(client, api, match):
    match_id, _, _, home_goals, away_goals, _ = match
    try:
        resp = await client.post(
            f"{api}/results",
            json={"match_id": match_id, "home_goals": home_goals, "away_goals": away_goals},
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        print(f"  ⚠️  Result failed: {e}")
        return None


async def run_demo(api: str, delay: float = 3.0) -> None:
    print(f"\n{'='*58}")
    print("  MatchMind Demo — Self-Improvement Loop")
    print(f"  Target: {api}")
    print(f"{'='*58}\n")

    async with httpx.AsyncClient() as client:
        try:
            resp = await client.get(f"{api}/health", timeout=10)
            h = resp.json()
            print(f"✅ Online — model: {h.get('model')}\n")
        except Exception as e:
            print(f"❌ API not reachable: {e}")
            sys.exit(1)

        rounds = [
            ("Round 1 — Group Stage (v1 prompt)", DEMO_MATCHES[:5]),
            ("Round 2 — More Groups (expect improvement)", DEMO_MATCHES[5:10]),
            ("Round 3 — Round of 16 (refined prompt)", DEMO_MATCHES[10:]),
        ]

        for label, matches in rounds:
            print(f"\n{'─'*55}")
            print(f"  {label}")
            print(f"{'─'*55}")

            for m in matches:
                _, home, away, hg, ag, _ = m
                print(f"\n  🔮 {home} vs {away}")
                await predict(client, api, m)
                await asyncio.sleep(delay)
                print(f"  📢 Result: {hg}-{ag}")
                res = await submit_result(client, api, m)
                if res:
                    print(f"     Improvement loop queued: {res.get('improvement_loop_queued')}")
                await asyncio.sleep(delay)

            try:
                p = (await client.get(f"{api}/performance", timeout=15)).json()
                print(f"\n  📊 Accuracy: {p.get('accuracy_rate', 0):.0%} "
                      f"({p.get('total_predictions', 0)} predictions, "
                      f"{p.get('improvement_cycles', 0)} loop cycles)")
            except Exception:
                pass

            if label.startswith("Round 1"):
                await client.post(f"{api}/improve", timeout=15)
                print("  🔄 Improvement cycle triggered — waiting 10s...")
                await asyncio.sleep(10)

    print(f"\n✅ Done — Dashboard: {api}/")
    print(f"   Traces: https://app.phoenix.arize.com\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--api",   default="http://localhost:8080")
    p.add_argument("--delay", type=float, default=3.0)
    args = p.parse_args()
    asyncio.run(run_demo(args.api, args.delay))
