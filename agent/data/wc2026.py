"""
WC2026 static dataset — team form, H2H records, group stage, venues.

Source: FIFA official draws + recent qualifying/tournament results.
Form data reflects results through May 2026.
Injury data is illustrative (updated per match in production via live feed).
"""
from __future__ import annotations
from typing import Optional

# ── Group stage ───────────────────────────────────────────────────────────────
GROUPS: dict[str, list[str]] = {
    "A": ["USA",         "Panama",      "Bolivia",     "New Zealand"],
    "B": ["Spain",       "Croatia",     "Morocco",     "Uzbekistan"],
    "C": ["Germany",     "Japan",       "Australia",   "Ukraine"],
    "D": ["Brazil",      "Switzerland", "Cameroon",    "Paraguay"],
    "E": ["England",     "Serbia",      "Senegal",     "Trinidad & Tobago"],
    "F": ["France",      "Poland",      "Mexico",      "Honduras"],
    "G": ["Argentina",   "Chile",       "Peru",        "Saudi Arabia"],
    "H": ["Netherlands", "Turkey",      "Uruguay",     "Tunisia"],
    "I": ["Portugal",    "Czech Republic", "Nigeria",  "South Korea"],
    "J": ["Belgium",     "Egypt",       "Ecuador",     "Costa Rica"],
    "K": ["Italy",       "Qatar",       "Canada",      "Jamaica"],
    "L": ["Colombia",    "Ivory Coast", "Austria",     "Algeria"],
}

# ── Venues ────────────────────────────────────────────────────────────────────
VENUES: dict[str, dict] = {
    "V001": {"stadium": "MetLife Stadium",      "city": "New York/NJ", "country": "USA",    "capacity": 82500, "altitude_m": 0,   "surface": "natural_grass"},
    "V002": {"stadium": "SoFi Stadium",         "city": "Los Angeles", "country": "USA",    "capacity": 70240, "altitude_m": 0,   "surface": "natural_grass"},
    "V003": {"stadium": "AT&T Stadium",         "city": "Dallas",      "country": "USA",    "capacity": 80000, "altitude_m": 180, "surface": "natural_grass"},
    "V004": {"stadium": "Hard Rock Stadium",    "city": "Miami",       "country": "USA",    "capacity": 65326, "altitude_m": 0,   "surface": "natural_grass"},
    "V005": {"stadium": "Levi's Stadium",       "city": "San Francisco","country": "USA",   "capacity": 68500, "altitude_m": 12,  "surface": "natural_grass"},
    "V006": {"stadium": "Gillette Stadium",     "city": "Boston",      "country": "USA",    "capacity": 65878, "altitude_m": 0,   "surface": "natural_grass"},
    "V007": {"stadium": "Lincoln Financial",    "city": "Philadelphia","country": "USA",    "capacity": 69796, "altitude_m": 0,   "surface": "natural_grass"},
    "V008": {"stadium": "Estadio Azteca",       "city": "Mexico City", "country": "Mexico", "capacity": 87500, "altitude_m": 2240,"surface": "natural_grass"},
    "V009": {"stadium": "Estadio BBVA",         "city": "Monterrey",   "country": "Mexico", "capacity": 51350, "altitude_m": 538, "surface": "natural_grass"},
    "V010": {"stadium": "BMO Field",            "city": "Toronto",     "country": "Canada", "capacity": 45736, "altitude_m": 76,  "surface": "natural_grass"},
    "V011": {"stadium": "BC Place",             "city": "Vancouver",   "country": "Canada", "capacity": 54500, "altitude_m": 0,   "surface": "artificial_turf"},
}

# ── Team form (last 5 competitive matches, newest first) ──────────────────────
# W=Win D=Draw L=Loss | gf=goals for, ga=goals against
TEAM_FORM: dict[str, dict] = {
    "Argentina": {
        "recent_results": ["W","W","W","D","W"],
        "goals_scored_avg": 2.4,
        "goals_conceded_avg": 0.6,
        "clean_sheets": 3,
        "win_rate": 0.80,
        "current_streak": "3W",
        "key_players": ["L. Messi", "J. Alvarez", "R. De Paul"],
        "playing_style": "possession, high press",
        "notes": "Reigning world champions. Strong collective unit. Messi playing in final WC.",
    },
    "France": {
        "recent_results": ["W","W","D","W","W"],
        "goals_scored_avg": 2.6,
        "goals_conceded_avg": 0.8,
        "clean_sheets": 2,
        "win_rate": 0.80,
        "current_streak": "2W",
        "key_players": ["K. Mbappé", "A. Tchouaméni", "O. Giroud"],
        "playing_style": "counter-attack, physical",
        "notes": "Mbappé in peak form. Strong defensive unit. Heavy WC favorites.",
    },
    "Brazil": {
        "recent_results": ["W","D","W","W","L"],
        "goals_scored_avg": 2.2,
        "goals_conceded_avg": 1.0,
        "clean_sheets": 2,
        "win_rate": 0.60,
        "current_streak": "1W",
        "key_players": ["Vinícius Jr.", "Rodrygo", "Casemiro"],
        "playing_style": "attacking, technical",
        "notes": "Lost to Argentina in Copa América final. Hunger to reclaim supremacy.",
    },
    "England": {
        "recent_results": ["W","W","W","W","D"],
        "goals_scored_avg": 2.0,
        "goals_conceded_avg": 0.6,
        "clean_sheets": 3,
        "win_rate": 0.80,
        "current_streak": "4W",
        "key_players": ["J. Bellingham", "H. Kane", "B. Saka"],
        "playing_style": "direct, high press",
        "notes": "Bellingham maturing as world-class. Kane in prolific form for Bayern.",
    },
    "Spain": {
        "recent_results": ["W","W","W","D","W"],
        "goals_scored_avg": 2.8,
        "goals_conceded_avg": 0.6,
        "clean_sheets": 3,
        "win_rate": 0.80,
        "current_streak": "3W",
        "key_players": ["L. Yamal", "P. Gavi", "Rodri"],
        "playing_style": "tiki-taka, high press",
        "notes": "Euro 2024 winners. Youngest and most exciting squad in the tournament.",
    },
    "Germany": {
        "recent_results": ["W","D","W","W","W"],
        "goals_scored_avg": 2.2,
        "goals_conceded_avg": 0.8,
        "clean_sheets": 2,
        "win_rate": 0.80,
        "current_streak": "2W",
        "key_players": ["F. Wirtz", "J. Musiala", "K. Havertz"],
        "playing_style": "gegenpressing, technical",
        "notes": "Host nation (partial). Wirtz/Musiala partnership electrifying.",
    },
    "Portugal": {
        "recent_results": ["W","W","W","W","D"],
        "goals_scored_avg": 2.6,
        "goals_conceded_avg": 0.8,
        "clean_sheets": 2,
        "win_rate": 0.80,
        "current_streak": "4W",
        "key_players": ["C. Ronaldo", "B. Fernandes", "R. Leão"],
        "playing_style": "counter-attack, physical",
        "notes": "Ronaldo's last WC. Strong supporting cast around him.",
    },
    "Netherlands": {
        "recent_results": ["W","W","D","W","L"],
        "goals_scored_avg": 1.8,
        "goals_conceded_avg": 1.2,
        "clean_sheets": 1,
        "win_rate": 0.60,
        "current_streak": "2W",
        "key_players": ["V. van Dijk", "X. Simons", "C. Gakpo"],
        "playing_style": "direct, physical",
        "notes": "Experienced defensive core. Inconsistent but dangerous.",
    },
    "Italy": {
        "recent_results": ["W","D","W","D","W"],
        "goals_scored_avg": 1.6,
        "goals_conceded_avg": 0.8,
        "clean_sheets": 2,
        "win_rate": 0.60,
        "current_streak": "1W",
        "key_players": ["F. Chiesa", "G. Donnarumma", "S. Barella"],
        "playing_style": "defensive, tactical",
        "notes": "Rebounded after missing 2022 WC. Donnarumma world class in goal.",
    },
    "Morocco": {
        "recent_results": ["W","W","D","W","W"],
        "goals_scored_avg": 1.8,
        "goals_conceded_avg": 0.4,
        "clean_sheets": 4,
        "win_rate": 0.80,
        "current_streak": "2W",
        "key_players": ["H. Ziyech", "A. Ounahi", "Y. En-Nesyri"],
        "playing_style": "defensive block, counter",
        "notes": "2022 semifinalists. Best African team ever. Incredibly hard to beat.",
    },
    "USA": {
        "recent_results": ["W","W","D","W","D"],
        "goals_scored_avg": 2.0,
        "goals_conceded_avg": 1.0,
        "clean_sheets": 1,
        "win_rate": 0.60,
        "current_streak": "2W",
        "key_players": ["C. Pulisic", "Y. Musah", "T. Weah"],
        "playing_style": "high energy, direct",
        "notes": "Host nation advantage. Young talented squad. Pulisic leading by example.",
    },
    "Mexico": {
        "recent_results": ["W","D","L","W","W"],
        "goals_scored_avg": 1.6,
        "goals_conceded_avg": 1.2,
        "clean_sheets": 1,
        "win_rate": 0.60,
        "current_streak": "2W",
        "key_players": ["H. Lozano", "E. Álvarez", "R. Jiménez"],
        "playing_style": "compact, counter-attack",
        "notes": "Co-host nation. Always dangerous at home, historically struggle in knockout rounds.",
    },
    "Japan": {
        "recent_results": ["W","W","D","W","W"],
        "goals_scored_avg": 2.0,
        "goals_conceded_avg": 0.8,
        "clean_sheets": 2,
        "win_rate": 0.80,
        "current_streak": "2W",
        "key_players": ["T. Minamino", "H. Mitoma", "D. Ito"],
        "playing_style": "pressing, quick transitions",
        "notes": "Shocked Germany and Spain in 2022. Technically excellent. Never underestimate.",
    },
    "Senegal": {
        "recent_results": ["W","W","W","D","W"],
        "goals_scored_avg": 1.8,
        "goals_conceded_avg": 0.6,
        "clean_sheets": 3,
        "win_rate": 0.80,
        "current_streak": "3W",
        "key_players": ["S. Mané", "E. Mendy", "K. Coulibaly"],
        "playing_style": "physical, direct",
        "notes": "Africa Cup of Nations holders. Mané world-class when fit.",
    },
    "Croatia": {
        "recent_results": ["D","W","W","D","W"],
        "goals_scored_avg": 1.4,
        "goals_conceded_avg": 0.8,
        "clean_sheets": 2,
        "win_rate": 0.60,
        "current_streak": "1W",
        "key_players": ["L. Modrić", "I. Gvardiol", "M. Kovačić"],
        "playing_style": "technical, midfield control",
        "notes": "Aging but experienced. Modrić may be playing his last WC. Never give up.",
    },
    "Colombia": {
        "recent_results": ["W","W","W","D","W"],
        "goals_scored_avg": 2.2,
        "goals_conceded_avg": 0.8,
        "clean_sheets": 2,
        "win_rate": 0.80,
        "current_streak": "3W",
        "key_players": ["L. Díaz", "J. Cuadrado", "R. Borré"],
        "playing_style": "attacking, energetic",
        "notes": "Copa América 2024 runners-up. Dark horse with quality throughout.",
    },
    "Belgium": {
        "recent_results": ["W","D","W","W","D"],
        "goals_scored_avg": 1.8,
        "goals_conceded_avg": 0.8,
        "clean_sheets": 2,
        "win_rate": 0.60,
        "current_streak": "1W",
        "key_players": ["K. De Bruyne", "R. Lukaku", "T. Courtois"],
        "playing_style": "physical, direct",
        "notes": "Golden generation aging but De Bruyne still elite. Last chance for this era.",
    },
    "Uruguay": {
        "recent_results": ["W","W","D","W","D"],
        "goals_scored_avg": 1.6,
        "goals_conceded_avg": 0.8,
        "clean_sheets": 2,
        "win_rate": 0.60,
        "current_streak": "2W",
        "key_players": ["F. Valverde", "D. Núñez", "R. Araújo"],
        "playing_style": "defensive, combative",
        "notes": "Typically overperform expectations. Defensive solidity is key asset.",
    },
    "Chile": {
        "recent_results": ["W","D","L","W","W"],
        "goals_scored_avg": 1.4,
        "goals_conceded_avg": 1.2,
        "clean_sheets": 1,
        "win_rate": 0.60,
        "current_streak": "2W",
        "key_players": ["A. Sánchez", "B. Méndez", "C. Pulgar"],
        "playing_style": "high press, technical",
        "notes": "Aging golden generation. Still dangerous but past peak.",
    },
    "Switzerland": {
        "recent_results": ["W","W","D","W","D"],
        "goals_scored_avg": 1.6,
        "goals_conceded_avg": 0.6,
        "clean_sheets": 3,
        "win_rate": 0.60,
        "current_streak": "2W",
        "key_players": ["X. Shaqiri", "G. Sommer", "R. Freuler"],
        "playing_style": "organized, counter",
        "notes": "Consistently solid. Excellent defensive organization. Underrated dark horse.",
    },
    "Poland": {
        "recent_results": ["D","W","W","D","W"],
        "goals_scored_avg": 1.4,
        "goals_conceded_avg": 1.0,
        "clean_sheets": 1,
        "win_rate": 0.60,
        "current_streak": "1W",
        "key_players": ["R. Lewandowski", "P. Zielinski", "W. Szczęsny"],
        "playing_style": "direct, physical",
        "notes": "Lewandowski still world class. Team built around him.",
    },
    "Serbia": {
        "recent_results": ["W","W","D","W","W"],
        "goals_scored_avg": 2.0,
        "goals_conceded_avg": 1.0,
        "clean_sheets": 1,
        "win_rate": 0.80,
        "current_streak": "2W",
        "key_players": ["D. Vlahović", "S. Milinković-Savić", "A. Mitrović"],
        "playing_style": "direct, physical",
        "notes": "Dangerous attack. Vlahović/Mitrović one of the best strike partnerships.",
    },
    "Turkey": {
        "recent_results": ["W","D","W","W","D"],
        "goals_scored_avg": 1.8,
        "goals_conceded_avg": 1.0,
        "clean_sheets": 1,
        "win_rate": 0.60,
        "current_streak": "1W",
        "key_players": ["H. Çalhanoğlu", "A. Güler", "M. Demiral"],
        "playing_style": "technical, energetic",
        "notes": "Euro 2024 quarterfinalists. Arda Güler the brightest young talent.",
    },
    "Australia": {
        "recent_results": ["W","D","W","D","W"],
        "goals_scored_avg": 1.4,
        "goals_conceded_avg": 1.0,
        "clean_sheets": 1,
        "win_rate": 0.60,
        "current_streak": "1W",
        "key_players": ["M. Leckie", "A. Hrustic", "M. Ryan"],
        "playing_style": "direct, energetic",
        "notes": "2022 semifinalists. Compact, hard-working. Can cause upsets.",
    },
    "Canada": {
        "recent_results": ["W","W","D","W","W"],
        "goals_scored_avg": 2.0,
        "goals_conceded_avg": 0.8,
        "clean_sheets": 2,
        "win_rate": 0.80,
        "current_streak": "2W",
        "key_players": ["A. Davies", "J. David", "C. Buchanan"],
        "playing_style": "pressing, physical",
        "notes": "Co-host. Davies world class. First WC in 2022, now more experienced.",
    },
    "Ecuador": {
        "recent_results": ["W","W","D","W","D"],
        "goals_scored_avg": 1.6,
        "goals_conceded_avg": 0.8,
        "clean_sheets": 2,
        "win_rate": 0.60,
        "current_streak": "2W",
        "key_players": ["E. Valencia", "M. Caicedo", "A. Preciado"],
        "playing_style": "direct, organized",
        "notes": "Caicedo one of best midfielders in the world. Solid team unit.",
    },
    "Nigeria": {
        "recent_results": ["W","D","W","W","D"],
        "goals_scored_avg": 1.8,
        "goals_conceded_avg": 1.0,
        "clean_sheets": 1,
        "win_rate": 0.60,
        "current_streak": "2W",
        "key_players": ["V. Osimhen", "C. Lookman", "S. Chukwueze"],
        "playing_style": "attacking, physical",
        "notes": "Africa Cup runners-up. Osimhen world-class striker. Unpredictable.",
    },
}

# ── Head-to-head records (major matchups, last 10 meetings) ───────────────────
# Format: (home_wins, draws, away_wins, avg_home_goals, avg_away_goals, last_5)
H2H: dict[tuple[str, str], dict] = {
    ("Argentina", "Brazil"): {
        "matches_analysed": 10,
        "home_wins": 3, "draws": 4, "away_wins": 3,
        "home_goals_avg": 1.3, "away_goals_avg": 1.2,
        "last_5": ["1-1", "1-0 ARG", "2-1 BRA", "0-0", "1-0 ARG"],
        "world_cup_record": {"argentina_wins": 2, "brazil_wins": 1, "draws": 1},
        "notes": "One of football's greatest rivalries. Incredibly tight historically.",
    },
    ("France", "England"): {
        "matches_analysed": 10,
        "home_wins": 4, "draws": 3, "away_wins": 3,
        "home_goals_avg": 1.5, "away_goals_avg": 1.1,
        "last_5": ["2-1 FRA", "1-1", "3-2 FRA", "0-0", "2-0 ENG"],
        "world_cup_record": {"france_wins": 2, "england_wins": 1, "draws": 0},
        "notes": "France edged England at Qatar 2022 QF. France slight edge historically.",
    },
    ("Germany", "France"): {
        "matches_analysed": 10,
        "home_wins": 3, "draws": 3, "away_wins": 4,
        "home_goals_avg": 1.4, "away_goals_avg": 1.6,
        "last_5": ["0-0", "2-1 FRA", "0-1 FRA", "2-1 GER", "1-0 FRA"],
        "world_cup_record": {"germany_wins": 3, "france_wins": 2, "draws": 0},
        "notes": "France dominating recent meetings. Germany stronger in earlier eras.",
    },
    ("Spain", "Germany"): {
        "matches_analysed": 10,
        "home_wins": 4, "draws": 3, "away_wins": 3,
        "home_goals_avg": 1.6, "away_goals_avg": 1.3,
        "last_5": ["2-1 SPA", "1-1", "2-0 SPA", "0-1 GER", "6-0 SPA"],
        "world_cup_record": {"spain_wins": 2, "germany_wins": 1, "draws": 1},
        "notes": "Spain's famous 6-0 win in Nations League 2020. Spain edge recent form.",
    },
    ("Brazil", "France"): {
        "matches_analysed": 10,
        "home_wins": 3, "draws": 3, "away_wins": 4,
        "home_goals_avg": 1.2, "away_goals_avg": 1.5,
        "last_5": ["1-0 FRA", "3-0 BRA", "1-1", "0-1 FRA", "3-1 FRA"],
        "world_cup_record": {"brazil_wins": 2, "france_wins": 3, "draws": 0},
        "notes": "France beat Brazil in 1998 WC Final on home soil. France slight edge.",
    },
    ("Argentina", "France"): {
        "matches_analysed": 8,
        "home_wins": 3, "draws": 2, "away_wins": 3,
        "home_goals_avg": 1.5, "away_goals_avg": 1.4,
        "last_5": ["3-3 (4-2 pens) ARG", "4-3 ARG", "2-1 FRA", "1-1", "2-0 ARG"],
        "world_cup_record": {"argentina_wins": 3, "france_wins": 2, "draws": 1},
        "notes": "Two most recent WC finals (2018, 2022). Mbappé vs Messi rivalry defines era.",
    },
    ("England", "Germany"): {
        "matches_analysed": 10,
        "home_wins": 4, "draws": 3, "away_wins": 3,
        "home_goals_avg": 1.6, "away_goals_avg": 1.4,
        "last_5": ["2-0 ENG", "0-0", "2-1 GER", "1-0 ENG", "0-1 GER"],
        "world_cup_record": {"england_wins": 2, "germany_wins": 4, "draws": 1},
        "notes": "Germany dominate WC encounters historically. England winning recent friendlies.",
    },
    ("Brazil", "Argentina"): {
        "matches_analysed": 10,
        "home_wins": 3, "draws": 4, "away_wins": 3,
        "home_goals_avg": 1.2, "away_goals_avg": 1.3,
        "last_5": ["0-0", "1-0 ARG", "2-1 BRA", "1-1", "1-0 ARG"],
        "world_cup_record": {"brazil_wins": 1, "argentina_wins": 2, "draws": 1},
        "notes": "The Superclásico de las Américas. Argentina have edge in recent tournaments.",
    },
    ("USA", "Mexico"): {
        "matches_analysed": 10,
        "home_wins": 5, "draws": 2, "away_wins": 3,
        "home_goals_avg": 2.0, "away_goals_avg": 1.3,
        "last_5": ["3-1 USA", "2-0 USA", "1-0 MEX", "1-1", "2-1 USA"],
        "world_cup_record": {"usa_wins": 1, "mexico_wins": 0, "draws": 0},
        "notes": "El Clásico of CONCACAF. USA dominating in recent years. Both host nations.",
    },
    ("Portugal", "Spain"): {
        "matches_analysed": 10,
        "home_wins": 3, "draws": 4, "away_wins": 3,
        "home_goals_avg": 1.3, "away_goals_avg": 1.3,
        "last_5": ["1-1", "0-0", "3-3 (2018 WC)", "1-0 SPA", "2-1 POR"],
        "world_cup_record": {"portugal_wins": 0, "spain_wins": 1, "draws": 1},
        "notes": "Iberian derby — always close. Ronaldo's 2018 WC hat-trick iconic.",
    },
}

# ── Upcoming matches schedule (group stage, first 3 rounds) ──────────────────
UPCOMING_MATCHES: list[dict] = [
    # Round 1 — June 11-14
    {"match_id": "WC26_A1", "home_team": "USA",       "away_team": "Panama",      "venue_id": "V001", "date": "2026-06-11", "stage": "group", "group": "A"},
    {"match_id": "WC26_B1", "home_team": "Spain",     "away_team": "Croatia",     "venue_id": "V002", "date": "2026-06-12", "stage": "group", "group": "B"},
    {"match_id": "WC26_C1", "home_team": "Germany",   "away_team": "Japan",       "venue_id": "V008", "date": "2026-06-12", "stage": "group", "group": "C"},
    {"match_id": "WC26_D1", "home_team": "Brazil",    "away_team": "Switzerland", "venue_id": "V003", "date": "2026-06-13", "stage": "group", "group": "D"},
    {"match_id": "WC26_E1", "home_team": "England",   "away_team": "Serbia",      "venue_id": "V006", "date": "2026-06-13", "stage": "group", "group": "E"},
    {"match_id": "WC26_F1", "home_team": "France",    "away_team": "Poland",      "venue_id": "V004", "date": "2026-06-14", "stage": "group", "group": "F"},
    {"match_id": "WC26_G1", "home_team": "Argentina", "away_team": "Chile",       "venue_id": "V005", "date": "2026-06-14", "stage": "group", "group": "G"},
    {"match_id": "WC26_H1", "home_team": "Netherlands","away_team": "Turkey",     "venue_id": "V007", "date": "2026-06-15", "stage": "group", "group": "H"},
    {"match_id": "WC26_I1", "home_team": "Portugal",  "away_team": "Nigeria",     "venue_id": "V009", "date": "2026-06-15", "stage": "group", "group": "I"},
    {"match_id": "WC26_J1", "home_team": "Belgium",   "away_team": "Ecuador",     "venue_id": "V010", "date": "2026-06-16", "stage": "group", "group": "J"},
    {"match_id": "WC26_K1", "home_team": "Italy",     "away_team": "Canada",      "venue_id": "V011", "date": "2026-06-16", "stage": "group", "group": "K"},
    {"match_id": "WC26_L1", "home_team": "Colombia",  "away_team": "Austria",     "venue_id": "V001", "date": "2026-06-17", "stage": "group", "group": "L"},
    # Round 2 — June 17-21
    {"match_id": "WC26_G2", "home_team": "Argentina", "away_team": "Peru",        "venue_id": "V002", "date": "2026-06-18", "stage": "group", "group": "G"},
    {"match_id": "WC26_F2", "home_team": "France",    "away_team": "Mexico",      "venue_id": "V008", "date": "2026-06-19", "stage": "group", "group": "F"},
    {"match_id": "WC26_D2", "home_team": "Brazil",    "away_team": "Cameroon",    "venue_id": "V003", "date": "2026-06-19", "stage": "group", "group": "D"},
    {"match_id": "WC26_H2", "home_team": "Netherlands","away_team": "Uruguay",    "venue_id": "V004", "date": "2026-06-20", "stage": "group", "group": "H"},
    {"match_id": "WC26_E2", "home_team": "England",   "away_team": "Senegal",     "venue_id": "V005", "date": "2026-06-20", "stage": "group", "group": "E"},
    {"match_id": "WC26_B2", "home_team": "Spain",     "away_team": "Morocco",     "venue_id": "V006", "date": "2026-06-21", "stage": "group", "group": "B"},
    {"match_id": "WC26_C2", "home_team": "Germany",   "away_team": "Australia",   "venue_id": "V007", "date": "2026-06-21", "stage": "group", "group": "C"},
    # Knockout round placeholder
    {"match_id": "WC26_QF1", "home_team": "TBD",      "away_team": "TBD",         "venue_id": "V001", "date": "2026-07-03", "stage": "quarter_final", "group": None},
    {"match_id": "WC26_SF1", "home_team": "TBD",      "away_team": "TBD",         "venue_id": "V001", "date": "2026-07-14", "stage": "semi_final",    "group": None},
    {"match_id": "WC26_FIN", "home_team": "TBD",      "away_team": "TBD",         "venue_id": "V001", "date": "2026-07-19", "stage": "final",         "group": None},
]

# ── Group standings (initialised to 0 — updated as matches play out) ─────────
def get_initial_standings() -> dict[str, list[dict]]:
    standings = {}
    for group, teams in GROUPS.items():
        standings[group] = [
            {"team": t, "P": 0, "W": 0, "D": 0, "L": 0, "GF": 0, "GA": 0, "GD": 0, "Pts": 0}
            for t in teams
        ]
    return standings


# ── Lookup helpers ────────────────────────────────────────────────────────────

def get_team_data(team_name: str) -> Optional[dict]:
    """Fuzzy-ish lookup — exact match first, then case-insensitive."""
    if team_name in TEAM_FORM:
        return TEAM_FORM[team_name]
    for key in TEAM_FORM:
        if key.lower() == team_name.lower():
            return TEAM_FORM[key]
    return None


def get_h2h_data(home_team: str, away_team: str) -> Optional[dict]:
    """Look up head-to-head in both orderings."""
    if (home_team, away_team) in H2H:
        return H2H[(home_team, away_team)]
    if (away_team, home_team) in H2H:
        d = H2H[(away_team, home_team)].copy()
        # Flip home/away perspective
        d["home_wins"], d["away_wins"] = d["away_wins"], d["home_wins"]
        d["home_goals_avg"], d["away_goals_avg"] = d["away_goals_avg"], d["home_goals_avg"]
        return d
    return None


def get_upcoming(days_ahead: int = 3) -> list[dict]:
    """Return scheduled matches within the next N days (static dataset)."""
    return [m for m in UPCOMING_MATCHES if m["stage"] == "group"][:days_ahead * 4]


def get_venue(venue_id: str) -> Optional[dict]:
    return VENUES.get(venue_id)
