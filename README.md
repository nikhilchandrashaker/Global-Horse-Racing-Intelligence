## Overview
This dataset provides a curated, flat-file sample slice of **JRA (Japan Racing Association) Horse Racing Results** covering the 2024 season. It features rich, row-level data capturing race conditions, running positions, and—crucially—complete 3-generation pedigree data. 

This public sandbox includes around **3 months** from early 2024 and is designed for backtesting, feature engineering, and baseline machine learning modeling.

---

## 🚀 Get the Complete 2024–2026 Dataset (US $49)
Tired of hitting scraping walls on netkeiba or trying to stitch together fragmented, outdated Japanese racing data? The complete, clean historical backtesting dataset is available on Gumroad.

👉 **[Download the Complete JRA Results Dataset (2024–2026) here](https://labmaster35.gumroad.com/l/imkqs)**

**What’s in the Full Version:**
* **Continuous Modern Coverage:** Complete historical records spanning January 2024 to July 2026.
* **Deep Pedigree Fields:** Every row pre-mapped with Sire, Dam, Dam's Sire, and Dam's Dam—perfect for breeding/surface-switch analysis.
* **Granular Race Details:** Full weather conditions, track configurations, final times, and running positions.
* **Market Context:** Win odds and favorite rankings to test active trading strategies.

---

## Data Schema
| Column Name | Type | Description |
|---|---|---|
| `race_date` | Date | YYYY-MM-DD of the race fixture |
| `race_course` | Categorical | JRA track name (e.g., Tokyo, Nakayama, Kyoto) |
| `race_track` | Categorical | Turf vs. Dirt surface indicator |
| `running_position` | String | Sequential passing order (e.g., "4>4>3>2nd") |
| `sire` / `dam` | Categorical | Parental pedigree data |
| `win_odds` | Float | Final payout multiplier |

*(Full schema details and dictionary included in the download)*

---

*Disclaimer: This dataset is for educational, research, and predictive modeling purposes only.*