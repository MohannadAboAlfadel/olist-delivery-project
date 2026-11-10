# Contributions

Each member commits from their own account and contributes to the project components.

| Work | Owner (name / ID) | Reviewer (name / ID) | Branch |
|---|---|---|---|
| Data quality & cleaning (`io_utils`, `audit`, `cleaning`, decisions log) | أحمد رضوان الأسدي | مهند أبو الفضل | `feature/data-quality` |
| Merge & features (`merge`, `target`, `features`, feature dictionary) | مهند أبو الفضل | ماسة المجركش | `feature/merge-features` |
| Analysis & figures (`eda`, `numpy_ops`) | ماسة المجركش | حمزة أبو كلام | `feature/eda` |
| Models & evaluation (`modeling`, experiments log) | حمزة أبو كلام | حمزة شلغين | `feature/modeling` |
| Tests & documentation (`tests/`, README, report) | حمزة شلغين | أحمد رضوان الأسدي | `feature/tests-docs` |

## Mutual Review Plan
Every member reviews exactly one teammate's branch using `docs/REVIEW_CHECKLIST.md`:
- Owner of role k is reviewed by role k+1 (cyclically: 1→2→3→4→5→1)
- Reviews are documented in pull requests and merge commits

## Team Members
1. أحمد رضوان الأسدي - Data Quality & Cleaning
2. مهند أبو الفضل - Merge & Features Engineering
3. ماسة المجركش - Exploratory Data Analysis & Visualizations
4. حمزة أبو كلام - Machine Learning Models & Evaluation
5. حمزة شلغين - Testing & Documentation
