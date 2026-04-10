# Smart Laptop Recommendation System

## Overview

This project builds an interactive dashboard for generating laptop upsell and accessory cross-sell recommendations using Apriori association rule mining and ROI analysis.

## Files

- `dashboard.py` — Streamlit dashboard with recommendation, survey overlay, and business analytics.
- `cleaned_dataset.csv` — Laptop catalogue and bundled accessory data.
- `Survey-Response-500.csv` — Survey responses from 500 users.
- `requirements.txt` — Python dependencies.
- `Final_Report.md` — Project report and documentation.
- `Presentation_Deck.md` — Slide-by-slide presentation outline.

## How to Run

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Launch the dashboard:

```bash
streamlit run dashboard.py
```

3. Open the local Streamlit URL shown in the terminal to view the demo.

## What the Dashboard Shows

- Laptop recommendation cards for upsell and cross-sell.
- Dynamic survey filters by age group and occupation.
- Accessory opportunity and gross profit analysis.
- Upgrade demand and business impact visualizations.
- Association rules derived from laptop bundle data.

## Notes

- Conversion rate is assumed at 20% for revenue projections.
- Cost model includes product, personnel, and campaign costs for every recommendation.
- The dashboard is designed for demonstration and stakeholder presentations.
