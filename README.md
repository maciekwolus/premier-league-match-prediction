# Premier League Match Prediction

A machine learning application that predicts the outcomes of Premier League matches based on historical match results and player statistics.

## How It Works

1. **Historical match data** — past Premier League match results are used as the foundation for training the prediction model.
2. **Player lineups** — each historical match includes information about which players participated.
3. **FIFA player statistics** — player lineup data is enriched with attributes from FIFA Manager (ratings, pace, shooting, passing, etc.), giving the model a quality signal for each squad.
4. **Prediction** — given a future match and its expected lineups, the model predicts the outcome (win / draw / loss).

## Data Sources

- Premier League historical match results
- Per-match player lineup records
- FIFA Manager player statistics table

## Goal

Given the two starting lineups for an upcoming Premier League fixture, output a predicted match result.
