# F1 Pit Stop Strategy Intelligence System

**A Multi-Layered Machine Learning and Statistical Framework for Formula 1 Race Strategy Prediction**

**Author:** Nguyen Tri  
**Date:** June 2026  
**Domain:** Formula 1 Strategy Analytics, Machine Learning, Survival Analysis

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Project Motivation and Research Questions](#2-project-motivation-and-research-questions)
3. [System Architecture](#3-system-architecture)
4. [Data Pipeline and Feature Engineering](#4-data-pipeline-and-feature-engineering)
5. [Deep Learning Pit Stop Prediction](#5-deep-learning-pit-stop-prediction)
6. [Stochastic and Survival Modelling](#6-stochastic-and-survival-modelling)
7. [Strategic Regression and Tactical Agility](#7-strategic-regression-and-tactical-agility)
8. [Overtake Classification](#8-overtake-classification)
9. [Descriptive Analytics](#9-descriptive-analytics)
10. [Results and Key Findings](#10-results-and-key-findings)
11. [Limitations](#11-limitations)
12. [Suggested Improvements](#12-suggested-improvements)
13. [Conclusion](#13-conclusion)
14. [Appendix: Project Files](#appendix-project-files)

---

## 1. Executive Summary

This report documents the design, implementation, and findings of a Formula 1 pit stop strategy intelligence system. The project combines deep learning, survival analysis, statistical testing, strategic regression, overtake classification, and descriptive analytics to model when teams are likely to pit and how pit strategy affects race outcomes.

The core deliverable is a three-layer Bidirectional LSTM (BiLSTM) model trained on 20-lap sliding windows of race telemetry. The model predicts the probability of a pit stop on the final lap of each window. These probabilities are then reused by downstream modules to analyse team risk profiles, tactical agility during Virtual Safety Car (VSC) events, strategic overtakes, and the relationship between pit crew performance and championship success.

| Metric | Value |
|--------|-------|
| Training Seasons | 7 seasons, 2018-2024 |
| Held-Out Test Season | 2025 |
| Total Laps Processed | 152,475 |
| Pit Stop Events | 4,818 |
| Positive Class Rate | 3.16% |
| Sequence Length | 20 laps |
| Engineered Features | 40+ |
| BiLSTM Layers | 3 bidirectional layers |
| Mean Model Confidence on True Pit Laps | 83.8% |
| Integrated Analysis Modules | 5 |

![Dataset Overview](plots/01_dataset_overview.png)

*Figure 1. Dataset overview showing lap time distributions, tyre life, pit stop timing, and seasonal pit stop rates across 2018-2025.*

---

## 2. Project Motivation and Research Questions

Formula 1 strategy is shaped by decisions made under uncertainty: tyre degradation, traffic, Safety Car timing, track position, compound availability, and competitor behaviour all influence whether a driver should pit. A model that estimates real-time pit stop probability can help identify undercut threats, evaluate tyre windows, support race debriefs, and quantify team-level strategic tendencies.

The project focuses on four research questions:

1. Can historical lap telemetry be used to predict pit stop events with useful precision and recall?
2. Which stochastic process best characterises the hazard of pitting as tyre age increases?
3. Is there a statistically meaningful relationship between pit crew speed and constructor championship performance?
4. Can a driver's reaction time to VSC events predict whether they finish above their qualifying position?

---

## 3. System Architecture

The system is organised as a multi-stage analytics pipeline. Each module produces either model-ready features, pit stop probabilities, or strategy insights consumed by later stages.

| Stage | Script / Module | Purpose | Main Output |
|-------|-----------------|---------|-------------|
| Data Collection | `download.py` | Download FastF1 race, lap, and telemetry data | Raw race CSV files |
| Preprocessing | `Preprocess.py` | Clean laps, aggregate telemetry, engineer features | `all_training_data.csv` |
| Deep Learning | `train.py`, `model.py` | Train BiLSTM pit stop classifier | `best_f1_model.pt` |
| Inference | `predict_pit_probabilities.py` | Generate per-lap pit stop probabilities | `PitStopProbability` |
| Strategy Analytics | `stochastic_modeling.py`, `strategic_regression.py`, `overtake_analysis.py`, `analyse.py` | Survival modelling, regression, overtake classification, visual analytics | Strategy metrics and plots |

---

## 4. Data Pipeline and Feature Engineering

### 4.1 Data Sources

| Source | Coverage | Usage |
|--------|----------|-------|
| FastF1 API | 2018-2025 Grand Prix races | Lap timing, telemetry, tyre compound, tyre age, position, DRS, throttle, brake, speed, RPM, gear |
| Ergast API | Historical Formula 1 data | Pit stop durations, constructor IDs, standings, race results |
| Preprocessed CSVs | All processed sessions | Model-ready feature tables and aggregated training data |

### 4.2 Download and Caching

The download module iterates over the FastF1 event schedule for each selected season, skips testing events, and avoids duplicate downloads through file-existence checks. For each Grand Prix, it saves race results, lap data, and telemetry. A short delay between sessions reduces rate-limit risk.

### 4.3 Preprocessing

Raw lap data is filtered to retain valid dry-weather compounds: `SOFT`, `MEDIUM`, and `HARD`. Timedelta fields are converted to seconds. Telemetry is aligned to each driver lap using session-time intervals, then aggregated into lap-level features.

### 4.4 Feature Groups

| Feature Group | Examples | Strategic Rationale |
|---------------|----------|---------------------|
| Race State | `LapNumber`, `Position`, `TrackStatus` | Captures context and phase of race |
| Tyre Degradation | `TyreLife`, `tire_performance_decay`, `relative_tire_age` | Represents primary pit stop trigger |
| Pace Evolution | `delta_laptime`, `rolling_pace_mean_5`, `pace_degradation_slope` | Detects pace drop-off and compound cliff |
| Telemetry | `Speed_mean`, `RPM`, `Throttle`, `Brake`, `DRS`, `nGear` | Captures car behaviour and driver load |
| Traffic and DRS | `DistanceToDriverAhead_mean`, `traffic_pressure`, `drs_dependency` | Measures undercut and dirty-air pressure |
| Strategy Priors | `historical_pit_lap`, `pit_window_delta` | Encodes historical team and compound tendencies |

### 4.5 Leakage Prevention

Direct future-information columns such as `PitInTime`, `PitOutTime`, `LapTime`, `Sector1Time`, `Sector2Time`, `Sector3Time`, and `SessionTime` are removed before modelling. The `historical_pit_lap` feature is calculated only from the training split and then merged into the test split to prevent future information from leaking into held-out predictions.

### 4.6 Sequence Construction

Sequences are generated per `(Year, RaceID, DriverNumber)` group and sorted by `LapNumber`. A 20-lap sliding window advances one lap at a time. The target label is the `HasPitStop` value on the final lap of the window. Driver-race groups shorter than 20 laps are excluded.

### 4.7 Class Imbalance Handling

Pit stops account for only 3.16% of all processed laps. The training procedure uses `BCEWithLogitsLoss` with positive-class weighting, so genuine pit stop events receive greater importance during optimisation. The final classification threshold is selected by maximising F1-score over the precision-recall curve instead of using a default 0.5 threshold.

---

## 5. Deep Learning Pit Stop Prediction

### 5.1 Model Architecture

The `F1PitStopPredictor` model processes input tensors of shape `(batch, 20, features)`. Three stacked Bidirectional LSTM layers learn sequential race context from the preceding 20 laps. The final hidden representation is passed through fully connected layers and converted into a binary pit stop probability.

| Layer | Input Dimension | Hidden Size | Output Dimension | Dropout |
|-------|-----------------|-------------|------------------|---------|
| BiLSTM-1 | D features | 256 | 512 | 0.20 |
| BiLSTM-2 | 512 | 128 | 256 | 0.30 |
| BiLSTM-3 | 256 | 64 | 128 | 0.30 |
| FC-1 | 128 | - | 64 | 0.20 |
| FC-2 | 64 | - | 32 | - |
| Output | 32 | - | 1 probability | - |

After each BiLSTM layer, dropout and batch normalisation are applied. The final token from the third BiLSTM output is fed into dense layers with ReLU activations. Gradient clipping with maximum norm 1.0 improves training stability.

### 5.2 Training Configuration

| Hyperparameter | Value | Reason |
|----------------|-------|--------|
| Optimiser | Adam | Handles sparse and noisy gradients well |
| Learning Rate | `1e-4` | Conservative learning rate for sequence modelling |
| Scheduler | `ReduceLROnPlateau`, patience 4, factor 0.5 | Reduces learning rate when AUC-PR stalls |
| Loss | `BCEWithLogitsLoss` with `pos_weight` | Stable binary loss with imbalance support |
| Primary Metric | AUC-PR | Better suited to rare-event classification than accuracy |
| Early Stopping | Patience 8 on AUC-PR | Limits overfitting |
| Batch Size | 64 training, 256 inference | Balances stability and throughput |
| Maximum Epochs | 50 | Allows convergence with early stopping |

### 5.3 Train / Test Split

The dataset is split temporally. Seasons 2018-2024 are used for training, while 2025 is held out for testing. This simulates real deployment, where a model trained on historical races must generalise to a future season.

### 5.4 Inference and Probability Mapping

The `predict_pit_probabilities.py` module loads `best_f1_model.pt` and generates a `PitStopProbability` score between 0 and 1 for every valid 20-lap window. The first 19 laps of each driver-race group receive a default probability of 0.0 because no full sequence is available yet.

Predictions are mapped back to the source DataFrame using `original_row_id`, avoiding alignment errors after sorting and grouping.

![Pit Stop Probability Distributions](plots/03_model_outputs.png)

*Figure 2. BiLSTM output distributions for all laps and by actual pit stop outcome. True pit laps receive substantially higher predicted probabilities.*

---

## 6. Stochastic and Survival Modelling

### 6.1 Time Series Stationarity

Raw lap times are non-stationary because fuel load decreases while tyres degrade. The Partial Autocorrelation Function (PACF) is used to inspect stint-level lag structure. An AR(1) first-difference filter is then applied:

```text
Delta X_t = X_t - X_{t-1}
```

This produces `Stationary_PaceDecay`, which better isolates tyre-induced degradation from traffic noise and fuel effects.

### 6.2 Cox Proportional Hazards Model

Pit stops are framed as a survival process. At each lap, a driver has a hazard of stopping. The Cox Proportional Hazards model is fitted with:

| Component | Variable |
|-----------|----------|
| Duration | `TyreLife` |
| Event Indicator | `HasPitStop` |
| Covariates | `Stationary_PaceDecay`, `traffic_pressure` |

The partial hazard is min-max normalised to produce a baseline per-lap pit probability that can be compared with the BiLSTM output.

### 6.3 Team Risk Profiles

For every pit stop event, tyre life at the stopping lap is recorded. Teams are compared using the expected tyre life at pit stop, `E[X]`, and its variance, `Var(X)`.

| Team | Mean Tyre Life, E[X] | Variance, Var(X) | Strategy Profile |
|------|----------------------|------------------|------------------|
| Ferrari | 19.3 laps | 99.3 | Most deterministic, safety-first |
| Racing Point | 20.5 laps | 107.5 | Consistent strategy windows |
| Mercedes | 20.5 laps | 111.0 | Disciplined, low-variance strategy |
| Kick Sauber | 20.2 laps | 167.1 | Opportunistic and reactive |
| Toro Rosso | 21.0 laps | 150.4 | Flexible, high-variance strategy |
| Alfa Romeo | 17.2 laps | 143.7 | Aggressive, earlier stopping tendency |

Low-variance teams such as Ferrari and Mercedes tend to pit within narrower windows. High-variance teams such as Kick Sauber and Toro Rosso appear more reactive to Safety Cars, undercut threats, and track-position opportunities.

![Team Strategy Profiles](plots/04_team_strategy.png)

*Figure 3. Team strategy profiles based on mean tyre life and tyre-life variance at pit stop.*

### 6.4 Statistical Testing

A Welch's t-test compares `Stationary_PaceDecay` under high and low traffic pressure. The result validates whether `traffic_pressure` contains useful information beyond raw telemetry and supports its inclusion in the feature set.

![Tyre Degradation Curves](plots/02_tyre_analysis.png)

*Figure 4. Compound-specific tyre degradation curves with one-standard-deviation confidence bands. The SOFT compound shows the steepest degradation cliff.*

---

## 7. Strategic Regression and Tactical Agility

### 7.1 Pit Duration vs Championship Points

Using Ergast data, median pit stop duration is calculated per constructor and compared with cumulative championship points. Constructors with fewer than 100 career points are filtered out to reduce noise from short-lived or low-sample teams. Pearson correlation is then used to test whether faster pit crew execution is associated with stronger championship performance.

### 7.2 VSC Tactical Agility Score

VSC and Safety Car deployments are detected using the `Stochastic_Shock_VSC_SC` flag. For each driver-race, the system measures how many laps pass between a VSC/SC event and a pit stop response. Drivers who do not pit during the opportunity window receive a penalty score of 5.0 laps.

This produces an `Agility_Score`, where lower values represent faster tactical reaction.

### 7.3 Random Forest Outcome Classifier

A Random Forest Classifier with 100 estimators predicts whether a driver finishes above their qualifying position.

| Feature | Description | Interpretation |
|---------|-------------|----------------|
| `GridPos` | Starting grid position | Baseline race-position prior |
| `Agility_Score` | Mean laps to react to VSC/SC | Tactical responsiveness |
| `PaceDecay` | AR(1)-filtered pace degradation | Tyre management quality |
| `Mean_Pit_Prob` | Average BiLSTM probability during race | Pit strategy pressure signal |

If `Agility_Score` feature importance exceeds 0.15, the framework treats tactical speed under VSC/SC conditions as a highly significant predictor of finishing above qualifying position.

---

## 8. Overtake Classification

The `overtake_analysis.py` module constructs lap-by-lap position matrices for each race and checks all driver pairs for position swaps. Each detected swap is classified as either strategic or on-track.

| Class | Definition |
|-------|------------|
| Strategic Overtake | Either driver pitted within +/-2 laps of the position change, or either driver had `PitStopProbability > 0.8` at the overtake lap |
| On-Track Overtake | No nearby pit stop and no high pit stop probability signal |

The analysis is limited to top-10 position changes to reduce noise from retirements, lapped cars, and backmarker position volatility.

![Race Dynamics](plots/05_race_dynamics.png)

*Figure 5. Race dynamics showing position changes by race phase and pit stops by compound per season.*

---

## 9. Descriptive Analytics

The `analyse.py` module provides population-level summary statistics and visualisations.

| Analysis | Key Finding |
|----------|-------------|
| Lap Time Distribution | Mean lap time is 89.6 seconds; bimodal peaks reflect compound and circuit differences |
| Pit Lap Distribution | Mean pit lap is 27.7, with the primary pit window concentrated between laps 20 and 35 |
| Tyre Life Distribution | Mean tyre life is 15.1 laps; SOFT compounds stop earlier |
| Position Change Timing | Early laps and the mid-race pit window account for most net position changes |
| Class Imbalance | 4,818 pit events from 152,475 laps, equal to a 3.16% positive rate |

![Feature Correlation Matrix](plots/06_feature_analysis.png)

*Figure 6. Feature correlation matrix showing relationships between engineered variables and the pit stop target.*

---

## 10. Results and Key Findings

### 10.1 Model Performance

- The BiLSTM reaches a mean confidence of 83.8% on true pit stop laps.
- The model remains effective despite the pit stop class representing only 3.16% of all laps.
- The temporal split using 2025 as the held-out season gives a more realistic estimate of generalisation than a random split.
- Training is stabilised through positive-class weighting, gradient clipping, learning-rate scheduling, and early stopping.

### 10.2 Strategic Insights

- Ferrari and Mercedes show the most deterministic pit strategies, with variance below 115 laps squared.
- Kick Sauber and Toro Rosso show high tyre-life variance, suggesting reactive and opportunity-driven strategies.
- The average pit lap is 27.7, with the strongest pit window between laps 20 and 35.
- SOFT compounds pit roughly 8-10 laps earlier than HARD compounds, consistent with steeper degradation.
- Most position changes occur during early race phases and around the main pit window.

### 10.3 Statistical Validation

- Welch's t-test supports the inclusion of `traffic_pressure` as a meaningful explanatory feature.
- Pearson correlation between pit stop duration and championship points provides evidence that operational pit crew speed is linked to sporting performance.
- Random Forest feature importance helps quantify the strategic value of reacting quickly to VSC/SC opportunities.

![Stochastic Analysis](plots/07_stochastic_analysis.png)

*Figure 7. Stochastic analysis outputs used to compare degradation, pit hazard, and strategy-related distributions.*

---

## 11. Limitations

- Weather variables are not yet fully integrated, limiting performance in wet or mixed-condition races.
- Live competitor intent is approximated from historical and telemetry-derived features rather than direct team radio or strategy calls.
- The BiLSTM provides strong predictive performance but limited interpretability compared with attention-based models.
- Pit stop probability is estimated per lap completion, so sub-lap strategy triggers are outside the current scope.
- Constructor and team naming changes across seasons may require additional normalisation for longitudinal comparisons.

---

## 12. Suggested Improvements

### 12.1 Model Architecture

- Add a Transformer or attention layer to improve interpretability and reveal which previous laps most influence pit stop probability.
- Use multi-task learning with auxiliary heads for tyre compound prediction and stint-length regression.
- Stack BiLSTM and Cox hazard outputs with a calibration meta-learner.

### 12.2 Data Enrichment

- Add weather variables such as track temperature, air temperature, humidity, rainfall probability, and wet tyre usage.
- Integrate live or historical pit stop duration into undercut and overcut risk modelling.
- Include setup proxies such as downforce level, fuel-load estimates, and team-specific tyre degradation baselines.

### 12.3 Deployment

- Wrap inference in a FastAPI service for live lap-by-lap pit stop probability updates.
- Build a Monte Carlo strategy simulator to compare future stint and compound options.
- Apply Platt scaling or isotonic regression to calibrate raw model probabilities against empirical pit stop rates.

---

## 13. Conclusion

The project demonstrates that Formula 1 pit stop strategy can be modelled as both a sequential prediction problem and a stochastic survival problem. The BiLSTM model captures short-term race context and produces high confidence on true pit stop laps, while the statistical modules explain broader team tendencies, traffic effects, VSC responsiveness, and strategic overtakes.

Together, the modules form a practical strategy intelligence framework: the model predicts when pit stops are likely, and the analytics layer explains why those decisions matter.

---

## Appendix: Project Files

| File | Role |
|------|------|
| `Code/download.py` | Downloads FastF1 session data |
| `Preprocessing/Preprocess.py` | Cleans data and engineers lap-level features |
| `Code/model.py` | Defines the BiLSTM pit stop model |
| `Code/train.py` | Trains and validates the deep learning model |
| `Code/predict_pit_probabilities.py` | Generates per-lap pit stop probabilities |
| `Code/stochastic_modeling.py` | Runs survival and stochastic strategy analysis |
| `Code/strategic_regression.py` | Evaluates pit duration, points, and tactical agility |
| `Code/overtake_analysis.py` | Classifies strategic and on-track overtakes |
| `Code/analyse.py` | Produces descriptive analytics |
| `report/visualize.py` | Generates report visualisations |
| `plots/*.png` | Report figures |
