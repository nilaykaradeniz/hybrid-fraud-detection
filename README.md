# Hybrid fraud detection for online accommodation reservations

An online reservation platform checks every payment attempt with fraud rules. The rules catch the attacks, but they also send many good customers to an extra check. This project keeps the rules and adds a machine learning model on top: where a rule is sure, the rule decides, and where a rule is only a hint, a risk score decides. It then compares this hybrid system with the rules alone.

The full story, with every step and its output, is in the notebook [`notebooks/hybrid_fraud_detection.ipynb`](notebooks/hybrid_fraud_detection.ipynb). The design and the reasons behind it are in the [project brief](docs/project_brief.md).

## The data

Synthetic transaction data from a fraud analytics case study: 79,557 payment attempts for accommodation reservations from January 2022, with 17 columns.

- No row needed a fix. All 5 blocking checks pass, and 10 quality checks found nothing to repair.
- 338 reservations have 0 nights. We keep them, and a rule uses them as a warning sign.
- There is no time of day, no customer ID and no confirmed fraud label.
- The payment provider marks 1.0% of the attempts as FRAUD, but this mark is close to random: a model trained on it scores a ROC AUC of 0.52.

Because there is no label, we build our own. We link the attempts that share a card, an email or an IP. This gives two attack clusters with 2,380 attempts. The label is only used to measure the results. No model is trained on it.

## The approach

![Hard rules act on their own; the model decides on soft-rule alerts](docs/hybrid_flow.png)

1. **Rules.** 9 hard rules act on their own (extra check, manual review or block). 4 soft rules only raise a flag.
2. **Risk score.** An Isolation Forest, a model that needs no labels, scores how unusual each attempt is. It learns from days 1 to 14 only.
3. **Decision.** For an attempt with only soft alerts, the risk band and the value of the reservation choose the action.
4. **Comparison.** The hybrid system against the rules alone, where every soft alert gets an extra check, on days 15 to 31.

The feedback loop in the diagram (analyst decisions and chargebacks as labels) is a design for a live system. It is not built here.

## Main results

Test days 15 to 31, with the same rules on both sides. "Friction" means an extra check, a manual review or a block.

| Metric | Rules only | Hybrid |
| --- | --- | --- |
| Attempts with friction | 8,090 | 913 |
| Share of friction that falls on good reservations | 89.7% | 9.0% |
| Good reservations with friction | 7,259 | 82 |
| Share of all good reservations with friction | 17.2% | 0.2% |
| Value of good reservations with friction | 19.0 million EUR | 0.2 million EUR |
| Fraud value caught | 3.13 million EUR | 3.13 million EUR |
| Manual reviews per day | 26.5 | 26.5 |

- The hybrid system bothers far fewer good customers and loses no fraud.
- The risk score ranks fraud with never-seen cards high: 97.5% of these 766 attempts are in the top 5% of the scores (ROC AUC 0.993, PR AUC 0.725).
- The result holds with day 10 or day 18 as the last training day.

**Read these numbers with one fact in mind.** On the test days the hard rules alone already catch every fraud attempt. So this test proves the gain for the good customers: the soft rules can stay switched on as a safety net at 82 extra checks, not 7,259. It does not prove that the model stops fraud that the hard rules miss.

For that case the notebook has a stress test: every fraud attempt gets its own IP, so the hard rules lose most of the attack. The hybrid system then catches only 26.3% of the fraud value with its current cut point. With a lower cut point it catches 100% and still gives good customers 87% less friction than the rules alone. The cut point is the main lever, and we did not tune it on the test days.

## How to run it

Python 3.10 or newer. The data file is not part of this repository. Put it in the `data` folder as `data/transactions.csv` before you run the code.

```bash
pip install -r requirements.txt

# Run the whole pipeline. It prints the data quality report and the main
# comparison, and saves a run folder in outputs/.
python -m src.pipeline

# Open the notebook. It is saved with all its outputs, so it can also be read without running it.
jupyter notebook notebooks/hybrid_fraud_detection.ipynb

# Run the tests.
pytest
```

Every threshold and setting is in `config.yaml`. A pipeline run takes well under a minute, and the whole notebook a few minutes.

## Limits

- **Synthetic data.** The patterns are cleaner than in real life. The results are a proof of concept, not expected production numbers.
- **Our own label.** It means "shares a card, email or IP with another attempt". Rules and features that count repeated cards, emails or IPs agree with it by design, which raises their precision and the ROC AUC of the model.
- **Rules designed with the whole month in view.** The numbers of the rules alone are optimistic. The comparison of the two setups is fair, because both use the same rules.
- **One attack in the test days.** All fraud of the test days comes from one attacker and one IP.
- **The cut points depend on the attack in the training days.** With another attack there, the risk bands and the number of extra checks would be different.
- **No time of day and one month of data.** Counts for the same day include later attempts of that day, and we cannot see seasons or slow attacks.
- **The model numbers move with the random seed.** In a check with 9 seeds the ROC AUC stayed at about 0.99, but the PR AUC went from 0.62 to 0.79. We keep seed 42.

The notebook lists all limits, the recommendations and the next steps in its last section.

## Repository layout

```text
data/transactions.csv                    the data (not in this repository)
docs/project_brief.md                    the business problem and the design
docs/hybrid_flow.png                     the flow diagram
src/
  data_loader.py                         DataLoader: load, blocking checks, split by time
  data_quality.py                        DataQualityChecker: the quality report
  data_profile.py                        DataProfiler: tables that describe the data
  split_check.py                         SplitChecker: do training and test days look alike (PSI)
  labels.py                              InvestigationLabeler: the label from linked attempts
  features.py                            FeatureBuilder: counts per card, email, IP and property
  rules.py                               RuleEngine: hard and soft rules, and the hotlist
  risk_model.py                          RiskModel: Isolation Forest, risk bands and SHAP
  decision.py                            DecisionPolicy: one action per attempt, for both setups
  evaluation.py                          Evaluator: all result tables
  pipeline.py                            FraudPipeline: runs the steps in order
notebooks/hybrid_fraud_detection.ipynb   every step, its output and what it means
tests/                                   six small tests
outputs/                                 saved runs (not committed)
config.yaml                              all thresholds and settings
requirements.txt
```
