# Software Reliability and Bug Risk Predictor

ML classifier trained on NASA/PROMISE JM1 software defect data. It uses static code metrics (McCabe/Halstead-style metrics and branch count) to estimate whether a software module is defect-prone.

## Train
pip install -r requirements.txt
python train_model.py

The script downloads `jm1.csv` when absent and saves `bug_risk_model.pkl`.

## Predict
Provide a CSV containing the same metric columns, then:
python predict.py module_metrics.csv

Output: bug probability and LOW/MEDIUM/HIGH risk level in `predictions.csv`.
