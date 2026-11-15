SmartFlow AI

Predicts ride/traffic demand for a given place and time. I built it for a competition, where predictions were submitted as a ranked file and scored on a leaderboard.

How it works

1. Location features Each (lat, lon) point is converted to a geohash. Each geohash is then replaced with a smoothed average of its past demand (Bayesian target encoding):

encoded = (n * area_mean + m * global_mean) / (n + m)

Areas with few records get pulled toward the global mean, so they don't overfit. For training rows the encoding is computed out-of-fold so the target never leaks into its own feature.

2. Time features Hour of day and day of week are encoded as sine/cosine pairs. This way 23:00 and 00:00 (or Sunday and Monday) are treated as close together, which plain integers can't do.

3. Model LightGBM and Random Forest are trained separately and blended 60/40.

4. Validation Time-ordered split: train on the first 80%, validate on the last 20%. Scored with R².

Results
Model	Validation R²
LightGBM	add
Random Forest	add
Ensemble	add
Project structure
smartflow-ai/
├── train.py            # features, training, validation, submission
├── requirements.txt
├── data/               # put train.csv and test.csv here (not tracked)
└── README.md
Setup
bash
git clone https://github.com/Rupsapatra3107/smartflow_ai.git
cd smartflow_ai
pip install -r requirements.txt

Add train.csv and test.csv to data/, then:

bash
python train.py

Expected columns: timestamp, lat, lon, demand. If yours are named differently, change CONFIG at the top of train.py.

The script prints validation R² for each model and writes submission_ranked.csv (test rows ranked by predicted demand).

Tech

Python, pandas, NumPy, scikit-learn, LightGBM, pygeohash

Possible improvements
Tune LightGBM and the blend weights instead of using fixed values
Add weather and holiday features
Try neighbouring-geohash features to capture spillover between areas
Author

Rupsa Patra · GitHub
