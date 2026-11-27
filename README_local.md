# Bank Marketing ML Project

Open `ML_Project.ipynb` in Jupyter and run all cells with the provided `bank-full (1).csv` in the same folder. The notebook uses only this attached dataset and creates an internal modeling copy; it does not edit the source CSV.

Install dependencies with `python -m pip install -r requirements.txt`. The notebook performs EDA, leakage-aware preprocessing, a stratified 80/20 train/test split (as requested; the teacher rubric's default is 70/30), baseline and six classifiers (including a linear SVM suitable for this dataset size), cross-validated logistic tuning, multiple classification metrics, visual comparisons, and permutation importance. Running all cells saves the feature/target split copies to `Dataset/train_split.csv` and `Dataset/test_split.csv`, and test comparison metrics to `model_comparison.csv`. Reported scores are calculated by the notebook.
