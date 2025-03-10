## Constrained Non-negative Matrix Factorization for Guided Topic Modeling of Minority Topics 


This repository contains code to run our novel topic modeling algorithm, along with evaluations comparing it against other representative models from the paper.

## Running the Algorithm

To run the topic modeling algorithm and save the discovered topics and documents associated with each topic, use the following command:

```bash
python script.py --data_path "./synthetic-data.csv" --output_path "./output.txt" --n_topics 30 --W_max 1e-9 --theta_min 0.4
You can customize the following parameters:

--data_path: Path to your input CSV file.
--output_path: Path to save the output results.
--n_topics: Number of topics to discover.
--W_max: Maximum value for W matrix.
--theta_min: Minimum threshold for theta.
Additionally, the following parameters can be customized inside the script-run.py file:

--MH_indices: List of indices for the mental health topics.
--seed_words: List of seed words for guiding topic discovery.


## Evaluation
The evaluation is performed using two objective metrics:

1. NMI (Normalized Mutual Information) score
2. Purity Score

To evaluate the model's performance, run the `Evaluation.py` script with the following command:

python Evaluation.py --data_path "./synthetic-data.csv"




We have implemented a custom purity score to ensure fairness in evaluation, particularly when dealing with imbalanced labels. The custom function excludes majority labels and focuses on minority predicted labels. The function is available in Evaluation.py as purity_score_filtered. 
In the Evaluation.py file, if you would like to customize different parameters or other SOTA models, feel free to do so. 
