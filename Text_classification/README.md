# Twitch Analysis 

## Model Details

### BERT Pre-Training

![Pre-training](./docs/pre_train_arch.png)

### BERT Fine Tuning

![Fine Tuning](./docs/fine_tune_arch.png)

## Model Evaluation

| Fold | Accuracy | F1 Score | Precision | Recall |
|------|----------|----------|-----------|--------|
| 1    | 0.995    | 0.995    | 0.995     | 0.995  |
| 2    | 0.995    | 0.995    | 0.995     | 0.995  |
| 3    | 0.995    | 0.995    | 0.995     | 0.995  |
| 4    | 0.995    | 0.995    | 0.995     | 0.995  |
| 5    | 0.995    | 0.995    | 0.995     | 0.995  |
| 6    | 1.0      | 1.0      | 1.0       | 1.0    |
| 7    | 0.985    | 0.985    | 0.985     | 0.985  |
| 8    | 1.0      | 1.0      | 1.0       | 1.0    |
| 9    | 1.0      | 1.0      | 1.0       | 1.0    |
| 10   | 1.0      | 1.0      | 1.0       | 1.0    |

## Model Evaluation per Class

| Matrix    | GG         | QS         | JK         |
|-----------|------------|------------|------------|
| Accuracy  | 0.99626087 | 0.98838259 | 0.9984127  |
| Precision | 0.99626087 | 0.98838259 | 0.9984127  |
| Recall    | 0.99841537 | 1.0        | 0.98978922 |
| F1 Score  | 0.99732105 | 0.99407429 | 0.99401951 |

## Model Evaluation Graphs

### Pre-training Losses

![Pre-training Losses](./docs/pre_train_losses.png)

### Fine-tune Training Losses

![Fine-tune Training Losses](./docs/fine_tune_train_losses.png)

### Fine-tune Validation Losses

![Fine-tune Validation Losses](./docs/fine_tune_val_losses.png)

### Fine-tune Validation Accuracy

![Fine-tune Validation Accuracy](./docs/fine_tune_val_accuracy.png)

### Fine-tune Validation Precision

![Fine-tune Validation Precision](./docs/fine_tune_val_precision.png)

### Fine-tune Validation Recall

![Fine-tune Validation Recall](./docs/fine_tune_val_recall.png)

### Fine-tune Validation F1 Score

![Fine-tune Validation F1 Score](./docs/fine_tune_val_f1_score.png)