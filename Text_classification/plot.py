from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
from glob import glob
import numpy as np

df = pd.read_csv('TB/new_data/base_train_loss.csv')

sns.set()

# df.plot(kind='line', x='step', y='value')
# plt.plot(df['step'], df['value'])
# plt.xlabel('Epochs')
# plt.ylabel('Losses')
# plt.title('Pre-training Losses')
# plt.savefig('new_docs/pre_train_losses.png', dpi=1000)
# plt.show()


##### Avg Train Losses ######
all_csv = glob('TB/new_data/fold_*_main_f1_score.csv')
all_losses = []

plt.figure()
# ax = plt.gca()

for fold_count in range(10):

    csv_file = 'TB/new_data/fold_{}_main_val_loss.csv'.format(fold_count)
    df = pd.read_csv(csv_file)

    all_losses.append(list(df['value']))

    plt.plot(df['step'], df['value'], alpha=0.25, label='fold ' + str(fold_count + 1))

# plt.tight_layout()


all_loss_np = np.array(all_losses)
# print(all_loss_np)
# print(all_loss_np.shape)

all_loss_mean = np.mean(all_loss_np, axis=0)

# print(all_loss_mean)
plt.plot(df['step'], all_loss_mean, label='average')
plt.title('Fine-tune Validation Losses')
plt.xlabel("Epochs")
plt.ylabel("Losses")
plt.legend()
plt.savefig('new_docs/fine_tune_val_losses.png', dpi=1000)
plt.show()