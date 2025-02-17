import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats


num_subjects = 2000 // 40

mean_per_subject = []
std_per_subject = []

# Calculate mean and std deviation for each subject
for i in range(num_subjects):
    start_idx = i * 40
    end_idx = (i + 1) * 40
    subject_data = eeg_data_reshaped[start_idx:end_idx, :, :]

    # Calculate mean and std deviation across all trials for each channel
    mean_subject = np.mean(subject_data, axis=0)  # mean across trials
    std_subject = np.std(subject_data, axis=0)    # std deviation across trials

    # Aggregate mean and std deviation for all channels
    mean_per_subject.append(np.mean(mean_subject, axis=1))
    std_per_subject.append(np.mean(std_subject, axis=1))

mean_per_subject = np.array(mean_per_subject)
std_per_subject = np.array(std_per_subject)

df_stats = pd.DataFrame({
    'Subject': np.arange(num_subjects),
    'Mean': np.mean(mean_per_subject, axis=1),
    'Std_Dev': np.mean(std_per_subject, axis=1)
})

anova_mean = stats.f_oneway(*[df_stats['Mean'][i::5] for i in range(5)])
anova_std = stats.f_oneway(*[df_stats['Std_Dev'][i::5] for i in range(5)])

print("ANOVA Results for Mean EEG Signal Across Subjects:")
print(f"F-Statistic: {anova_mean.statistic}, p-Value: {anova_mean.pvalue}")

print("\nANOVA Results for Standard Deviation of EEG Signal Across Subjects:")
print(f"F-Statistic: {anova_std.statistic}, p-Value: {anova_std.pvalue}")

plt.figure(figsize=(12, 6))
sns.boxplot(x='Subject', y='Mean', data=df_stats)
plt.title('Mean EEG Signal Across Subjects')
plt.xlabel('Subject')
plt.ylabel('Mean Value')
plt.xticks(ticks=np.arange(0, num_subjects, 2), labels=np.arange(1, num_subjects+1, 2))
plt.tight_layout()
plt.savefig('mean_eeg_signal.eps')
plt.savefig('mean_eeg_signal.pdf')
plt.close()

plt.figure(figsize=(12, 6))
sns.boxplot(x='Subject', y='Std_Dev', data=df_stats)
plt.title('Standard Deviation of EEG Signal Across Subjects')
plt.xlabel('Subject')
plt.ylabel('Standard Deviation')
plt.xticks(ticks=np.arange(0, num_subjects, 2), labels=np.arange(1, num_subjects+1, 2))
plt.tight_layout()
plt.savefig('std_dev_eeg_signal.eps')
plt.savefig('std_dev_eeg_signal.pdf')
plt.close()

plt.figure(figsize=(12, 6))
sns.violinplot(data=df_stats[['Mean', 'Std_Dev']])
plt.title('Violin Plot of EEG Signal Statistics')
plt.ylabel('Value')
plt.tight_layout()
plt.savefig('violin_plot_eeg_signal.eps')
plt.savefig('violin_plot_eeg_signal.pdf')
plt.close()

plt.figure(figsize=(12, 6))
sns.histplot(df_stats['Mean'], bins=30, kde=True)
plt.title('Distribution of Mean EEG Signal Across Subjects')
plt.xlabel('Mean Value')
plt.ylabel('Frequency')
plt.tight_layout()
plt.savefig('distribution_mean_eeg_signal.eps')
plt.savefig('distribution_mean_eeg_signal.pdf')
plt.close()

df_stats.to_csv('subject_statistics.csv', index=False)