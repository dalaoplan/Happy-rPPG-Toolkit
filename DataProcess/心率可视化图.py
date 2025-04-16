import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# 创建示例数据
np.random.seed(42)
n = 30  # 每组30个样本

conditions = ['Bright-Fixed', 'Dark-Fixed', 'Bright-Moving', 'Dark-Moving']
states = ['Rest', 'Exercise']

data = []
for condition in conditions:
    for state in states:
        if state == 'Rest':
            hr = np.random.normal(loc=75, scale=7, size=n)
        else:
            hr = np.random.normal(loc=100, scale=10, size=n)
        for h in hr:
            data.append([condition, state, h])

df = pd.DataFrame(data, columns=['Lighting', 'State', 'HR'])




# plt.figure(figsize=(8, 5))
# sns.boxplot(x="Lighting", y="HR", hue="State", data=df)
# plt.title("Heart Rate Distribution by Lighting Condition and State (Boxplot)")
# plt.ylabel("Heart Rate (bpm)")
# plt.xlabel("Lighting Condition")
# plt.legend(title="State")
# plt.tight_layout()
# plt.show()


plt.figure(figsize=(8, 5))
sns.violinplot(x="Lighting", y="HR", hue="State", data=df, split=True)
plt.title("Heart Rate Distribution by Lighting Condition and State (Violin Plot)")
plt.ylabel("Heart Rate (bpm)")
plt.xlabel("Lighting Condition")
plt.legend(title="State")
plt.tight_layout()
plt.show()
