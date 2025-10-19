import matplotlib.pyplot as plt
import numpy as np

# # Client 17 데이터 (예시)
# client_id = 17
# mode = "image_only"
# baseline_loss = 0.7029
# updated_loss  = 0.4248
# baseline_auc  = 0.4991
# updated_auc   = 0.5573

# Client 19 데이터
client_id = 19
mode = "text_only"
baseline_loss = 0.6643
updated_loss  = 0.5610
baseline_auc  = 0.5148
updated_auc   = 0.5674

# 그래프용 데이터 구성
metrics = ["Loss", "Macro AUC"]
baseline_vals = [baseline_loss, baseline_auc]
updated_vals  = [updated_loss, updated_auc]

x = np.arange(len(metrics))
width = 0.35

plt.figure(figsize=(6, 4))
bars1 = plt.bar(x - width/2, baseline_vals, width, label='Baseline', color='lightgray')
bars2 = plt.bar(x + width/2, updated_vals,  width, label='Updated', color='skyblue')

plt.title(f"Client {client_id} ({mode}) — Loss & mAUC Comparison")
plt.xticks(x, metrics)
plt.ylabel("Value")
plt.grid(alpha=0.3, axis='y')
plt.legend()

# 막대 위에 값 표시
for bars in [bars1, bars2]:
    for b in bars:
        plt.text(b.get_x() + b.get_width()/2, b.get_height() + 0.01,
                 f"{b.get_height():.4f}", ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.show()
