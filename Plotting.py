import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# 创建一个示例数据集
data = {
    'time_seconds': [0, 3.06523, 5.27219, 7.11133, 8.88916, 10.85091, 12.19961, 13.67092, 15.14223,
                     16.49093, 17.90093, 19.12702, 20.53703, 22.00834, 23.66356, 25.25748, 27.28053,
                     30.16184, 35.06621, 40.64492, 46.59147, 51.80235, 57.62629, 64.98283, 71.54242,
                     76.01766, 80.30897, 84.7229, 89.19814, 94.1025, 97.29034, 103.42079, 107.65081,
                     111.69691, 119.23737, 122.60912],
    'value': [0.01109, 0.04879, 0.09647, 0.14859, 0.1996, 0.25171, 0.2994, 0.35151, 0.39808,
              0.4502, 0.49788, 0.54667, 0.59879, 0.64647, 0.70081, 0.75071, 0.79839, 0.8505,
              0.90151, 0.92036, 0.93367, 0.93921, 0.94587, 0.9503, 0.95696, 0.96139, 0.96472,
              0.96804, 0.96915, 0.9747, 0.97802, 0.98024, 0.98357, 0.98579, 0.98911, 0.99022]
}

# 将数据转换为 DataFrame
df = pd.DataFrame(data)

# 风格设置，并使用 set_theme() 替代 set()
sns.set_theme(style="whitegrid")

# 绘制数据，使用 'time_seconds' 作为 x 轴
plt.figure(figsize=(10, 6))
plt.plot(df['time_seconds'], df['value'], marker='o', linestyle='-')

# 添加标题和标签
plt.title('Time Series Data')
plt.xlabel('Time (seconds)')
plt.ylabel('Value')

# 自定义 x 轴刻度
# 获取当前 x 轴的最小值和最大值
x_min, x_max = plt.xlim()

# 定义刻度的位置（例如每隔 10 秒一个刻度）
custom_xticks = range(0, int(x_max) + 1, 10)

# 设置新的 x 轴刻度
plt.xticks(custom_xticks)

# 可选：旋转 x 轴刻度标签以提高可读性
plt.xticks(rotation=45)

# 显示图形
plt.tight_layout()
plt.show()
