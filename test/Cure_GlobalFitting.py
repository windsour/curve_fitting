import numpy as np
from scipy.optimize import least_squares
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

# 气体常数
R = 8.314  # J/(mol·K)

# 动力学微分方程：dalpha/dt = A * exp(-E/(R*T)) * (1 - alpha)^n
def d_alpha_dt(alpha, T, E, n, A):
    return A * np.exp(-E / (R * T)) * (1 - alpha)**n

# 数值积分求解 alpha(t)，使用前向欧拉法
def integrate_alpha(t, T, E, n, A, alpha0=0.0):
    alpha = np.zeros_like(t)
    alpha[0] = alpha0
    for i in range(1, len(t)):
        dt = t[i] - t[i-1]
        d_alpha = d_alpha_dt(alpha[i-1], T, E, n, A) * dt
        alpha[i] = alpha[i-1] + d_alpha
        if alpha[i] >= 1.0:
            alpha[i] = 1.0
    return alpha      #输出一个存储alpha数据的数组

# 全局模型预测函数（用于 least_squares）
def model_prediction(params, t_list, T_list, alpha0_list=None):
    """
    params: [E, n, A]
    t_list: list of time arrays
    T_list: list of temperatures (scalar for each run)
    alpha0_list: optional list of initial alpha values
    """
    E, n, A = params
    predictions = []
    for i in range(len(t_list)):
        t = t_list[i]        #此处的t_list为列表，只有t_list[0]、t_list[1]和t_list[2]
        T = T_list[i]
        alpha0 = alpha0_list[i] if alpha0_list is not None else 0.0   #优先使用提供的alpha初始值，否则默认为alpha初始值零
        pred = integrate_alpha(t, T, E, n, A, alpha0)
        predictions.append(pred)
    return np.concatenate(predictions)      #输出一个整合了所有alpha数据的数组

# ========================
# 1. 输入三组实验数据
# ========================

# --- 55°C = 328.15 K ---
t_55 = np.array([
    1544.868, 9931.356, 21186.864, 38180.484, 50098.104, 61574.292,
    72609.12, 82099.08, 91809.72, 100637.568, 108803.34, 118072.584,
    125576.28, 134404.128, 142569.9, 150073.56, 159342.804, 167729.292,
    179426.196, 190461.024, 205026.984, 219151.548, 233938.188, 246076.524,
    257994.108, 275208.444, 299485.044, 326630.7, 341417.376, 360838.656,
    385335.936, 400564.008, 416454.156, 429475.248, 444924.0
])  # seconds
alpha_55 = np.array([
    0.00998, 0.01774, 0.04435, 0.09093, 0.13972, 0.19073, 0.24284, 0.29496,
    0.34597, 0.39919, 0.45131, 0.49456, 0.54667, 0.59768, 0.64315, 0.69748,
    0.74738, 0.79173, 0.84718, 0.89153, 0.93478, 0.95806, 0.97248, 0.98246,
    0.98579, 0.98690, 0.99355, 0.99355, 0.99133, 0.99688, 0.99355, 0.99355,
    0.99355, 0.99688, 0.99688
])
T_55 = 328.15  # 55 + 273.15
# print("55摄氏度时的数据数:","\nt_55:",len(t_55),"\nalpha_55:",len(alpha_55),"\n")



# --- 60°C = 333.15 K ---
t_60 = np.array([0,8827.848,17435.016,26483.58,35311.428,43256.484,50539.464,
    58484.556,64443.348,70622.856,75698.856,81657.684,87616.476,93795.984,99092.7,
    105934.284,112334.472,120279.564,129548.808,139700.844,152059.824,162211.86,
    174350.16,186709.176,198626.76,212972.04,226655.208,234600.3,246297.204,
    264835.692,283815.612,302133.384,318906.324,338989.716,358631.676,375625.296,
    394384.5,411819.516,425723.4,442717.02
])
alpha_60 = np.array([
    0.00776, 0.01885, 0.04325, 0.09093, 0.14083, 0.1874, 0.24395, 0.29829,
    0.34375, 0.39808, 0.44355, 0.49234, 0.54667, 0.59879, 0.64758, 0.70302,
    0.75625, 0.80837, 0.86492, 0.90817, 0.93921, 0.95585, 0.96804, 0.9747,
    0.97802, 0.97581, 0.97802, 0.98357, 0.98246, 0.98579, 0.98579, 0.98579,
    0.98911, 0.99022, 0.99355, 0.99133, 0.99133, 0.99466, 0.99466, 0.99688
])
T_60 = 333.15
# print("60摄氏度时的数据数:","\nt_60:",len(t_60),"\nalpha_60:",len(alpha_60),"\n")






# --- 65°C = 338.15 K ---
t_65 = np.array([
    0, 11034.828, 18979.884, 25600.788, 32000.976, 39063.276, 43918.596,
    49215.312, 54512.028, 59367.348, 64443.348, 68857.272, 73933.308,
    79230.024, 85188.816, 90926.928, 98210.908, 108582.624, 126238.356,
    146321.712, 167729.292, 186488.460, 207454.644, 233938.188, 257552.712,
    273663.576, 289112.292, 305002.440, 321113.304, 338769.000, 350245.224,
    372314.844, 387542.916, 402108.876, 429254.532, 441392.832
])
alpha_65 = np.array([
    0.01109, 0.04879, 0.09647, 0.14859, 0.1996, 0.25171, 0.2994, 0.35151,
    0.39808, 0.4502, 0.49788, 0.54667, 0.59879, 0.64647, 0.70081, 0.75071,
    0.79839, 0.8505, 0.90151, 0.92036, 0.93367, 0.93921, 0.94587, 0.9503,
    0.95696, 0.96139, 0.96472, 0.96804, 0.96915, 0.9747, 0.97802, 0.98024,
    0.98357, 0.98579, 0.98911, 0.99022
])
# print("65摄氏度时的数据数:","\nt_65:",len(t_65),"\nalpha_65:",len(alpha_65))





T_65 = 338.15

# ========================
# 组织数据列表
# ========================
t_list = [t_55, t_60, t_65]
alpha_list = [alpha_55, alpha_60, alpha_65]
T_list = [T_55, T_60, T_65]

# 设置每个实验的初始 alpha
alpha0_list = [alpha_55[0], alpha_60[0], alpha_65[0]]

# 拼接真实数据用于比较
y_true = np.concatenate(alpha_list)

# ========================
# 初始猜测和边界
# ========================
initial_guess = [70000, 2, 1e4]  # E, n, A
bounds = ([0, 0, 0], [200000, 5.0, 1e15])

# ========================
# 定义残差函数
# ========================
def residuals(params):
    print("Current params:", params)    #输出每次迭代得到的E、n、A值
    pred = model_prediction(params, t_list, T_list, alpha0_list)
    return pred - y_true

# ========================
# 执行全局拟合
# ========================
result = least_squares(residuals, initial_guess, bounds=bounds, ftol=1e-8, xtol=1e-8)
E_fit, n_fit, A_fit = result.x

print(f"\n✅ 全局拟合完成！")
print(f"活化能 E = {E_fit:.2f} J/mol")
print(f"反应级数 n = {n_fit:.2f}")
print(f"指前因子 A = {A_fit:.2e} 1/s")
print(f"拟合残差平方和 (cost): {result.cost:.6f}")

# ========================
# 计算 R²（总）
# ========================
y_pred = model_prediction([E_fit, n_fit, A_fit], t_list, T_list, alpha0_list)
ss_res = np.sum((y_true - y_pred) ** 2)
ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
r2 = 1 - ss_res / ss_tot
print(f"R² (global) = {r2:.6f}")

# ========================
# 绘图
# ========================
plt.figure(figsize=(10, 7))
colors = ['tab:blue', 'tab:orange', 'tab:green']
labels = ['55°C (328.15 K)', '60°C (333.15 K)', '65°C (338.15 K)']

for i in range(3):
    t = t_list[i]
    exp = alpha_list[i]
    sim = integrate_alpha(t, T_list[i], E_fit, n_fit, A_fit, alpha0_list[i])

    plt.plot(t, exp, 'o', color=colors[i], label=f'Exp - {labels[i]}', markersize=4)
    plt.plot(t, sim, '-', color=colors[i], alpha=0.8, label=f'Fit - {labels[i]}')

plt.xlabel('Time (s)')
plt.ylabel('Conversion alpha')
# plt.title('Global Kinetic Fitting Across 3 Temperatures\nModel: $dalpha/dt = A \cdot e^{-E/(RT)} \cdot (1-alpha)^n$')
plt.title(r'Global Kinetic Fitting Across 3 Temperatures\nModel: $dalpha/dt = A \cdot e^{-E/(RT)} \cdot (1-alpha)^n$')
plt.legend()
plt.grid(True, alpha=0.5)
plt.tight_layout()
plt.show()

# ========================
# 输出拟合参数总结
# ========================
print("\n📊 拟合参数总结:")
print(f"E  = {E_fit:8.2f} ± ? J/mol     (需通过协方差或 bootstrap 估计误差)")
print(f"n  = {n_fit:8.2f}")
print(f"A  = {A_fit:8.2e} 1/s")
print(f"R² = {r2:8.6f}")



# print("params",params)
