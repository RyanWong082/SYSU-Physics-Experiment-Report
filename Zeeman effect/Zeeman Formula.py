import numpy as np

# 1. 定義常數與實驗數據 (請根據你實際嘅數據替換數值)
c = 299792458.0        # 光速 (m/s)
d = 0.00205            # 干涉儀兩塊反射鏡之間嘅距離 (m)
B = 0.769              # 磁場強度 (Tesla)

# 圓環直徑數據 (可以係單一數值，或者係一列數字)
# 將你嘅數據填入中括號內，例如 [1.2, 1.5, 1.8]
D_a = np.array([94])        # 替換成 D_a (即係圖中嘅 D_N) 嘅數據
D_b = np.array([125])        # 替換成 D_b 嘅數據
D_N = np.array([125])        # 替換成 D_N 嘅數據 (通常同 D_a 一樣)
D_N_minus_1 = np.array([244])# 替換成 D_N-1 嘅數據

# 2. 代入公式計算實驗得出嘅 e/m (實驗值)
term1 = (2 * np.pi * c) / (d * B)
term2 = (D_b**2 - D_a**2) / (D_N_minus_1**2 - D_N**2)
e_over_m_exp = term1 * term2

# 3. 計算相對誤差百分比
e_over_m_theo = 1.75881962e11  # e/m 嘅理論值 (C/kg)

# 使用 np.abs() 確保誤差係正數
relative_error = np.abs(e_over_m_exp - e_over_m_theo) / e_over_m_theo * 100

# 4. 顯示結果
print("--- 計算結果 ---")
print("實驗得出嘅 e/m 值 (C/kg):")
print(e_over_m_exp)
print("\n相對誤差百分比 (%):")
# round() 可以幫你控制顯示嘅小數點位數，例如 round(..., 2) 顯示兩位小數
print(np.round(relative_error, 2))