#!/usr/bin/env python3
"""
RX2 實驗：升溫與降溫曲線分析
分析不同電壓(5-10V)下的熱容與對流換熱參數
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.optimize import curve_fit

# ── 字體設定（避免中文方塊）──────────────────────────────────
mpl.rcParams['font.family'] = ['Arial', 'DejaVu Sans']
mpl.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 150

# ── 讀取數據 ──────────────────────────────────────────────────
df = pd.read_csv('/Users/ryanwong/Desktop/RX2/Temperture_Data.csv',
                 skiprows=1)  # skip "表格 1" header
df.columns = ['t', 'T5', 'T6', 'T7', 'T8', 'T9', 'T10']
time = df['t'].values.astype(float)
voltages = [5, 6, 7, 8, 9, 10]
cols   = ['T5', 'T6', 'T7', 'T8', 'T9', 'T10']

# ── 物理參數 ─────────────────────────────────────────────────
T_e = float(df['T5'].iloc[0])   # 室溫 (所有通道 t=0 均為 25.6°C)
R   = 400.0                      # 電阻 Ω  (max 0.25 W @ 10 V → R=V²/P)
t_off = 180                      # 斷電時刻 (s)

heat_mask = time <= t_off
cool_mask = time >= t_off
t_heat = time[heat_mask]
t_cool = time[cool_mask]
t_cool_rel = t_cool - t_off     # 降溫相對時間

# ── 指數模型 ─────────────────────────────────────────────────
def heating_model(t, T_inf, tau):
    """T(t) = T_e + (T_inf - T_e)*(1 - exp(-t/tau))"""
    return T_e + (T_inf - T_e) * (1 - np.exp(-t / tau))

def cooling_model(t_rel, T_peak, tau):
    """T(t) = T_e + (T_peak - T_e)*exp(-t_rel/tau)"""
    return T_e + (T_peak - T_e) * np.exp(-t_rel / tau)

# ── 擬合各電壓的降溫曲線，提取 τ ─────────────────────────────
tau_vals   = {}
T_inf_vals = {}

print(f"室溫 T_e = {T_e:.2f} °C\n")
print(f"{'V':>4}  {'τ_cool (s)':>12}  {'T_inf (°C)':>12}  {'ΔT_inf (°C)':>13}")
print("-" * 50)

for V, col in zip(voltages, cols):
    T_all   = df[col].values.astype(float)
    T_c     = T_all[cool_mask]
    T_peak0 = T_c[0]

    # 降溫擬合
    popt_c, _ = curve_fit(
        cooling_model, t_cool_rel, T_c,
        p0=[T_peak0, 60],
        bounds=([T_e, 5], [100, 300])
    )
    T_peak_fit, tau_c = popt_c

    # 由降溫τ外推真正穩態溫度
    # 升溫 180 s ≠ ∞, 修正: T_inf_true = T_e + ΔT_peak/(1-exp(-t_off/τ))
    T_inf_true = T_e + (T_peak_fit - T_e) / (1 - np.exp(-t_off / tau_c))

    tau_vals[V]   = tau_c
    T_inf_vals[V] = T_inf_true

    print(f"{V:>4}V  {tau_c:>12.1f}  {T_inf_true:>12.2f}  {T_inf_true-T_e:>13.2f}")

tau_mean = np.mean(list(tau_vals.values()))
tau_std  = np.std(list(tau_vals.values()))
print(f"\n平均時間常數 τ = ({tau_mean:.1f} ± {tau_std:.1f}) s\n")

# ── 由穩態溫差計算 hA 和 ρVCp ────────────────────────────────
print(f"假設電阻 R = {R:.0f} Ω\n")
print(f"{'V':>4}  {'Q (mW)':>10}  {'ΔT_inf':>10}  {'hA (mW/°C)':>14}  {'ρVCp (J/°C)':>14}")
print("-" * 60)

hA_vals = {}
rho_Cp_vals = {}
for V in voltages:
    Q = V**2 / R * 1000          # mW
    dT = T_inf_vals[V] - T_e
    hA = Q / dT                  # mW/°C
    rho_Cp = tau_mean * hA / 1000  # J/°C
    hA_vals[V]    = hA
    rho_Cp_vals[V] = rho_Cp
    print(f"{V:>4}V  {Q:>10.1f}  {dT:>10.2f}  {hA:>14.3f}  {rho_Cp:>14.4f}")

hA_mean    = np.mean(list(hA_vals.values()))
rho_Cp_mean = np.mean(list(rho_Cp_vals.values()))
print(f"\n平均 hA = {hA_mean:.3f} mW/°C = {hA_mean/1000:.5f} W/°C")
print(f"平均 ρVCp = τ·hA = {tau_mean:.1f} × {hA_mean/1000:.5f} = {rho_Cp_mean:.4f} J/°C")

# ═══════════════════════════════════════════════════════════════
# 圖 1：升溫與降溫曲線（雙子圖）
# ═══════════════════════════════════════════════════════════════
colors = plt.cm.plasma(np.linspace(0.1, 0.85, 6))
V_labels = [f'{V} V' for V in voltages]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5), sharey=False)

for i, (V, col) in enumerate(zip(voltages, cols)):
    T_all = df[col].values.astype(float)
    ax1.plot(t_heat, T_all[heat_mask], color=colors[i], lw=1.6,
             label=V_labels[i])
    ax2.plot(t_cool_rel, T_all[cool_mask], color=colors[i], lw=1.6,
             label=V_labels[i])

ax1.axhline(T_e, color='gray', ls='--', lw=0.8, alpha=0.7)
ax1.text(5, T_e + 0.5, f'$T_e$ = {T_e:.1f} °C', fontsize=8, color='gray')
ax1.set_xlabel('Time (s)', fontsize=11)
ax1.set_ylabel('Temperature (°C)', fontsize=11)
ax1.set_title('(a) Heating curves (0 – 180 s)', fontsize=11)
ax1.legend(fontsize=8, loc='lower right')
ax1.grid(True, alpha=0.3)
ax1.set_xlim(0, 180)

ax2.axhline(T_e, color='gray', ls='--', lw=0.8, alpha=0.7)
ax2.text(3, T_e + 0.5, f'$T_e$ = {T_e:.1f} °C', fontsize=8, color='gray')
ax2.set_xlabel('Time after power-off (s)', fontsize=11)
ax2.set_ylabel('Temperature (°C)', fontsize=11)
ax2.set_title('(b) Cooling curves (180 – 360 s)', fontsize=11)
ax2.legend(fontsize=8, loc='upper right')
ax2.grid(True, alpha=0.3)
ax2.set_xlim(0, 180)

plt.tight_layout()
plt.savefig('/Users/ryanwong/Desktop/RX2/fig_heat_cool.pdf', bbox_inches='tight')
plt.savefig('/Users/ryanwong/Desktop/RX2/fig_heat_cool.png', dpi=200, bbox_inches='tight')
plt.close()
print("\n[圖1] fig_heat_cool 已保存")

# ═══════════════════════════════════════════════════════════════
# 圖 2：降溫指數擬合（示意 10V 曲線），對數坐標顯示τ
# ═══════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

# 左：所有電壓降溫 + 擬合曲線
ax = axes[0]
t_fit_arr = np.linspace(0, 180, 500)
for i, (V, col) in enumerate(zip(voltages, cols)):
    T_all = df[col].values.astype(float)
    T_c   = T_all[cool_mask]
    tau_c = tau_vals[V]
    T_pk  = T_c[0]
    T_curve = cooling_model(t_fit_arr, T_pk, tau_c)
    ax.plot(t_cool_rel, T_c, 'o', ms=2.5, color=colors[i], alpha=0.5)
    ax.plot(t_fit_arr, T_curve, '-', color=colors[i], lw=1.8,
            label=f'{V} V  τ={tau_c:.0f} s')

ax.axhline(T_e, color='gray', ls='--', lw=0.8)
ax.set_xlabel('Time after power-off (s)', fontsize=11)
ax.set_ylabel('Temperature (°C)', fontsize=11)
ax.set_title('(a) Exponential fits to cooling curves', fontsize=11)
ax.legend(fontsize=8, loc='upper right')
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 180)

# 右：ln(T-Te) vs t → 斜率 = -1/τ (以 10V 為例)
ax = axes[1]
for i, (V, col) in enumerate(zip(voltages, cols)):
    T_all = df[col].values.astype(float)
    T_c   = T_all[cool_mask]
    theta = T_c - T_e
    # 排除 theta ≤ 0
    mask = theta > 0.05
    t_plot = t_cool_rel[mask]
    ln_theta = np.log(theta[mask])
    ax.plot(t_plot, ln_theta, '-', color=colors[i], lw=1.6,
            label=f'{V} V')

ax.set_xlabel('Time after power-off (s)', fontsize=11)
ax.set_ylabel(r'$\ln(T - T_e)$', fontsize=11)
ax.set_title(r'(b) $\ln(T-T_e)$ vs. time (slope = $-1/\tau$)', fontsize=11)
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 180)

plt.tight_layout()
plt.savefig('/Users/ryanwong/Desktop/RX2/fig_cooling_fit.pdf', bbox_inches='tight')
plt.savefig('/Users/ryanwong/Desktop/RX2/fig_cooling_fit.png', dpi=200, bbox_inches='tight')
plt.close()
print("[圖2] fig_cooling_fit 已保存")

# ═══════════════════════════════════════════════════════════════
# 圖 3：穩態溫差 ΔT∞ vs V²（線性驗證）
# ═══════════════════════════════════════════════════════════════
V_arr  = np.array(voltages, dtype=float)
V2_arr = V_arr**2
dT_arr = np.array([T_inf_vals[V] - T_e for V in voltages])

# 線性擬合 ΔT = (1/R·hA)·V²  (過原點)
slope_force, = np.linalg.lstsq(V2_arr[:, None], dT_arr, rcond=None)[0]
hA_from_fit  = 1000 / (R * slope_force)   # mW/°C
rho_Cp_fit   = tau_mean * hA_from_fit / 1000  # J/°C

fig, ax = plt.subplots(figsize=(5.5, 4.5))
V2_fine = np.linspace(0, 110, 300)
ax.scatter(V2_arr, dT_arr, s=60, zorder=5, color='steelblue', label='Measured $\\Delta T_\\infty$')
ax.plot(V2_fine, slope_force * V2_fine, 'r-', lw=1.8,
        label=f'Linear fit (through origin)\nSlope = {slope_force:.3f} °C/V²')
ax.set_xlabel('$V^2$ (V$^2$)', fontsize=12)
ax.set_ylabel('$\\Delta T_\\infty = T_\\infty - T_e$ (°C)', fontsize=12)
ax.set_title('Steady-state temperature rise vs. heating power\n'
             f'(R = {R:.0f} Ω  →  $hA$ = {hA_from_fit:.2f} mW/°C)', fontsize=10)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 110)
ax.set_ylim(0, 50)

# 標注各電壓點
for V, dT in zip(voltages, dT_arr):
    ax.annotate(f'{V} V', (V**2, dT), textcoords='offset points',
                xytext=(4, 4), fontsize=8, color='steelblue')

plt.tight_layout()
plt.savefig('/Users/ryanwong/Desktop/RX2/fig_steady_state.pdf', bbox_inches='tight')
plt.savefig('/Users/ryanwong/Desktop/RX2/fig_steady_state.png', dpi=200, bbox_inches='tight')
plt.close()
print("[圖3] fig_steady_state 已保存")

# ── 最終摘要 ─────────────────────────────────────────────────
print("\n" + "="*60)
print("  最終估計結果（R = 400 Ω）")
print("="*60)
print(f"  時間常數      τ  = ({tau_mean:.1f} ± {tau_std:.1f}) s")
print(f"  對流換熱係數  hA = {hA_from_fit:.2f} mW/°C  =  {hA_from_fit/1000:.5f} W/°C")
print(f"  電阻熱容      ρVCp = τ · hA = {rho_Cp_fit:.4f} J/°C")
print("="*60)
