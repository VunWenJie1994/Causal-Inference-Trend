import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from patsy import dmatrix
import statsmodels.api as sm

# ============================
# Load data
# ============================
data = pd.read_excel(
    "C:/Users/vunji306/Desktop/Work/Work with Torbjorn/Temporal Modelling of Biodiversity Dynamics/"
    "2_GettingFellingTimeSeries/7_DirectRegressionApproach_DZ/7_3ChangeInBiodiversity/Dataset/T_index/"
    "Updated_df1_with_inserted_rows_T_index.xlsx"
)

# ============================
# Step 1: Compute y(t) = b(t) - b(0)
# ============================
def compute_y(df):
    df = df.copy()
    b0 = df[df['t_index'] == 0]['Species richness'].values[0]
    df['y_t'] = df['Species richness'] - b0
    return df

data = data.groupby('DelytaID').apply(compute_y).reset_index(drop=True)

# ============================
# Step 2: Create relative time variable "t"
# ============================
def compute_relative_t(df):
    df = df.copy()
    # Get baseline time where t_index == 0
    t0 = df.loc[df['t_index'] == 0, 'Time'].values[0]
    # Compute relative time
    df['t'] = df['Time'] - t0
    return df

data = data.groupby('DelytaID').apply(compute_relative_t).reset_index(drop=True)

# ============================
# Step 3: Compute t* for intervention plots
# ============================
def compute_t_star(row, ref_times):
    if row['AtgardUtford_Detail'] == 1:
        return row['t_intervention'] - ref_times[row['DelytaID']]
    else:
        return 999  # Large value for control plots

# Reference times at baseline (t_index = 0)
ref_times = data[data['t_index'] == 0].set_index('DelytaID')['Time'].to_dict()
data['t_star'] = data.apply(lambda row: compute_t_star(row, ref_times), axis=1)

# ============================
# Step 4: Create spline basis for f(t) and δ(t−t*)
# ============================
knots = [1, 2, 3.5, 8]

# f(t): spline basis for relative time
f_basis = dmatrix("bs(t, knots=knots, degree=1, include_intercept=False)",
                  {"t": data['t']}, return_type='dataframe')

# δ(t−t*): intervention effect spline
data['t_minus_t_star'] = data['t'] - data['t_star']
delta_basis = dmatrix("bs(t_minus_t_star, knots=knots, degree=1, include_intercept=False)",
                      {"t_minus_t_star": data['t_minus_t_star']}, return_type='dataframe')

# Combine design matrix
X = pd.concat([f_basis, delta_basis], axis=1)
X = sm.add_constant(X)

# ============================
# Step 5: Fit regression model
# ============================
model = sm.OLS(data['y_t'], X).fit()
data['predicted'] = model.predict(X)

# ============================
# Step 6: Plot results
# ============================
plt.figure(figsize=(10, 6))
for label, group in data.groupby('AtgardUtford_Detail'):
    label_name = 'Control' if label == 0 else 'Intervention'
    mean_vals = group.groupby('t')['predicted'].mean()
    plt.plot(mean_vals.index, mean_vals.values, label=label_name, marker='o')

plt.xlabel('Relative Time (t)')
plt.ylabel('Change in Species Richness (y(t))')
plt.title('Effect of Intervention on Species Richness Change')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()



############################################# Checking data


# Filter intervention plots
intervention_data = data[data['AtgardUtford_Detail'] == 1]

# Filter control plots
control_data = data[data['AtgardUtford_Detail'] == 0]

plt.figure(figsize=(12, 6))
# Intervention dot plots
for _, group in intervention_data.groupby('DelytaID'):
    plt.scatter(group['t_star'], group['y_t'], color='orange', alpha=0.6, label='Intervention, t*' if _ == intervention_data['DelytaID'].iloc[0] else "")

# Control plots
for _, group in control_data.groupby('DelytaID'):
    plt.scatter(group['t'], group['y_t'], color='blue', alpha=0.6, label='Control, t' if _ == control_data['DelytaID'].iloc[0] else "")

# Axis and layout
plt.ylim(-20, 20)
plt.xlim(0, 30)
plt.title('Biodiversity Change Over Time – Intervention vs Control Plots')
plt.xlabel('Relative Time (t)')
plt.ylabel('Change in Biodiversity (y(t))')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
