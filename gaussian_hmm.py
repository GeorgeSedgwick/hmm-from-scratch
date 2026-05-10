from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler
import pandas as pd
import os
import numpy as np
import plotly.graph_objects as go

# Wrap the process in a loop, storing BIC for each number of components tested:

# Initialise and configure the GaussianHMM

# Params:
# n_components=3 : 4 states of market regime: Bull/Bear/WeakBull/Recovery
# covariance_type="full" : each state uses a full, unrestricted, covariance matrix
# n_iter : the maximum number of iterations performed, if converged=False, increase.
model = GaussianHMM(n_components=4, covariance_type="full", n_iter=100, random_state=1)
scaler = StandardScaler()


# Set filepath of CSV
csv_dir = "~/python-projects/hmm/datasets/SPY.csv"
filepath = os.path.join(csv_dir)

# Read csv data into DataFrame
df = pd.read_csv(filepath)

# Construct a new columns for returns
df['Returns'] = df['Close'].pct_change()

# Construct a new column for Mean Returns
df['1_Month_Mean_Returns'] = df['Returns'].rolling(window=21).mean()

# Construct a new column for Vol
df['Volatility'] = df['Returns'].rolling(window=21).std()

# Construct a new column for Skewness
df['Skew'] = df['Returns'].rolling(window=21).skew()

# Construct a new columns for 5 day momentum
df['5_Day_Momentum'] = df['Close'].pct_change(5)
#print(df.isin([np.inf, -np.inf]).sum())
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(inplace=True)
#print(df.isin([np.inf, -np.inf]).sum())
#print(f'Total NaNs after: {df.isna().sum()}')

# Momentum had a min of -16 and a max of 58, 58 is an outlier and causing the model fit to be obscured (No bull regimes identified)
# To fix this, I must remove outliers beyond a certain threshold (potentially 3 std)
mean = df['5_Day_Momentum'].mean()
momentum_std = df['5_Day_Momentum'].std()

df['5_Day_Momentum'] = np.where(df['5_Day_Momentum'] > mean + momentum_std * 3, mean + momentum_std * 2, df['5_Day_Momentum'])
df['5_Day_Momentum'] = np.where(df['5_Day_Momentum'] < mean - momentum_std * 3, mean - momentum_std * 2, df['5_Day_Momentum'])

# Checking the mean of 5_Day_Momentum as groupby Regime was showing all negative momentum
print(df['5_Day_Momentum'].mean())

# AFTER THESE CHANGES:

# Regime 0: Strong Bull
# Regime 1: Bear
# Regime 2: Recovery
# Regime 3: Flat

# Drop NaNs
df = df.dropna()

# Prepare and scale features (X)
X = df[['1_Month_Mean_Returns', 'Volatility', 'Skew', '5_Day_Momentum']]
X_new = scaler.fit_transform(X)

#print(X_new)

model.fit(X_new)

# Check if convergence occurred
print(f"Convergence Occurred: {model.monitor_.converged}")

hidden_states = model.predict(X_new)

# Bayesian Information Critereon
bic = model.bic(X_new)



df['Regime'] = hidden_states
print(df.groupby('Regime')['Returns'].mean())
print(df.groupby('Regime')['Volatility'].mean())
print(df.groupby('Regime')['Skew'].mean())
print(df.groupby('Regime')['5_Day_Momentum'].mean())


# Check how many total regime switches there are
print(f"Regime switches before smoothing: {df['Regime'].diff().ne(0).sum()}")

# 2705 regime switches, nearly every other day it flips
# Need to smooth this

# To do that, I use the following
df['Regime'] = df['Regime'].rolling(10).apply(lambda x: x.mode()[0])

# Check again
print(f"Regime switches after smoothing: {df['Regime'].diff().ne(0).sum()}")
# At 10-day smoothing, 214 flips occur, HUGE DIFF

# Forward Fill and Backward Fill to remove NaNs but maintain indexing, change datatype to int
df['Regime'] = df['Regime'].ffill().bfill().astype(int)


# Check contents of X_new
print(X_new.min(axis=0), X_new.max(axis=0))


# Plot the regimes on price chart
fig = go.Figure()

fig.add_trace(
    go.Scatter(
        x=df.Date,
        y=df.Close, 
        mode='lines', 
        name="Close",
        line=dict(color='black')))


start_index = 0
current_regime = df['Regime'].iloc[0]

colours = {
    0: "rgba(255, 0, 0, 0.15)",
    1: "rgba(0, 0, 255, 0.15)",
    2: "rgba(0, 255, 0, 0.15)",
    3: "rgba(255, 255, 0, 0.3)",
}

for i in range(1, len(df)):
    if df['Regime'].iloc[i] != current_regime:

        fig.add_vrect(
            x0=df['Date'].iloc[start_index],
            x1=df['Date'].iloc[i],
            fillcolor=colours[current_regime],
            opacity=0.5,
            line_width=0
        )
        
        start_index = i
        current_regime = df['Regime'].iloc[i]

fig.add_vrect(
    x0=df['Date'].iloc[start_index],
    x1=df['Date'].iloc[-1],
    fillcolor=colours[current_regime],
    opacity=0.5,
    line_width=0
)

fig.show()