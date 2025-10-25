import requests
import time
from prophet import Prophet
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Querying CPU usage data from Prometheus API
PROMETHEUS_URL = "http://192.168.29.158:9090"
Query = '100 - (avg by (instance)(rate(node_cpu_seconds_total{mode="idle"}[5m])) * 100)'

end = int(time.time())
start = end - 3600
step = "30s"

url = f"{PROMETHEUS_URL}/api/v1/query_range?query={Query}&start={start}&end={end}&step={step}"
response = requests.get(url).json()

# Adding the extracted data into two columns as timestamp & cpu_usage
results = response["data"]["result"][0]["values"]
df = pd.DataFrame(results, columns=["timestamp", "cpu_usage"])
df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
df["cpu_usage"] = df["cpu_usage"].astype(float)

# Writing the data into a csv file
print(df.head())
df.to_csv("cpu_usage.csv", index=False)
print("\n✅ Saved cpu_usage.csv with", len(df), "rows.")

# Load the CPU usage data
df = pd.read_csv('cpu_usage.csv')
df['timestamp'] = pd.to_datetime(df['timestamp'])
df = df.sort_values('timestamp')

# Prepare for Prophet (requires columns 'ds' and 'y')
df_prophet = df.rename(columns={'timestamp': 'ds', 'cpu_usage': 'y'})

# Optional: visualize the trend before modeling
plt.figure(figsize=(10,5))
plt.plot(df_prophet['ds'], df_prophet['y'])
plt.title('CPU Usage (%) over Time')
plt.xlabel('Time')
plt.ylabel('CPU Usage (%)')
plt.show()

# Create and fit the model
model = Prophet(interval_width=0.95)
model.fit(df_prophet)

# Define how far into the future to forecast
future = model.make_future_dataframe(periods=20, freq='30S')
forecast = model.predict(future)

# Plot the forecast
model.plot(forecast)
plt.title('CPU Usage Forecast')
plt.xlabel('Time')
plt.ylabel('CPU Usage (%)')
plt.show()

# Optional: plot forecast components
model.plot_components(forecast)
plt.show()

# Merge actual and predicted
merged = pd.merge(df_prophet, forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']], on='ds', how='left')

# Calculate residual (difference)
merged['error'] = merged['y'] - merged['yhat']

### 🔹 CHANGED: Compute adaptive threshold based on recent volatility
window_size = max(3, int(len(merged) * 0.05))  # 5% of data length or min 3 points
merged['rolling_std'] = merged['error'].rolling(window=window_size).std()

### 🔹 CHANGED: Flag only SPIKES (actual much higher than predicted)
merged['pred_anomaly'] = np.where(merged['error'] > (2 * merged['rolling_std']), 1, 0)

### 🔹 CHANGED: Replace abs(error) > threshold → directional logic for spikes only
# This ignores negative deviations (drops)

# Visualize
plt.figure(figsize=(10,5))
plt.plot(merged['ds'], merged['y'], label='Actual CPU', color='steelblue')
plt.plot(merged['ds'], merged['yhat'], label='Predicted CPU', color='orange', linestyle='--')

# Highlight only spike anomalies
plt.scatter(merged.loc[merged['pred_anomaly'] == 1, 'ds'],
            merged.loc[merged['pred_anomaly'] == 1, 'y'],
            color='red', label='Spike Anomaly', marker='x', s=80)

plt.title('Predicted vs Actual CPU Usage — Spike Anomalies Only')
plt.xlabel('Time')
plt.ylabel('CPU Usage (%)')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()

print("✅ Spike anomalies detected:", merged['pred_anomaly'].sum())

import json
import requests

# Filter only spike anomalies
anomalies = merged[merged['pred_anomaly'] == 1]

if anomalies.empty:
    print("✅ No spike anomalies detected — all clear.")
else:
    print(f"🚨 {len(anomalies)} anomalies detected. Preparing JSON payloads...\n")

    # Convert each anomaly row to JSON
    for _, row in anomalies.iterrows():
        log_entry = {
            "timestamp": row['ds'].isoformat(),
            "cpu_usage": round(row['y'], 2),
            "predicted": round(row['yhat'], 2),
            "error": round(row['error'], 2),
            "message": f"Spike anomaly detected at {row['ds']} with CPU usage {round(row['y'], 2)}%"
        }

        print(json.dumps(log_entry, indent=2))

        # --- Optional: uncomment this to send to a webhook/Loki later ---
        # requests.post("http://<YOUR-WEBHOOK-URL>", json=log_entry)



