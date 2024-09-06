import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt


class DataNormalization:
    def __init__(self, file_path):
        self.file_path = file_path

    def data_reading(self):
        data = pd.read_csv(self.file_path)
        return data

    def trains(self):
        df = self.data_reading()
        X = df[['LAT', 'LON', 'h']]
        y = df['Difference in N']

        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Normalizing LAT and LON using sinusoidal projections
        x_train[['LAT', 'LON']] = np.sin(np.radians(x_train[['LAT', 'LON']]))
        x_test[['LAT', 'LON']] = np.sin(np.radians(x_test[['LAT', 'LON']]))

        # Scale features using MinMaxScaler
        min_max_scaler = MinMaxScaler()
        x_train = min_max_scaler.fit_transform(x_train)
        x_test = min_max_scaler.transform(x_test)

        # Train the model
        model = LinearRegression()
        model.fit(x_train, y_train)

        # Make predictions
        y_pred_normalized = model.predict(x_test)

        # Denormalize predictions
        y_pred = y_pred_normalized * (y.max() - y.min()) + y.min()
        y_test_denormalized = y_test * (y.max() - y.min()) + y.min()

        # Evaluate the model
        mse = mean_squared_error(y_test_denormalized, y_pred)
        mae = mean_absolute_error(y_test_denormalized, y_pred)
        r2 = r2_score(y_test_denormalized, y_pred)

        return mse, mae, r2, y_pred, y_test_denormalized

    def plotting(self):
        mse, mae, r2, y_pred, y_test = self.trains()

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Scatter plot
        axes[0].scatter(y_test, y_pred)
        axes[0].set_xlabel('True Values')
        axes[0].set_ylabel('Predictions')
        axes[0].set_title('True vs Predicted Values')

        # Bar plot for evaluation metrics
        metrics = ['MSE', 'MAE', 'R2']
        values = [mse, mae, r2]
        bars = axes[1].bar(metrics, values)
        axes[1].set_title('Evaluation Metrics')

        # Adding values on top of each bar
        for bar, value in zip(bars, values):
            yval = bar.get_height()
            axes[1].text(bar.get_x() + bar.get_width() / 2.0, yval, round(value, 4),
                         va='bottom')  # va: vertical alignment

        plt.tight_layout()
        plt.savefig(r"E:\VICENTIA\YAW\NGL\hybrid_geoid_sample_data.png")
        plt.close()
     

# Example usage:
calling = DataNormalization(r"E:\VICENTIA\YAW\NGL\hybrid_geoid_sample_data.csv")
calling.plotting()