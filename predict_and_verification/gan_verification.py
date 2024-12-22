import os
import json
import pandas as pd
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Dense, LSTM, Concatenate, LeakyReLU
from tensorflow.keras.optimizers import Adam
import argparse


def parse_item_list(item_arg):
    """
    Parse item list from a single item name, a .txt file, or a .json file.
    """
    if os.path.isfile(item_arg):
        _, ext = os.path.splitext(item_arg)
        if ext == ".txt":
            with open(item_arg, "r", encoding="utf-8") as f:
                items = [line.strip() for line in f if line.strip()]
        elif ext == ".json":
            with open(item_arg, "r", encoding="utf-8") as f:
                items = json.load(f)
        else:
            raise ValueError("Unsupported file format. Use .txt or .json.")
    else:
        items = [item_arg]
    return items


def create_dataset(series, window_size=30):
    """
    Create supervised learning dataset for CGAN.
    """
    X, Y = [], []
    for i in range(len(series) - window_size):
        X.append(series[i:i + window_size])
        Y.append(series[i + window_size])
    return np.array(X), np.array(Y)


def build_generator(window_size, noise_dim):
    cond_input = Input(shape=(window_size, 1), name="condition_input")
    cond_feat = LSTM(50)(cond_input)
    noise_input = Input(shape=(noise_dim,), name="noise_input")
    x = Concatenate()([cond_feat, noise_input])
    x = Dense(50)(x)
    x = LeakyReLU(alpha=0.2)(x)
    out = Dense(1, activation='linear')(x)
    return Model([cond_input, noise_input], out, name="Generator")


def build_discriminator(window_size):
    cond_input = Input(shape=(window_size, 1), name="condition_input_d")
    cond_feat = LSTM(50)(cond_input)
    price_input = Input(shape=(1,), name="price_input_d")
    x = Concatenate()([cond_feat, price_input])
    x = Dense(50)(x)
    x = LeakyReLU(alpha=0.2)(x)
    out = Dense(1, activation='sigmoid')(x)
    return Model([cond_input, price_input], out, name="Discriminator")


def main():
    # Add argparse support
    parser = argparse.ArgumentParser(description="CGAN Verification")
    parser.add_argument("--item", type=str, required=True, help="Item name or path to a .txt/.json file containing item names.")
    args = parser.parse_args()

    # Parse item list
    items = parse_item_list(args.item)

    # Directories
    price_dir = "item_smooth_prices"
    output_dir = "consequence/verification/gan"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(f"{output_dir}/picture", exist_ok=True)
    os.makedirs(f"{output_dir}/comparison_curves", exist_ok=True)

    # Dates
    test_start = datetime(2024, 12, 1)
    test_end = datetime(2024, 12, 31)
    plot_start = datetime(2024, 11, 1)
    window_size = 30
    noise_dim = 10
    epochs = 100
    batch_size = 32

    for item_name in items:
        item_path = os.path.join(price_dir, f"{item_name}.csv")
        if not os.path.exists(item_path):
            print(f"{item_path} not found, skipping {item_name}")
            continue

        item_df = pd.read_csv(item_path, parse_dates=["date"], index_col="date")
        if item_df.empty:
            print(f"No data for {item_name}, skipping CGAN forecast.")
            continue

        train_df = item_df[item_df.index < test_start]
        test_df = item_df[(item_df.index >= test_start) & (item_df.index <= test_end)]

        if test_df.empty or len(train_df) < window_size:
            print(f"Not enough train or test data for {item_name}, skipping.")
            continue

        series = train_df['price'].values.reshape(-1, 1)
        scaler = MinMaxScaler((0, 1))
        scaled_train = scaler.fit_transform(series)

        X_train, Y_train = create_dataset(scaled_train, window_size)
        if len(X_train) == 0:
            print(f"Not enough training data after windowing for {item_name}, skipping.")
            continue

        D = build_discriminator(window_size)
        G = build_generator(window_size, noise_dim)

        d_optimizer = Adam(0.0002, 0.5)
        g_optimizer = Adam(0.0002, 0.5)

        dataset = tf.data.Dataset.from_tensor_slices((X_train, Y_train)).shuffle(len(X_train)).batch(batch_size)

        @tf.function
        def train_step(real_cond, real_price):
            noise = tf.random.normal([tf.shape(real_cond)[0], noise_dim])
            with tf.GradientTape() as tape_d, tf.GradientTape() as tape_g:
                fake_price = G([real_cond, noise], training=True)
                real_output = D([real_cond, real_price], training=True)
                fake_output = D([real_cond, fake_price], training=True)
                d_loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(tf.ones_like(real_output), real_output)) + \
                         tf.reduce_mean(tf.keras.losses.binary_crossentropy(tf.zeros_like(fake_output), fake_output))
                g_loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(tf.ones_like(fake_output), fake_output))

            gradients_of_d = tape_d.gradient(d_loss, D.trainable_variables)
            d_optimizer.apply_gradients(zip(gradients_of_d, D.trainable_variables))

            gradients_of_g = tape_g.gradient(g_loss, G.trainable_variables)
            g_optimizer.apply_gradients(zip(gradients_of_g, G.trainable_variables))

            return d_loss, g_loss

        for epoch in range(epochs):
            d_loss_list, g_loss_list = [], []
            for real_cond, real_price in dataset:
                d_loss, g_loss = train_step(real_cond, real_price)
                d_loss_list.append(d_loss.numpy())
                g_loss_list.append(g_loss.numpy())
            if (epoch + 1) % 20 == 0:
                print(f"[{item_name}] Epoch {epoch+1}/{epochs} D_loss: {np.mean(d_loss_list):.4f}, G_loss: {np.mean(g_loss_list):.4f}")

        # Test set prediction
        test_series = test_df['price'].values.reshape(-1, 1)
        scaled_test = scaler.transform(test_series)
        full_series = np.concatenate([scaled_train, scaled_test], axis=0)

        test_start_idx = len(scaled_train)
        real_values = test_df['price'].values
        test_dates = test_df.index

        cond = tf.convert_to_tensor(full_series[test_start_idx - window_size:test_start_idx].reshape(1, window_size, 1), dtype=tf.float32)
        future_preds = []
        for _ in range(len(test_df)):
            noise = tf.random.normal([1, noise_dim])
            pred_price = G([cond, noise], training=False)
            pred_val = pred_price[0][0]
            future_preds.append(pred_val.numpy())
            cond = tf.concat([cond[:, 1:, :], tf.reshape(pred_val, [1, 1, 1])], axis=1)

        future_preds = np.array(future_preds).reshape(-1, 1)
        future_preds_inv = scaler.inverse_transform(future_preds)

        # Metrics
        mse = mean_squared_error(real_values, future_preds_inv[:, 0])
        mae = mean_absolute_error(real_values, future_preds_inv[:, 0])
        rmse = np.sqrt(mse)

        print(f"{item_name} - December CGAN forecast metrics:")
        print(f"MSE: {mse:.4f}, MAE: {mae:.4f}, RMSE: {rmse:.4f}")

        # Save results
        result_df = pd.DataFrame({
            "date": test_dates,
            "real_price": real_values,
            "predicted_price": future_preds_inv[:, 0]
        })
        result_out_path = os.path.join(output_dir, f"compare_{item_name}.csv")
        result_df.to_csv(result_out_path, index=False, date_format='%Y-%m-%d')
        print(f"✅ CGAN comparison saved: {result_out_path}")

        text_str = f"MSE: {mse:.4f}\nMAE: {mae:.4f}\nRMSE: {rmse:.4f}"

        # Plot 1: December range
        plt.figure(figsize=(10, 5))
        plt.plot(test_dates, real_values, label='Real Price', color='blue')
        plt.plot(test_dates, future_preds_inv[:, 0], label='Predicted Price (CGAN)', color='purple')
        plt.title(f"{item_name} December CGAN Forecast vs Real")
        plt.xlabel("Date")
        plt.ylabel("Price")
        plt.legend()
        plt.text(test_dates.min(), max(real_values.max(), future_preds_inv[:, 0].max()) * 0.9, text_str,
                 fontsize=10, bbox=dict(facecolor='white', alpha=0.7))

        picture_path = os.path.join(f"{output_dir}/picture", f"{item_name}_december_comparison.png")
        plt.savefig(picture_path, dpi=100)
        plt.close()
        print(f"✅ December comparison plot saved: {picture_path}")

        # Plot 2: November to December range
        plt.figure(figsize=(10, 5))
        extended_plot_df = item_df.loc[(item_df.index >= plot_start) & (item_df.index <= test_end)]
        plt.plot(extended_plot_df.index, extended_plot_df['price'], label='Historical Price', color='blue')
        plt.plot(test_dates, future_preds_inv[:, 0], label='Predicted Price (CGAN)', color='purple')
        plt.title(f"{item_name} November to December CGAN Forecast vs Real")
        plt.xlabel("Date")
        plt.ylabel("Price")
        plt.legend()
        plt.text(plot_start, max(extended_plot_df['price'].max(), future_preds_inv[:, 0].max()) * 0.9, text_str,
                 fontsize=10, bbox=dict(facecolor='white', alpha=0.7))

        picture_path_nov = os.path.join(f"{output_dir}/picture", f"{item_name}_november_to_december_comparison.png")
        plt.savefig(picture_path_nov, dpi=100)
        plt.close()
        print(f"✅ November to December comparison plot saved: {picture_path_nov}")

        # Save metrics
        metrics_path = os.path.join(f"{output_dir}/comparison_curves", f"{item_name}_metrics.txt")
        with open(metrics_path, "w", encoding="utf-8") as mf:
            mf.write(f"MSE: {mse:.4f}\nMAE: {mae:.4f}\nRMSE: {rmse:.4f}\n")
        print(f"✅ Metrics saved: {metrics_path}")


if __name__ == "__main__":
    main()
