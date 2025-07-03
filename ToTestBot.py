import logging
import pandas as pd
from tvDatafeed import TvDatafeed, Interval
from indian_companies import indian_companies  # Import the list of Indian companies
from time import sleep
from datetime import datetime, timedelta
import numpy as np
import time
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, accuracy_score
from telegram import Bot
import asyncio

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Your bot token and channel chat ID
TOKEN = '7547264968:AAHkkVIah6x283RR-QRIFb7Xj7Gu3KinOxU'
CHAT_ID = '@torioxindianstock' 
bot = Bot(token=TOKEN)

async def send_message(message):
    await bot.send_message(chat_id=CHAT_ID, text=message)

def send_message_sync(message):
    bot.send_message(chat_id=CHAT_ID, text=message)
    """Sends a message synchronously by managing the event loop."""
    try:
        # Get the current event loop or create a new one if needed
        loop = asyncio.get_event_loop()
    except RuntimeError as e:
        if 'There is no current event loop in thread' in str(e):
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        else:
            raise
    
    if loop.is_closed():
        # Create a new loop if the current one is closed
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    loop.run_until_complete(send_message(message))

def calculate_andean_oscillator(data):
    """Calculate Andean Oscillator."""
    data['Momentum'] = data['close'] - data['close'].shift(10)
    data['Andean_Oscillator'] = data['Momentum'].rolling(window=10).mean()
   
    # Example of using different conditions
    data['buySignal'] = data['Andean_Oscillator'] > 0
    data['sellSignal'] = data['Andean_Oscillator'] < 0
   
    return data
 
def calculate_trend_signal(df):
    """Calculate the Trend Signal Indicator."""
    def percent(nom, div):
        return 100 * nom / div

    def f1(m, n):
        return m if m >= n else 0.0

    def f2(m, n):
        return 0.0 if m >= n else -m

    # Ensure df is valid
    if df is None or df.empty:
        print("DataFrame is None or empty.")
        return df

    # Input parameters
    Multiplier = 2.5
    atrPeriods = 14
    atrCalcMethod = "Method 1"
    stopLossVal = 0.5

    df['hl2'] = (df['high'] + df['low']) / 2
    df['src1'] = df['open'].rolling(window=5).mean().shift(1)
    df['src2'] = df['close'].rolling(window=12).mean()
    df['momm1'] = df['src1'].diff()
    df['momm2'] = df['src2'].diff()
    df['m1'] = df.apply(lambda row: f1(row['momm1'], row['momm2']), axis=1)
    df['m2'] = df.apply(lambda row: f2(row['momm1'], row['momm2']), axis=1)
    df['sm1'] = df['m1'].rolling(window=1).sum()
    df['sm2'] = df['m2'].rolling(window=1).sum()

    df['cmoCalc'] = percent(df['sm1'] - df['sm2'], df['sm1'] + df['sm2'])

    df['hh'] = df['high'].rolling(window=2).max()
    df['hpivot'] = df['hh'].shift(2)
    df['ll'] = df['low'].rolling(window=2).min()
    df['lpivot'] = df['ll'].shift(2)

    df['rsiCalc'] = df['close'].rolling(window=9).apply(lambda x: 100 - (100 / (1 + (x.diff().clip(lower=0).mean() / x.diff().clip(upper=0).abs().mean()))))

    df['sup'] = (df['rsiCalc'] < 25) & (df['cmoCalc'] > 50) & df['lpivot'].notna()
    df['res'] = (df['rsiCalc'] > 75) & (df['cmoCalc'] < -50) & df['hpivot'].notna()

    df['atr'] = df['close'].rolling(window=atrPeriods).apply(lambda x: x.max() - x.min())
    df['atr2'] = df['atr'].rolling(window=atrPeriods).mean()
    df['atr'] = df['atr'] if atrCalcMethod == "Method 1" else df['atr2']

    df['up'] = df['hl2'] - (Multiplier * df['atr'])
    df['up'] = df['up'].where(df['close'].shift(1) > df['up'].shift(1), np.maximum(df['up'], df['up'].shift(1)))

    df['dn'] = df['hl2'] + (Multiplier * df['atr'])
    df['dn'] = df['dn'].where(df['close'].shift(1) < df['dn'].shift(1), np.minimum(df['dn'], df['dn'].shift(1)))

    df['trend'] = 1
    df['trend'] = df['trend'].where((df['trend'].shift(1) != -1) | (df['close'] <= df['dn'].shift(1)), 1)
    df['trend'] = df['trend'].where((df['trend'].shift(1) != 1) | (df['close'] >= df['up'].shift(1)), -1)

    df['buySignal'] = (df['trend'] == 1) & (df['trend'].shift(1) == -1)
    df['sellSignal'] = (df['trend'] == -1) & (df['trend'].shift(1) == 1)

    df['pos'] = 0.0
    df['pos'] = df['buySignal'].astype(float).where(df['buySignal'], df['sellSignal'].astype(float) * -1).fillna(df['pos'].shift(1))

    df['longCond'] = df['buySignal'] & (df['pos'].shift(1) != 1)
    df['shortCond'] = df['sellSignal'] & (df['pos'].shift(1) != -1)

    df['entryOfLongPosition'] = df['open'].where(df['longCond']).ffill()
    df['entryOfShortPosition'] = df['open'].where(df['shortCond']).ffill()

    return df

def calculate_stop_loss_and_take_profit(data, window=4):
    """Calculate stop loss and take profit levels based on a 1:3 risk-reward ratio."""
    data['Stop_Loss'] = np.nan
    data['Take_Profit'] = np.nan
    data['Risk_to_Reward_Ratio'] = np.nan

    for i in range(len(data)):
        if i < window:
            continue  # Skip calculations until we have enough candles

        recent_data = data.iloc[i-window:i]
        current_candle = data.iloc[i]

        if data['buySignal'].iloc[i]:
            # Long position calculations
            stop_loss = recent_data['low'].min()  # Lowest low of the recent candles
            entry_price = current_candle['close']  # Entry price is the close price of the current candle
            risk = entry_price - stop_loss  # Risk is the difference between entry and stop loss
            take_profit = entry_price + (3 * risk)  # Take profit is 3 times the risk

            # Store calculated values in the DataFrame
            data.at[data.index[i], 'Stop_Loss'] = stop_loss
            data.at[data.index[i], 'Take_Profit'] = take_profit
            data.at[data.index[i], 'Risk_to_Reward_Ratio'] = 1 / 3  # Ratio for 1:3 risk-reward

        elif data['sellSignal'].iloc[i]:
            # Short position calculations
            stop_loss = recent_data['high'].max()  # Highest high of the recent candles
            entry_price = current_candle['close']  # Entry price is the close price of the current candle
            risk = stop_loss - entry_price  # Risk is the difference between stop loss and entry
            take_profit = entry_price - (3 * risk)  # Take profit is 3 times the risk

            # Store calculated values in the DataFrame
            data.at[data.index[i], 'Stop_Loss'] = stop_loss
            data.at[data.index[i], 'Take_Profit'] = take_profit
            data.at[data.index[i], 'Risk_to_Reward_Ratio'] = 1 / 3  # Ratio for 1:3 risk-reward

    return data
 
def train_model(data):
    """Train a machine learning model to predict the close price."""
    features = ['open', 'high', 'low', 'volume', 'Andean_Oscillator']
    target = 'close'
   
    # Ensure Trend_Signal column is present
    if 'Trend_Signal' not in data.columns:
        data['Trend_Signal'] = 0  # Default value if column is missing
 
    # Drop rows with NaN values in specified columns
    data = data.dropna(subset=features + [target])
   
    X = data[features].dropna()
    y = data[target][X.index]
   
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
   
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
   
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    accuracy = accuracy_score(y_test.round(), y_pred.round())
   
    return model, mae, accuracy
 
def predict_close(model, data):
    """Use the trained model to predict the close price."""
    features = ['open', 'high', 'low', 'volume', 'Andean_Oscillator']
    data = data[features].dropna()
   
    if data.empty:
        return np.nan
   
    return model.predict(data).mean()
 
def calculate_final_predicted_close_accuracy(actual_close, predicted_close):
    """Calculate accuracy of the final predicted close price."""
    if actual_close == 0:
        return np.nan  # Avoid division by zero
    accuracy = abs((predicted_close - actual_close) / actual_close) * 100
    return accuracy
 
def calculate_signal_mae(actual_signals, predicted_signals):
    """Calculate MAE for the trading signals and return as a percentage."""
    if len(actual_signals) == 0:
        return np.nan  # Avoid division by zero
   
    # Convert signals to numerical values (e.g., Buy = 1, Sell = -1, None = 0)
    signal_map = {'Buy': 1, 'Sell': -1, 'None': 0}
    actual_values = [signal_map.get(sig, 0) for sig in actual_signals]
    predicted_values = [signal_map.get(sig, 0) for sig in predicted_signals]
 
    mae = mean_absolute_error(actual_values, predicted_values)
   
    # Calculate maximum possible MAE
    max_mae = len(actual_signals)  # As each signal could be wrong
   
    # Convert MAE to percentage
    mae_percentage = (mae / max_mae) * 100
   
    return mae_percentage
 
def fetch_data_with_retries(symbol, retries=1):
    """Fetch data with retries in case of connection issues."""
    tv = TvDatafeed()
    df = None
    for i in range(retries):
        try:
            df = tv.get_hist(symbol, "NSE", Interval.in_1_hour, n_bars=5000)
            if df is not None and not df.empty:
                break
        except Exception as e:
            print(f"Attempt {i+1}: Failed to fetch data for {symbol}. Error: {e}")
    return df

def calculate_signal_accuracy(df):
    """Calculate signal accuracy and ensure it does not exceed 100%."""
    total_signals = len(df)
    correct_signals = ((df['buySignal'] & df['buySignal'].shift(-1)) | (df['sellSignal'] & df['sellSignal'].shift(-1))).sum()

    # Calculate signal accuracy as a percentage
    signal_accuracy = (correct_signals / total_signals) * 100 if total_signals > 0 else 0

    # Ensure that the signal accuracy does not exceed 100%
    signal_accuracy = min(signal_accuracy, 100)

    return signal_accuracy
 
def print_trade_info(df, symbol):
    """Print trade information in the specified format and save to CSV."""
    latest_row = df.iloc[-1]
 
    # Extract necessary values
    date = latest_row.name.date()
    time = latest_row.name.time()
    symbol_price = latest_row['close']
    predicted_historical_close = latest_row['Historical_Predicted_Close']
    predicted_real_time_close = latest_row['Real_Time_Predicted_Close']
    final_predicted_close = latest_row['Final_Predicted_Close']
    final_predicted_close_accuracy = latest_row.get('Final_Predicted_Close_Accuracy', 'N/A')
    open_price = latest_row['open']
    high_price = latest_row['high']
    low_price = latest_row['low']
    close_price = latest_row['close']
    volume = latest_row['volume']
    andean_oscillator_value = latest_row['Andean_Oscillator']
    trend_signal = "Buy" if latest_row['buySignal'] else "Sell" if latest_row['sellSignal'] else "None"
    trend = "Up" if latest_row['buySignal'] else "Down" if latest_row['sellSignal'] else "Neutral"
    trading_signal = f"Buy" if latest_row['buySignal'] else f"Sell" if latest_row['sellSignal'] else "None"
    take_profit = latest_row.get('Take_Profit', 'N/A')
    stop_loss = latest_row.get('Stop_Loss', 'N/A')
    signal_mae = latest_row.get('Signal_MAE', 'N/A')
    signal_accuracy = latest_row.get('signal_accuracy', 'N/A')
   
    # Create the message
    message = (
        f"Date: {date}\n"
        f"Time: {time}\n"
        f"Symbol: {symbol}\n"
        f"Symbol Price: {symbol_price}\n"
        f"Final Predicted Close Price: {final_predicted_close}\n"
        f"Final Predicted Close Price Accuracy: {final_predicted_close_accuracy:.2f}%\n"
        f"Take Profit: {take_profit}\n"
        f"Stop Loss: {stop_loss}\n"
        f"Trading Signal: {trading_signal}\n"
        f"Signal Accuracy: Confidence Level {signal_accuracy}% {trading_signal}"
    )
    # Send message to Telegram
    send_message_sync(message)
   
    # Print information with neon colors
    print(f"Date: {date}")
    print(f"Time: {time}")
    print(f"Symbol: {symbol}")
    print(f"Symbol Price: {symbol_price}")
    print(f"Predicted Historical Close Price: {predicted_historical_close}")
    print(f"Predicted Real Time Close Price: {predicted_real_time_close}")
    print(f"Final Predicted Close Price: {final_predicted_close}")
    print(f"Final Predicted Close Price Accuracy: {final_predicted_close_accuracy:.2f}%")
    print(f"Open price: {open_price}")
    print(f"High price: {high_price}")
    print(f"Low price: {low_price}")
    print(f"Close price: {close_price}")
    print(f"Volume: {volume}")
    print(f"Andean Oscillator value: {andean_oscillator_value}")
    print(f"Trend Signal: {trading_signal}")
    print(f"Trend: {trend}")
    print(f"Take Profit: {take_profit}")
    print(f"Stop Loss: {stop_loss}")
    print(f"Trading Signal: {trading_signal}")
    print(f"Signal Accuracy : Confidence Level {signal_accuracy}% {trading_signal}")
 
    print("-" * 40)
 
def process_symbol(symbol, statistics):
    """Process each symbol and calculate buy/sell signals."""
    df = fetch_data_with_retries(symbol)
    if df is None or df.empty:
        print(f"Failed to fetch data for {symbol} after multiple attempts.")
        return

    # Calculate indicators
    df = calculate_trend_signal(df)
    df = calculate_andean_oscillator(df)
    df = calculate_stop_loss_and_take_profit(df)
   
    # Train model and make predictions
    model, _, _ = train_model(df)
    historical_predicted_close = predict_close(model, df)
   
    # Real-time prediction (latest available data)
    real_time_predicted_close = predict_close(model, df.iloc[-1:])
   
    # Calculate final predicted close price
    final_predicted_close = (historical_predicted_close + real_time_predicted_close) / 2
   
    # Calculate accuracy
    actual_close = df['close'].iloc[-1]
    final_predicted_close_accuracy = calculate_final_predicted_close_accuracy(actual_close, final_predicted_close)
 
    # Collect actual and predicted signals
    actual_signals = ['Buy' if x else 'Sell' if y else 'None' for x, y in zip(df['buySignal'], df['sellSignal'])]
    predicted_signals = ['Buy' if df['buySignal'].iloc[-1] else 'Sell' if df['sellSignal'].iloc[-1] else 'None'] * len(df)
 
    signal_mae = calculate_signal_mae(actual_signals, predicted_signals)

    signal_accuracy = calculate_signal_accuracy(df)
   
 
    # Add predicted close prices and accuracy to the DataFrame for printing
    df['Historical_Predicted_Close'] = historical_predicted_close
    df['Real_Time_Predicted_Close'] = real_time_predicted_close
    df['Final_Predicted_Close'] = final_predicted_close
    df['Final_Predicted_Close_Accuracy'] = final_predicted_close_accuracy
    df['Signal_MAE'] = signal_mae
    df['signal_accuracy'] = signal_accuracy
   
    # Print the latest trade information
    print_trade_info(df, symbol)
   
    # Save the DataFrame to a CSV file
    df.to_csv(f'{symbol}_processed_data.csv', index=True)
 
    # Calculate statistics
    total_trades = len(df)
    total_buy_trades = df['buySignal'].sum()
    total_sell_trades = df['sellSignal'].sum()

    # Assuming a trade is a win if the take profit was hit, and a loss if the stop loss was hit
    win_trades = df['Take_Profit'].notna().sum()
    loss_trades = df['Stop_Loss'].notna().sum()

    win_rate = (win_trades / total_buy_trades) * 100 if total_buy_trades > 0 else 0
    loss_rate = (loss_trades / total_sell_trades) * 100 if total_sell_trades > 0 else 0
 
    total_profit_loss = df['Take_Profit'].sum() - df['Stop_Loss'].sum()
    average_profit_per_trade = df[df['buySignal']]['Take_Profit'].mean() if total_buy_trades > 0 else 0
    average_loss_per_trade = df[df['sellSignal']]['Stop_Loss'].mean() if total_sell_trades > 0 else 0
    maximum_drawdown = df['Stop_Loss'].min() if not df['Stop_Loss'].empty else 0
 
    statistics.append({
        'Symbol': symbol,
        'Total Trades': total_trades,
        'Total Buy Trades': total_buy_trades,
        'Total Sell Trades': total_sell_trades,
        'Win Trade': win_trades,
        'Loss Trade': loss_trades,
        'Win Rate': win_rate,
        'Loss Rate': loss_rate,
        'Total Profit/Loss': total_profit_loss,
        'Average Profit Per Trade': average_profit_per_trade,
        'Average Loss Per Trade': average_loss_per_trade,
        'Maximum Drawdown': maximum_drawdown
    })
 
    return statistics

def trading_logic(symbols):
    statistics = []
    results_for_csv = []  # To collect results for saving to CSV

    # Define the trading start and stop times
    trading_start_time = datetime.strptime("09:15", "%H:%M").time()
    trading_end_time = datetime.strptime("16:00", "%H:%M").time()

    for symbol in symbols:
        current_time = datetime.now().time()

        # Check if the current time is within the trading window
        if trading_start_time <= current_time <= trading_end_time:
            start_time = time.time()
            process_symbol(symbol, statistics)
            end_time = time.time()
            elapsed_time = end_time - start_time
            print(f'Processed {symbol} in {elapsed_time:.2f} seconds')

            # Delay before processing the next symbol
            time.sleep(120)
        else:
            print("Trading window has closed. Stopping the program.")
            break

    if statistics:  # Check if statistics is not empty
        # Save statistics to CSV
        stats_df = pd.DataFrame(statistics)
        stats_df.to_csv('trading_statistics.csv', index=False)

        # Collect and save additional information to a separate CSV file
        if 'Total Profit/Loss' in stats_df.columns:
            profitable_symbols = stats_df[stats_df['Total Profit/Loss'] > 0].shape[0]
            loss_making_symbols = stats_df[stats_df['Total Profit/Loss'] < 0].shape[0]

            accuracy_percentage = (profitable_symbols / len(symbols)) * 100

            # Ensure that accuracy_percentage is not beyond 100%
            accuracy_percentage = min(accuracy_percentage, 100)

            # Add this information to the results list
            results_for_csv.append({
                'Accuracy (%)': accuracy_percentage,
                'Number of Profitable Symbols': profitable_symbols,
                'Number of Loss-Making Symbols': loss_making_symbols
            })

        # Save results to a separate CSV file
        results_df = pd.DataFrame(results_for_csv)
        results_df.to_csv('trading_results_summary.csv', index=False)
    else:
        print("No data collected. The statistics list is empty.")

statistics = []
trading_logic(indian_companies)