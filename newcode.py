# import logging
# import pandas as pd
# import numpy as np
# from tvDatafeed import TvDatafeed, Interval
# from indian_companies import indian_companies  # Import the list of Indian companies
# from time import sleep
# from datetime import datetime, timedelta
# import time

# logger = logging.getLogger(__name__)
# Multiplier = 2.5  # Sensitivity
# atrPeriods = 14  # ATR Length
# stop_loss_percent = 0.005  # 0.5% Stop Loss 

# # Helper functions for Trend Indicator
# def percent(nom, div):
#     return 100 * nom / div

# def f1(m, n):
#     return m if m >= n else 0.0

# def f2(m, n):
#     return 0.0 if m >= n else -m

# def calculate_andean_oscillator(data):
#     """Calculate Andean Oscillator."""
#     data['Momentum'] = data['close'] - data['close'].shift(50)
#     data['Andean_Oscillator'] = data['Momentum'].rolling(window=50).mean()
   
#     # Example of using different conditions
#     data['buySignal'] = data['Andean_Oscillator'] > 0
#     data['sellSignal'] = data['Andean_Oscillator'] < 0

#     data['Andean_Oscillator_Bullish'] = data['Andean_Oscillator'].clip(lower=0)
#     data['Andean_Oscillator_Bearish'] = data['Andean_Oscillator'].clip(upper=0).abs()
   
#     return data

# def calculate_trend_signal(df):
#     Multiplier = 2.5  # Sensitivity
#     atrPeriods = 14  # ATR Length
#     stop_loss_percent = 0.005  # 0.5% Stop Loss
 
#     df['hl2'] = (df['high'] + df['low']) / 2
#     df['atr'] = df['close'].rolling(window=atrPeriods).apply(lambda x: x.max() - x.min())
 
#     # Calculate up and down lines
#     df['up'] = df['hl2'] - (Multiplier * df['atr'])
#     df['dn'] = df['hl2'] + (Multiplier * df['atr'])
#     df['up'] = df['up'].where(df['close'].shift(1) > df['up'].shift(1), np.maximum(df['up'], df['up'].shift(1)))
#     df['dn'] = df['dn'].where(df['close'].shift(1) < df['dn'].shift(1), np.minimum(df['dn'], df['dn'].shift(1)))
 
#     # Determine trend
#     df['trend'] = 1  # Default to uptrend
#     df['trend'] = df['trend'].where((df['trend'].shift(1) != -1) | (df['close'] > df['dn'].shift(1)), 1)
#     df['trend'] = df['trend'].where((df['trend'].shift(1) != 1) | (df['close'] < df['up'].shift(1)), -1)
 
#     # Buy and sell signals
#     df['buySignal'] = (df['trend'] == 1) & (df['trend'].shift(1) == -1)
#     df['sellSignal'] = (df['trend'] == -1) & (df['trend'].shift(1) == 1)
 
#     return df
 
# def check_trend_signals(df):
#     df['Confirmed_Buy_Signal'] = (df['buySignal'] & (df['Andean_Oscillator_Bullish'] > df['Andean_Oscillator_Bearish']))
#     df['Confirmed_Sell_Signal'] = (df['sellSignal'] & (df['Andean_Oscillator_Bearish'] > df['Andean_Oscillator_Bullish']))
#     return df

# import logging
# import pandas as pd
# import numpy as np
# from tvDatafeed import TvDatafeed, Interval
# from indian_companies import indian_companies  # Import the list of Indian companies
# from time import sleep
# from datetime import datetime
# import time

# logger = logging.getLogger(__name__)

# # Helper functions for Trend Indicator
# def percent(nom, div):
#     return 100 * nom / div

# def f1(m, n):
#     return m if m >= n else 0.0

# def f2(m, n):
#     return 0.0 if m >= n else -m

# def calculate_andean_oscillator(data):
#     """Calculate Andean Oscillator."""
#     data['Momentum'] = data['close'] - data['close'].shift(50)
#     data['Andean_Oscillator'] = data['Momentum'].rolling(window=50).mean()
   
#     # Example of using different conditions
#     data['buySignal'] = data['Andean_Oscillator'] > 0
#     data['sellSignal'] = data['Andean_Oscillator'] < 0

#     data['Andean_Oscillator_Bullish'] = data['Andean_Oscillator'].clip(lower=0)
#     data['Andean_Oscillator_Bearish'] = data['Andean_Oscillator'].clip(upper=0).abs()
   
#     return data

# def calculate_trend_signal(df):
#     Multiplier = 2.5  # Sensitivity
#     atrPeriods = 14  # ATR Length
#     stop_loss_percent = 0.005  # 0.5% Stop Loss
 
#     df['hl2'] = (df['high'] + df['low']) / 2
#     df['atr'] = df['close'].rolling(window=atrPeriods).apply(lambda x: x.max() - x.min())
 
#     # Calculate up and down lines
#     df['up'] = df['hl2'] - (Multiplier * df['atr'])
#     df['dn'] = df['hl2'] + (Multiplier * df['atr'])
#     df['up'] = df['up'].where(df['close'].shift(1) > df['up'].shift(1), np.maximum(df['up'], df['up'].shift(1)))
#     df['dn'] = df['dn'].where(df['close'].shift(1) < df['dn'].shift(1), np.minimum(df['dn'], df['dn'].shift(1)))
 
#     # Determine trend
#     df['trend'] = 1  # Default to uptrend
#     df['trend'] = df['trend'].where((df['trend'].shift(1) != -1) | (df['close'] > df['dn'].shift(1)), 1)
#     df['trend'] = df['trend'].where((df['trend'].shift(1) != 1) | (df['close'] < df['up'].shift(1)), -1)
 
#     # Buy and sell signals
#     df['buySignal'] = (df['trend'] == 1) & (df['trend'].shift(1) == -1)
#     df['sellSignal'] = (df['trend'] == -1) & (df['trend'].shift(1) == 1)
 
#     return df


 
# def check_trend_signals(df):
#     df['Confirmed_Buy_Signal'] = (df['buySignal'] & (df['Andean_Oscillator_Bullish'] > df['Andean_Oscillator_Bearish']))
#     df['Confirmed_Sell_Signal'] = (df['sellSignal'] & (df['Andean_Oscillator_Bearish'] > df['Andean_Oscillator_Bullish']))
#     return df

# def set_stop_loss_take_profit(df):
#     """Set stop loss and take profit values based on entry and volatility."""
#     stop_loss_percent = 0.005  # 0.5% stop loss as defined in the Pine Script

#     # Ensure we're calculating only for confirmed buy/sell signals
#     for i in range(len(df)):
#         if df.iloc[i]['Confirmed_Buy_Signal']:
#             entry_price = df.iloc[i]['close']
#             df.at[i, 'entryOfLongPosition'] = entry_price
#             df.at[i, 'stopLoss'] = entry_price * (1 - stop_loss_percent)
#             df.at[i, 'takeProfit1'] = entry_price * (1 + stop_loss_percent)
#             df.at[i, 'takeProfit2'] = entry_price * (1 + 2 * stop_loss_percent)
#             df.at[i, 'takeProfit3'] = entry_price * (1 + 3 * stop_loss_percent)

#         elif df.iloc[i]['Confirmed_Sell_Signal']:
#             entry_price = df.iloc[i]['close']
#             df.at[i, 'entryOfShortPosition'] = entry_price
#             df.at[i, 'stopLoss'] = entry_price * (1 + stop_loss_percent)
#             df.at[i, 'takeProfit1'] = entry_price * (1 - stop_loss_percent)
#             df.at[i, 'takeProfit2'] = entry_price * (1 - 2 * stop_loss_percent)
#             df.at[i, 'takeProfit3'] = entry_price * (1 - 3 * stop_loss_percent)

# def print_trade_info(head_df, tail_df, symbol):
#     # Extract the latest data from df.tail() for trend signals, entry price, stop loss, and take profit
#     latest_row = tail_df.iloc[-1]
    
#     # Extract datetime, open, high, low, close, and volume from df.head()
#     print("------------------------------")
#     print(f"📅 datetime: {head_df.index[-1]}")  # Accessing the latest datetime from head_df
#     print(f"🔢 symbol: {symbol}")
#     print(f"💵 open: {head_df['open'].iloc[-1]}")
#     print(f"📈 high: {head_df['high'].iloc[-1]}")
#     print(f"📉 low: {head_df['low'].iloc[-1]}")
#     print(f"❌ close: {head_df['close'].iloc[-1]}")
#     print(f"📊 volume: {head_df['volume'].iloc[-1]}")
    
#     trend_signal = "None"
#     entry_price = stop_loss = tp1 = tp2 = tp3 = 'N/A'
 
#     # Check for a confirmed buy signal
#     if latest_row.get('Confirmed_Buy_Signal', False):
#         trend_signal = "Confirmed Buy"
#         entry_price = latest_row.get('entryOfLongPosition', 'N/A')
#         stop_loss = latest_row.get('stopLoss', 'N/A')
#         tp1 = latest_row.get('takeProfit1', 'N/A')
#         tp2 = latest_row.get('takeProfit2', 'N/A')
#         tp3 = latest_row.get('takeProfit3', 'N/A')

#     # Check for a confirmed sell signal
#     elif latest_row.get('Confirmed_Sell_Signal', False):
#         trend_signal = "Confirmed Sell"
#         entry_price = latest_row.get('entryOfShortPosition', 'N/A')
#         stop_loss = latest_row.get('stopLoss', 'N/A')
#         tp1 = latest_row.get('takeProfit1', 'N/A')
#         tp2 = latest_row.get('takeProfit2', 'N/A')
#         tp3 = latest_row.get('takeProfit3', 'N/A')
    
#     # Print trend signal and trading information
#     print(f"Trend Signal: {trend_signal}")
#     print(f"Entry Price: {entry_price}, Stop Loss: {stop_loss}, TP1: {tp1}, TP2: {tp2}, TP3: {tp3}")
#     print("------------------------------")

# def main():
#     tv = TvDatafeed()

#     while True:
#         for symbol in indian_companies:
#             try:
#                 # Fetch historical data
#                 df = tv.get_hist(symbol=symbol, exchange='NSE', interval=Interval.in_15_minute, n_bars=500)
                
#                 if df is None or df.empty:
#                     logger.error(f"No data returned for symbol {symbol}")
#                     continue
                
#                 df['symbol'] = symbol
                
#                 # Reset the index to ensure it is properly aligned
#                 df.reset_index(inplace=True)
#                 df.set_index('datetime', inplace=True)  # Ensure 'datetime' column is set as index
                
#                 # Apply calculations
#                 df = calculate_trend_signal(df)
#                 df = calculate_andean_oscillator(df)
#                 df = check_trend_signals(df)
#                 set_stop_loss_take_profit(df)
                
                
#                 # Extract data for print_trade_info
#                 head_df = df.head()
#                 tail_df = df.tail()
                
#                 # Print trade information
#                 print_trade_info(head_df, tail_df, symbol)
            
#             except Exception as e:
#                 logger.error(f"Error processing symbol {symbol}: {e}")
        
#             # Wait for 10 seconds before processing the next symbol
#             time.sleep(10)

# if __name__ == "__main__":
#     main()









import logging
import pandas as pd
import numpy as np
from tvDatafeed import TvDatafeed, Interval
from indian_companies import indian_companies  # Import the list of Indian companies
from time import sleep
from datetime import datetime, timedelta
import time

logger = logging.getLogger(__name__)
Multiplier = 2.5  # Sensitivity
atrPeriods = 14  # ATR Length
stop_loss_percent = 0.005  # 0.5% Stop Loss 

# Helper functions for Trend Indicator
def percent(nom, div):
    return 100 * nom / div

def f1(m, n):
    return m if m >= n else 0.0

def f2(m, n):
    return 0.0 if m >= n else -m

def calculate_andean_oscillator(df, length=14):
    df['highMA'] = df['high'].rolling(window=length).mean()
    df['lowMA'] = df['low'].rolling(window=length).mean()
    
    df['oscillator'] = df['highMA'] - df['lowMA']
    df['oscillatorSignal'] = df['oscillator'].rolling(window=length).mean()
    
    df['Andean_Oscillator'] = df['oscillator']
    df['Andean_Oscillator_Bullish'] = df['oscillator'] > df['oscillatorSignal']
    df['Andean_Oscillator_Bearish'] = df['oscillator'] < df['oscillatorSignal']
    
    return df

def calculate_trend_signal(df, short_period=9, long_period=21, multiplier=2.5, atr_periods=14, stop_loss_percent=0.005):
    df['shortMA'] = df['close'].rolling(window=short_period).mean()
    df['longMA'] = df['close'].rolling(window=long_period).mean()
    
    df['buySignal'] = (df['shortMA'] > df['longMA']) & (df['shortMA'].shift(1) <= df['longMA'].shift(1))
    df['sellSignal'] = (df['shortMA'] < df['longMA']) & (df['shortMA'].shift(1) >= df['longMA'].shift(1))
    
    df['Confirmed_Buy_Signal'] = df['buySignal'].shift(-1)
    df['Confirmed_Sell_Signal'] = df['sellSignal'].shift(-1)
    

    df['hl2'] = (df['high'] + df['low']) / 2
    df['atr'] = df['close'].rolling(window=atr_periods).apply(lambda x: x.max() - x.min(), raw=False)
    
    # Calculate up and down lines
    df['up'] = df['hl2'] - (multiplier * df['atr'])
    df['dn'] = df['hl2'] + (multiplier * df['atr'])
    df['up'] = df['up'].where(df['close'].shift(1) > df['up'].shift(1), np.maximum(df['up'], df['up'].shift(1)))
    df['dn'] = df['dn'].where(df['close'].shift(1) < df['dn'].shift(1), np.minimum(df['dn'], df['dn'].shift(1)))
    
    # Determine trend
    df['trend'] = 1  # Default to uptrend
    df['trend'] = df['trend'].where((df['trend'].shift(1) != -1) | (df['close'] > df['dn'].shift(1)), 1)
    df['trend'] = df['trend'].where((df['trend'].shift(1) != 1) | (df['close'] < df['up'].shift(1)), -1)
    
    # Buy and sell signals
    df['buySignal'] = (df['trend'] == 1) & (df['trend'].shift(1) == -1)
    df['sellSignal'] = (df['trend'] == -1) & (df['trend'].shift(1) == 1)
    
    return df

 
def check_trend_signals(df):
    df['Confirmed_Buy_Signal'] = (df['buySignal'] & (df['Andean_Oscillator_Bullish'] > df['Andean_Oscillator_Bearish']))
    df['Confirmed_Sell_Signal'] = (df['sellSignal'] & (df['Andean_Oscillator_Bearish'] > df['Andean_Oscillator_Bullish']))
    return df

def set_stop_loss_take_profit(df, stop_loss_percent = 0.005):
    """Set stop loss and take profit values based on entry and volatility."""
    stop_loss_percent = 0.005  # 0.5% stop loss as defined in the Pine Script

    # Ensure we're calculating only for confirmed buy/sell signals
    for i in range(len(df)):
        if df.iloc[i]['Confirmed_Buy_Signal']:
            entry_price = df.iloc[i]['close']
            df.at[i, 'entryOfLongPosition'] = entry_price
            df.at[i, 'stopLoss'] = entry_price * (1 - stop_loss_percent)
            df.at[i, 'takeProfit1'] = entry_price * (1 + stop_loss_percent)
            df.at[i, 'takeProfit2'] = entry_price * (1 + 2 * stop_loss_percent)
            df.at[i, 'takeProfit3'] = entry_price * (1 + 3 * stop_loss_percent)

        elif df.iloc[i]['Confirmed_Sell_Signal']:
            entry_price = df.iloc[i]['close']
            df.at[i, 'entryOfShortPosition'] = entry_price
            df.at[i, 'stopLoss'] = entry_price * (1 + stop_loss_percent)
            df.at[i, 'takeProfit1'] = entry_price * (1 - stop_loss_percent)
            df.at[i, 'takeProfit2'] = entry_price * (1 - 2 * stop_loss_percent)
            df.at[i, 'takeProfit3'] = entry_price * (1 - 3 * stop_loss_percent)

def print_trade_info(head_df, tail_df, symbol):
    # Extract the latest data from df.tail() for trend signals, entry price, stop loss, and take profit
    latest_row = tail_df.iloc[-1]
    
    # Extract datetime, open, high, low, close, and volume from df.head()
    print("------------------------------")
    print(f"📅 datetime: {head_df.index[-1]}")  # Accessing the latest datetime from head_df
    print(f"🔢 symbol: {symbol}")
    print(f"💵 open: {head_df['open'].iloc[-1]}")
    print(f"📈 high: {head_df['high'].iloc[-1]}")
    print(f"📉 low: {head_df['low'].iloc[-1]}")
    print(f"❌ close: {head_df['close'].iloc[-1]}")
    print(f"📊 volume: {head_df['volume'].iloc[-1]}")
    
    trend_signal = "None"
    entry_price = stop_loss = tp1 = tp2 = tp3 = 'N/A'
 
    # Check for a confirmed buy signal
    if latest_row.get('Confirmed_Buy_Signal', False):
        trend_signal = "Confirmed Buy"
        entry_price = latest_row.get('entryOfLongPosition', 'N/A')
        stop_loss = latest_row.get('stopLoss', 'N/A')
        tp1 = latest_row.get('takeProfit1', 'N/A')
        tp2 = latest_row.get('takeProfit2', 'N/A')
        tp3 = latest_row.get('takeProfit3', 'N/A')

    # Check for a confirmed sell signal
    elif latest_row.get('Confirmed_Sell_Signal', False):
        trend_signal = "Confirmed Sell"
        entry_price = latest_row.get('entryOfShortPosition', 'N/A')
        stop_loss = latest_row.get('stopLoss', 'N/A')
        tp1 = latest_row.get('takeProfit1', 'N/A')
        tp2 = latest_row.get('takeProfit2', 'N/A')
        tp3 = latest_row.get('takeProfit3', 'N/A')
    
    # Print trend signal and trading information
    print(f"Trend Signal: {trend_signal}")
    print(f"Entry Price: {entry_price}, Stop Loss: {stop_loss}, TP1: {tp1}, TP2: {tp2}, TP3: {tp3}")
    print("------------------------------")

def main():
    tv = TvDatafeed()

    while True:
        for symbol in indian_companies:
            try:
                # Fetch historical data
                df = tv.get_hist(symbol=symbol, exchange='NSE', interval=Interval.in_15_minute, n_bars=500)
                
                if df is None or df.empty:
                    logger.error(f"No data returned for symbol {symbol}")
                    continue
                
                df['symbol'] = symbol
                
                # Reset the index to ensure it is properly aligned
                df.reset_index(inplace=True)
                df.set_index('datetime', inplace=True)  # Ensure 'datetime' column is set as index
                
                # Apply calculations
                df = calculate_trend_signal(df)
                df = calculate_andean_oscillator(df)
                df = check_trend_signals(df)
                set_stop_loss_take_profit(df)
                
                
                # Extract data for print_trade_info
                head_df = df.head()
                tail_df = df.tail()
                
                # Print trade information
                print_trade_info(head_df, tail_df, symbol)
            
            except Exception as e:
                logger.error(f"Error processing symbol {symbol}: {e}")
        
            # Wait for 10 seconds before processing the next symbol
            time.sleep(10)

if __name__ == "__main__":
    main()







# import logging
# import pandas as pd
# from tvDatafeed import TvDatafeed, Interval
# from indian_companies import indian_companies
# from time import sleep
# from datetime import datetime
# import numpy as np
# from telegram import Bot
# import asyncio
# import time
 
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)
 
# TOKEN = '7547264968:AAHkkVIah6x283RR-QRIFb7Xj7Gu3KinOxU'
# CHAT_ID = '@torioxindianstock'
# bot = Bot(token=TOKEN)
 
# async def send_message(message):
#     await bot.send_message(chat_id=CHAT_ID, text=message)
 
# def send_message_sync(message):
#     bot.send_message(chat_id=CHAT_ID, text=message)
#     try:
#         loop = asyncio.get_event_loop()
#     except RuntimeError as e:
#         if 'There is no current event loop in thread' in str(e):
#             loop = asyncio.new_event_loop()
#             asyncio.set_event_loop(loop)
#         else:
#             raise
#     if loop.is_closed():
#         loop = asyncio.new_event_loop()
#         asyncio.set_event_loop(loop)
#     loop.run_until_complete(send_message(message))
 
# def calculate_andean_oscillator(data):
#     # Adjusted for Length and Signal Length
#     data['Momentum'] = data['close'] - data['close'].shift(50)
#     data['Andean_Oscillator'] = data['Momentum'].rolling(window=50).mean()
#     data['Andean_Oscillator_Bullish'] = data['Andean_Oscillator'].clip(lower=0)
#     data['Andean_Oscillator_Bearish'] = data['Andean_Oscillator'].clip(upper=0).abs()
#     return data



# def calculate_trend_signal(df,multiplier=2.5, atr_periods=14, stop_loss_percent=0.005):
#     Multiplier = 2.5  # Sensitivity
#     atrPeriods = 14  # ATR Length
#     stop_loss_percent = 0.005  # 0.5% Stop Loss
 
#     df['hl2'] = (df['high'] + df['low']) / 2
#     df['atr'] = df['close'].rolling(window=atrPeriods).apply(lambda x: x.max() - x.min())
 
#     # Calculate up and down lines
#     df['up'] = df['hl2'] - (Multiplier * df['atr'])
#     df['dn'] = df['hl2'] + (Multiplier * df['atr'])
#     df['up'] = df['up'].where(df['close'].shift(1) > df['up'].shift(1), np.maximum(df['up'], df['up'].shift(1)))
#     df['dn'] = df['dn'].where(df['close'].shift(1) < df['dn'].shift(1), np.minimum(df['dn'], df['dn'].shift(1)))
 
#     # Determine trend
#     df['trend'] = 1  # Default to uptrend
#     df['trend'] = df['trend'].where((df['trend'].shift(1) != -1) | (df['close'] > df['dn'].shift(1)), 1)
#     df['trend'] = df['trend'].where((df['trend'].shift(1) != 1) | (df['close'] < df['up'].shift(1)), -1)
 
#     # Buy and sell signals
#     df['buySignal'] = (df['trend'] == 1) & (df['trend'].shift(1) == -1)
#     df['sellSignal'] = (df['trend'] == -1) & (df['trend'].shift(1) == 1)
 
#     return df
 
# def check_trend_signals(df):
#     df['Confirmed_Buy_Signal'] = (df['buySignal'] & (df['Andean_Oscillator_Bullish'] > df['Andean_Oscillator_Bearish']))
#     df['Confirmed_Sell_Signal'] = (df['sellSignal'] & (df['Andean_Oscillator_Bearish'] > df['Andean_Oscillator_Bullish']))
#     return df
 
# def fetch_data_with_retries(symbol, retries=3):
#     """Fetch data with retries in case of connection issues."""
#     tv = TvDatafeed()  # Initialize the datafeed
#     df = None
#     for attempt in range(retries):
#         try:
#             df = tv.get_hist(symbol, "NSE", Interval.in_1_hour, n_bars=5000)  # Fetch historical data
#             if df is not None and not df.empty:
#                 return df
#         except Exception as e:
#             print(f"Attempt {attempt + 1}: Failed to fetch data for {symbol}. Error: {e}")
#             sleep(5)  # Wait for 5 seconds before retrying
#     print(f"Failed to fetch data for {symbol} after {retries} attempts.")
#     return None
 
# def set_stop_loss_take_profit(df):
#     """Set stop loss and take profit values based on entry and volatility."""
#     stop_loss_percent = 0.005  # 0.5% stop loss as defined in the Pine Script
#     for i in range(len(df)):
#         if df.iloc[i]['Confirmed_Buy_Signal']:
#             entry_price = df.iloc[i]['close']
#             df.at[i, 'entryOfLongPosition'] = entry_price
#             df.at[i, 'stopLoss'] = entry_price * (1 - stop_loss_percent)
#             df.at[i, 'takeProfit1'] = entry_price * (1 + stop_loss_percent)
#             df.at[i, 'takeProfit2'] = entry_price * (1 + 2 * stop_loss_percent)
#             df.at[i, 'takeProfit3'] = entry_price * (1 + 3 * stop_loss_percent)
#         elif df.iloc[i]['Confirmed_Sell_Signal']:
#             entry_price = df.iloc[i]['close']
#             df.at[i, 'entryOfShortPosition'] = entry_price
#             df.at[i, 'stopLoss'] = entry_price * (1 + stop_loss_percent)
#             df.at[i, 'takeProfit1'] = entry_price * (1 - stop_loss_percent)
#             df.at[i, 'takeProfit2'] = entry_price * (1 - 2 * stop_loss_percent)
#             df.at[i, 'takeProfit3'] = entry_price * (1 - 3 * stop_loss_percent)
 
# def print_trade_info(df, symbol):
#     latest_row = df.iloc[-1]
#     trend_signal = "None"
#     entry_price = stop_loss = tp1 = tp2 = tp3 = 'N/A'
 
#     if latest_row['Confirmed_Buy_Signal']:
#         trend_signal = "Confirmed Buy"
#         entry_price = latest_row['entryOfLongPosition']
#         stop_loss = latest_row['stopLoss']
#         tp1 = latest_row['takeProfit1']
#         tp2 = latest_row['takeProfit2']
#         tp3 = latest_row['takeProfit3']
#     elif latest_row['Confirmed_Sell_Signal']:
#         trend_signal = "Confirmed Sell"
#         entry_price = latest_row['entryOfShortPosition']
#         stop_loss = latest_row['stopLoss']
#         tp1 = latest_row['takeProfit1']
#         tp2 = latest_row['takeProfit2']
#         tp3 = latest_row['takeProfit3']
 
#     message = (
#         f"Symbol: {symbol}\n"
#         f"Trend Signal: {trend_signal}\n"
#         f"Entry Price: {entry_price}\n"
#         f"Stop Loss: {stop_loss}\n"
#         f"Take Profit 1: {tp1}\n"
#         f"Take Profit 2: {tp2}\n"
#         f"Take Profit 3: {tp3}\n"
#     )
 
#     trading_start_time = datetime.strptime("09:15", "%H:%M").time()
#     trading_end_time = datetime.strptime("16:00", "%H:%M").time()
#     current_time = datetime.now().time()
 
#     if trading_start_time <= current_time <= trading_end_time:
#         send_message_sync(message)
#         print("Message sent:", message)
#     else:
#         print("Outside trading window, message not sent.")
 
#     print(message)
 
# def process_symbol(symbol, statistics):
#     df = fetch_data_with_retries(symbol)
#     if df is None or df.empty:
#         print(f"Failed to fetch data for {symbol} after multiple attempts.")
#         return
   
#     df = calculate_trend_signal(df)
#     df = calculate_andean_oscillator(df)
#     df = check_trend_signals(df)
   
#     # Debug: print the last few rows of 'Confirmed_Buy_Signal' and 'Confirmed_Sell_Signal'
#     print(df[['Confirmed_Buy_Signal', 'Confirmed_Sell_Signal']].tail())  # Check the last few rows
   
#     set_stop_loss_take_profit(df)  # Set stop loss and take profit levels
#     print_trade_info(df, symbol)
   
#     df.to_csv(f'{symbol}_processed_data.csv', index=True)
 
#     total_trades = len(df)
#     total_buy_trades = df['Confirmed_Buy_Signal'].sum()
#     total_sell_trades = df['Confirmed_Sell_Signal'].sum()
#     statistics.append({
#         'Symbol': symbol,
#         'Total Trades': total_trades,
#         'Total Buy Trades': total_buy_trades,
#         'Total Sell Trades': total_sell_trades
#     })
 
# def trading_logic(symbols):
#     statistics = []
#     while True:
#         for symbol in symbols:
#             start_time = time.time()
#             process_symbol(symbol, statistics)
#             end_time = time.time()
#             elapsed_time = end_time - start_time
#             print(f'Processed {symbol} in {elapsed_time:.2f} seconds')
#             time.sleep(30)
 
#         if statistics:
#             stats_df = pd.DataFrame(statistics)
#             stats_df.to_csv('trading_statistics.csv', index=False)
#         else:
#             print("No data collected. The statistics list is empty.")
 
#         sleep(300)
 
# statistics = []
# trading_logic(indian_companies)