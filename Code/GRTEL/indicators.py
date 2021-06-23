import numpy as np
import pandas as pd
import copy




# Calculate n-day RSI
def get_RSI(data, n=14):
    # First make a copy of the data frame twice
    up_df, down_df = data['change_in_price'].copy(), data['change_in_price'].copy()
    
    # For up days, if the change is less than 0 set to 0.
    up_df[up_df < 0] = 0
    # For down days, if the change is greater than 0 set to 0.
    down_df[down_df > 0] = 0
    # We need change_in_price to be absolute.
    down_df = down_df.abs()
    
    # Calculate the EWMA (Exponential Weighted Moving Average)
    ewma_up = up_df.ewm(span=n).mean()
    ewma_down = down_df.ewm(span=n).mean()
    
    # Calculate the Relative Strength
    relative_strength = ewma_up / ewma_down

    # Calculate the Relative Strength Index
    relative_strength_index = 100.0 - (100.0 / (1.0 + relative_strength))
    
    return relative_strength_index



# Calculate the n-day Stochastic Oscillator
def get_Stochastic_Oscillator(data, n=14):
    # Make a copy of the high and low column.
    low, high = data['Low'].copy(), data['High'].copy()
    
    # Calculate rolling min and max.
    low = low.rolling(window=n).min()
    high = high.rolling(window=n).max()
    
    # Calculate the Stochastic Oscillator.
    k_percent = 100 * ((data['Close'] - low) / (high - low))
    
    return k_percent



# Calculate the Williams %R
def get_Williams(data, n=14):
    # Make a copy of the high and low column.
    low, high = data['Low'].copy(), data['High'].copy()
    
    # Calculate rolling min and max.
    low = low.rolling(window=n).min()
    high = high.rolling(window=n).max()
    
    # Calculate William %R indicator.
    r_percent = ((high - data['Close']) / (high - low)) * -100
    
    return r_percent



# Calculate the MACD
def get_MACD(data):
    ema_26 = data['Close'].ewm(span=26).mean()
    ema_12 = data['Close'].ewm(span=12).mean()

    macd = ema_12 - ema_26

    # Calculate the EMA of MACD
    ema_9_macd = macd.ewm(span=9).mean()
    
    return macd, ema_9_macd
    

    
# Calculate On Balance Volume
def get_OBV(data):
    volumes = data['Volume']
    changes = data['change_in_price']

    prev_obv = 0
    obv_values = []
    for change, volume in zip(changes, volumes):
        if change > 0:
            current_obv = prev_obv + volume
        elif change < 0:
            current_obv = prev_obv - volume
        else:
            current_obv = prev_obv

        obv_values.append(current_obv)
        prev_obv = current_obv

    return pd.Series(obv_values, index=data.index)    