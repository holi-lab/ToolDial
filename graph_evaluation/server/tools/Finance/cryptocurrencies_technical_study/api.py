import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def triple_exponentially_smoothed_average_trix(timeframe: str, period: int, symbol: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The Triple Exponentially Smoothed Average (TRIX) is a technical analysis indicator that is used to identify trends in the market. It is a variation of the Exponential Moving Average (EMA), which assigns more weight to recent price data than older data.
		
		The TRIX is calculated by taking a triple exponential moving average of the price data, and then calculating the rate of change of the TEMA. The result is a momentum indicator that can be used to identify potential trend reversals and entry and exit points for trades.
		
		The TRIX is designed to reduce lag and provide a more responsive signal to changes in the market. It can be used to identify bullish and bearish divergences, as well as to confirm other technical analysis indicators.
		
		Traders can use the TRIX in conjunction with other technical analysis tools to gain a more complete picture of the market and to make more informed trading decisions. Overall, the TRIX is a useful tool for traders looking to identify trends in the market and to stay ahead of changes in price action."
    
    """
    url = f"https://cryptocurrencies-technical-study.p.rapidapi.com/crypto/trix/{symbol}/{timeframe}/{period}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cryptocurrencies-technical-study.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def accumulation_distribution_line_adl(timeframe: str, symbol: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Accumulation Distribution Line (ADL) is a technical analysis indicator used to identify changes in the supply and demand of a particular asset. It is also known as the Accumulation Distribution Index (ADI).
		
		The ADL is calculated by taking the cumulative total of the volume of an asset traded over a specific period of time, with an adjustment for the price movement of the asset during that same period. The formula is as follows:
		
		ADL = Previous ADL + ((Close - Low) - (High - Close)) / (High - Low) * Volume
		
		The ADL is plotted as a line on a chart, with the slope and direction of the line indicating changes in the buying and selling pressure of the asset.
		
		When the ADL line is rising, it indicates that the buying pressure for the asset is increasing. Conversely, when the ADL line is falling, it suggests that the selling pressure for the asset is increasing. Traders can use the ADL in conjunction with other technical analysis indicators to confirm potential trend reversals or breakouts.
		
		Overall, the Accumulation Distribution Line can be a useful tool for traders and investors looking to identify changes in supply and demand for a particular asset. However, it should be used in conjunction with other analysis tools and market information to make more informed trading decisions."
    
    """
    url = f"https://cryptocurrencies-technical-study.p.rapidapi.com/crypto/adl/{symbol}/{timeframe}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cryptocurrencies-technical-study.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def commodity_channel_index_cci(symbol: str, timeframe: str, period: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The Commodity Channel Index (CCI) is a technical analysis indicator that is used to measure the deviation of an asset's price from its statistical average. It was developed by Donald Lambert in the late 1970s and is commonly used by traders to identify potential overbought or oversold conditions.
		
		The CCI is calculated by taking the difference between the asset's typical price and its simple moving average over a specified period of time, and then dividing this value by a constant multiple of the mean absolute deviation. The resulting value is then plotted on a scale of -100 to +100.
		
		A reading above +100 indicates that the asset is overbought, while a reading below -100 indicates that it is oversold. Traders can use the CCI to identify potential trend reversals and to confirm other technical analysis indicators.
		
		For example, if the CCI is trending higher while the price of the security is trending lower, it may indicate that there is strong buying pressure in the market, and that a potential reversal in the price trend may be imminent.
		
		Overall, the CCI is a useful tool for traders looking to stay ahead of changes in market sentiment and to identify potential trading opportunities. However, like all technical analysis indicators, it should be used in conjunction with other analysis tools and market information to make more informed trading decisions."
    
    """
    url = f"https://cryptocurrencies-technical-study.p.rapidapi.com/crypto/cci/{symbol}/{timeframe}/{period}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cryptocurrencies-technical-study.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def bollinger_bands_bb(timeframe: str, stddev: int, period: int, symbol: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Bollinger Bands (BB) is a popular technical analysis tool used by traders to identify potential price trends and to measure volatility. It consists of a simple moving average (SMA) with upper and lower bands that are two standard deviations away from the SMA.
		
		The upper and lower bands are calculated by adding and subtracting two standard deviations from the SMA, respectively. The resulting bands form a channel around the price, which can help traders identify potential overbought or oversold conditions.
		
		When the price moves close to the upper band, it may indicate that the asset is overbought, while a move close to the lower band may indicate that it is oversold. Traders can use the BB to identify potential trend reversals and to confirm other technical analysis indicators.
		
		For example, if the price of the asset is moving higher while the BB is widening, it may indicate that there is strong buying pressure in the market and that the price trend may continue. Conversely, if the price of the asset is moving lower while the BB is narrowing, it may indicate that the market is becoming less volatile and that a potential price trend reversal may occur.
		
		Overall, Bollinger Bands is a useful tool for traders looking to stay ahead of changes in market sentiment and to identify potential trading opportunities. However, like all technical analysis indicators, it should be used in conjunction with other analysis tools and market information to make more informed trading decisions."
    
    """
    url = f"https://cryptocurrencies-technical-study.p.rapidapi.com/crypto/bb/{symbol}/{timeframe}/{period}/{stddev}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cryptocurrencies-technical-study.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def ichimoku_cloud(symbol: str, timeframe: str, displacement: int, spanperiod: int, conversionperiod: int, baseperiod: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The Ichimoku Cloud, also known as the Ichimoku Kinko Hyo, is a technical analysis tool developed by Japanese journalist Goichi Hosoda in the late 1930s. It consists of a set of indicators that work together to provide a comprehensive view of price action and market trends.
		
		The Ichimoku Cloud comprises five lines, including the Tenkan-sen (Conversion Line), Kijun-sen (Base Line), Senkou Span A (Leading Span A), Senkou Span B (Leading Span B), and Chikou Span (Lagging Span). These lines are calculated based on different time periods and are used to identify support and resistance levels, as well as potential trend reversals and entry/exit points.
		
		The cloud itself is formed by the area between Senkou Span A and Senkou Span B, and is often used to identify the overall trend of a market. When the price is above the cloud, it is considered to be in an uptrend, while a price below the cloud indicates a downtrend.
		
		Overall, the Ichimoku Cloud is a popular tool among technical analysts and traders due to its ability to provide a comprehensive view of market trends and price action, as well as its ease of use and versatility across different markets and timeframes."
    
    """
    url = f"https://cryptocurrencies-technical-study.p.rapidapi.com/crypto/ichimoku/{symbol}/{timeframe}/{conversionperiod}/{baseperiod}/{spanperiod}/{displacement}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cryptocurrencies-technical-study.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def wilders_smoothing_wema(timeframe: str, symbol: str, period: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Wilders Smoothing (WEMA) is a technical analysis indicator that is used to identify trends in the market. It is a variation of the Exponential Moving Average (EMA), which assigns more weight to recent price data than older data.
		
		The WEMA is calculated by taking the average of the price data over a specified period of time, and then smoothing it using a factor that is determined by the length of the period. The result is a trend-following indicator that can be used to identify potential entry and exit points for trades.
		
		The WEMA is designed to reduce lag and provide a more responsive signal to changes in the market. It can be used to identify bullish and bearish trends, as well as to confirm other technical analysis indicators.
		
		Traders can use the WEMA in conjunction with other technical analysis tools to gain a more complete picture of the market and to make more informed trading decisions. Overall, the WEMA is a useful tool for traders looking to identify trends in the market and to stay ahead of changes in price action."
    
    """
    url = f"https://cryptocurrencies-technical-study.p.rapidapi.com/crypto/wema/{symbol}/{timeframe}/{period}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cryptocurrencies-technical-study.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def weighted_moving_average_wma(timeframe: str, symbol: str, period: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The Weighted Moving Average (WMA) is a technical analysis indicator that is used to identify trends in the market. It is a variation of the Moving Average (MA), which assigns more weight to recent price data than older data.
		
		The WMA is calculated by taking the average of the price data over a specified period of time, with each data point weighted according to its position in the time period. This means that the more recent data points have a higher weight than older data points.
		
		The WMA is designed to provide a more responsive signal to changes in the market than other types of moving averages. It can be used to identify bullish and bearish trends, as well as to confirm other technical analysis indicators.
		
		Traders can use the WMA in conjunction with other technical analysis tools to gain a more complete picture of the market and to make more informed trading decisions. Overall, the WMA is a useful tool for traders looking to identify trends in the market and to stay ahead of changes in price action."
    
    """
    url = f"https://cryptocurrencies-technical-study.p.rapidapi.com/crypto/wma/{symbol}/{timeframe}/{period}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cryptocurrencies-technical-study.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def moving_average_convergence_divergence_macd(symbol: str, signalperiod: int, fastperiod: int, timeframe: str, slowperiod: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Moving Average Convergence Divergence (MACD) is a technical analysis indicator that is used to identify changes in momentum and trend direction. Developed by Gerald Appel in the late 1970s, MACD is a widely used indicator in technical analysis.
		
		MACD is calculated by subtracting the 26-period Exponential Moving Average (EMA) from the 12-period EMA, and then plotting a 9-period EMA of the MACD line as a signal line. The MACD line and the signal line are plotted together on a chart, with the MACD line being the faster of the two and the signal line being the slower.
		
		The MACD line and the signal line can both be used to identify buy and sell signals. When the MACD line crosses above the signal line, it is considered a buy signal, while a cross below the signal line is considered a sell signal. Traders can also look for divergence between the MACD line and the price of the asset being analyzed, which can indicate a potential trend reversal.
		
		MACD can also be used to identify changes in momentum. When the MACD line moves farther away from the signal line, it suggests that the momentum of the trend is increasing. Conversely, when the MACD line moves closer to the signal line, it suggests that the momentum of the trend is decreasing.
		
		Overall, the Moving Average Convergence Divergence is a versatile indicator that can be used in a variety of ways to identify changes in trend direction and momentum. However, like all technical analysis indicators, it should be used in conjunction with other analysis tools and market information to make more informed trading decisions."
    
    """
    url = f"https://cryptocurrencies-technical-study.p.rapidapi.com/crypto/macd/{symbol}/{timeframe}/{fastperiod}/{slowperiod}/{signalperiod}"
    querystring = {}
    if symbol:
        querystring['symbol'] = symbol
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cryptocurrencies-technical-study.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def williams_percent_range_w_r(symbol: str, period: int, timeframe: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The Williams Percent Range (Williams %R) is a technical analysis indicator developed by Larry Williams. It is a momentum oscillator that measures overbought and oversold levels in the market.
		
		The Williams %R indicator is calculated by taking the highest high over a certain time period and subtracting the current closing price, and then dividing that result by the highest high minus the lowest low over the same time period. This value is then multiplied by -100 to obtain a percentage value that oscillates between 0 and -100.
		
		When the Williams %R indicator is above -20, it is considered overbought and suggests that the market may be due for a reversal. Conversely, when the indicator is below -80, it is considered oversold and suggests that the market may be due for a rebound.
		
		Traders can use the Williams %R indicator in conjunction with other technical analysis tools to confirm potential trend reversals and to identify entry and exit points for trades. Overall, the Williams %R indicator is a useful tool for identifying overbought and oversold market conditions, and can help traders make more informed trading decisions."
    
    """
    url = f"https://cryptocurrencies-technical-study.p.rapidapi.com/crypto/wr/{symbol}/{timeframe}/{period}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cryptocurrencies-technical-study.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def force_index_fi(symbol: str, period: int, timeframe: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The Force Index (FI) is a technical analysis indicator that measures the strength of buying and selling pressure in the market. It was developed by Dr. Alexander Elder and is based on the idea that price movements are influenced by the strength of the force behind them.
		
		The FI is calculated by taking the difference between the current and previous price and multiplying it by the volume of trades. This value is then smoothed using a moving average to reduce the impact of short-term fluctuations.
		
		The resulting FI value is plotted on a chart, with positive values indicating buying pressure and negative values indicating selling pressure. Traders can use the FI to identify potential trend reversals and to confirm other technical analysis indicators.
		
		For example, if the FI is trending higher while the price of the security is trending lower, it may indicate that there is strong buying pressure in the market, and that a potential reversal in the price trend may be imminent.
		
		Overall, the FI is a useful tool for traders looking to stay ahead of changes in market sentiment and to identify potential trading opportunities. However, like all technical analysis indicators, it should be used in conjunction with other analysis tools and market information to make more informed trading decisions."
    
    """
    url = f"https://cryptocurrencies-technical-study.p.rapidapi.com/crypto/fi/{symbol}/{timeframe}/{period}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cryptocurrencies-technical-study.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def stochastic_oscillator_sto(timeframe: str, symbol: str, signalperiod: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The Stochastic Oscillator (STO) is a technical analysis indicator that is used to measure the momentum of a security and to identify potential trend reversals. It was developed by George Lane in the 1950s.
		
		The STO compares the closing price of a security to its price range over a specified period of time. It is calculated by taking the difference between the current closing price and the lowest low over the specified period, and then dividing that difference by the highest high over the same period. This value is then multiplied by 100 to obtain a percentage that oscillates between 0 and 100.
		
		The STO is often used in conjunction with other technical analysis tools to confirm potential trend reversals and to identify entry and exit points for trades. When the STO is above 80, it is considered overbought and suggests that the security may be due for a price decline. Conversely, when the STO is below 20, it is considered oversold and suggests that the security may be due for a price increase.
		
		Traders can adjust the time period of the STO to suit their trading strategy and the market conditions they are trading in. Overall, the STO is a useful tool for traders looking to identify potential trend reversals and to make more informed trading decisions."
    
    """
    url = f"https://cryptocurrencies-technical-study.p.rapidapi.com/crypto/sto/{symbol}/{timeframe}/{period}/{signalperiod}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cryptocurrencies-technical-study.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def on_balance_volume_obv(timeframe: str, symbol: str, period: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "On Balance Volume (OBV) is a technical analysis indicator that is used to measure buying and selling pressure in the market. The OBV is based on the idea that the volume of trades can indicate the strength of a trend.
		
		The OBV is calculated by adding the volume of trades to a running total when the price of the security increases, and subtracting the volume when the price of the security decreases. This running total is then plotted on a chart, with the trend of the OBV indicating the direction of buying and selling pressure.
		
		Traders can use the OBV to identify potential trend reversals and to confirm other technical analysis indicators. For example, if the OBV is trending higher while the price of the security is trending lower, it may indicate that there is strong buying pressure in the market, and that a potential reversal in the price trend may be imminent.
		
		The OBV is a useful tool for traders looking to stay ahead of changes in market sentiment and to identify potential trading opportunities. However, like all technical analysis indicators, it should be used in conjunction with other analysis tools and market information to make more informed trading decisions."
    
    """
    url = f"https://cryptocurrencies-technical-study.p.rapidapi.com/crypto/obv/{symbol}/{timeframe}/{period}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cryptocurrencies-technical-study.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def awesome_oscillator_ao(timeframe: str, fastperiod: str, slowperiod: str, symbol: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The Awesome Oscillator (AO) is a technical analysis indicator that is used to measure market momentum. It was developed by Bill Williams and is designed to help traders identify potential buy and sell signals.
		
		The AO is calculated by subtracting a 34-period simple moving average (SMA) from a 5-period SMA, with the resulting values plotted as a histogram. When the histogram is above the zero line, it indicates that the short-term momentum is higher than the long-term momentum, which may suggest a bullish trend. When the histogram is below the zero line, it indicates that the short-term momentum is lower than the long-term momentum, which may suggest a bearish trend.
		
		Traders can use the AO to identify potential buy and sell signals. For example, a buy signal may occur when the histogram crosses above the zero line, while a sell signal may occur when the histogram crosses below the zero line.
		
		Overall, the Awesome Oscillator is a useful tool for traders looking to identify potential changes in market momentum and to identify potential trading opportunities. However, like all technical analysis indicators, it should be used in conjunction with other analysis tools and market information to make more informed trading decisions."
    
    """
    url = f"https://cryptocurrencies-technical-study.p.rapidapi.com/crypto/ao/{symbol}/{timeframe}/{fastperiod}/{slowperiod}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cryptocurrencies-technical-study.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def rate_of_change_roc(symbol: str, period: int, timeframe: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The Rate of Change (ROC) is a technical analysis indicator that is used to measure the percentage change in a security's price over a specified period of time. It is a momentum indicator that helps traders identify potential trend reversals and confirm other technical analysis indicators.
		
		The ROC is calculated by taking the current price of a security and dividing it by the price from a specified number of periods ago. The resulting value is then multiplied by 100 to provide a percentage change in the security's price.
		
		A positive ROC reading indicates that the security's price has increased over the specified period of time, while a negative ROC reading indicates that the security's price has decreased. Traders typically use a ROC period of 14 days, but the period can be adjusted to meet specific trading strategies.
		
		Traders can use the ROC in conjunction with other technical analysis tools to gain a more complete picture of the market and to make more informed trading decisions. Overall, the ROC is a useful tool for traders looking to identify potential trend reversals and to stay ahead of changes in price action."
    
    """
    url = f"https://cryptocurrencies-technical-study.p.rapidapi.com/crypto/roc/{symbol}/{timeframe}/{period}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cryptocurrencies-technical-study.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def average_true_range_atr(symbol: str, timeframe: str, period: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Average True Range (ATR) is a technical analysis indicator that is used to measure volatility in financial markets. Developed by J. Welles Wilder, ATR calculates the average range of price movements over a given period of time, taking into account any gaps in price movement.
		
		The ATR is calculated as the moving average of the true range, where the true range is the maximum value of the following three calculations:
		
		Current high minus current low
		Absolute value of current high minus previous close
		Absolute value of current low minus previous close
		The ATR value is expressed in the same units as the price of the asset being analyzed, such as dollars for stocks or pips for forex. A higher ATR indicates greater volatility, while a lower ATR indicates lower volatility.
		
		Traders use the ATR to determine appropriate levels for stop-loss orders, as well as to identify potential price breakouts or reversals. For example, if the ATR value is high, it may suggest that the price of the asset is likely to move more in the near future, indicating that wider stop-loss orders may be necessary to accommodate for potential price swings.
		
		Overall, the Average True Range is a useful tool for traders looking to measure market volatility and to manage risk in their trading strategies. However, like all technical analysis indicators, it should be used in conjunction with other analysis tools and market information to make more informed trading decisions."
    
    """
    url = f"https://cryptocurrencies-technical-study.p.rapidapi.com/crypto/atr/{symbol}/{timeframe}/{period}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cryptocurrencies-technical-study.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def relative_strength_index_rsi(timeframe: str, period: int, symbol: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The Relative Strength Index (RSI) is a technical analysis indicator that is used to measure the strength and momentum of a security's price action. The RSI is calculated based on the average gains and losses of a security over a specified period of time.
		
		The RSI ranges from 0 to 100, with a reading of 70 or above indicating that the security is overbought and a reading of 30 or below indicating that the security is oversold. Traders use the RSI to identify potential trend reversals, as well as to confirm other technical analysis indicators.
		
		The RSI is calculated by comparing the average gain of a security's price during a specified period of time to the average loss during the same period. The resulting ratio is then plotted on a chart and smoothed to provide a more accurate reading of the security's momentum.
		
		Traders can use the RSI in conjunction with other technical analysis tools to gain a more complete picture of the market and to make more informed trading decisions. Overall, the RSI is a useful tool for traders looking to identify potential trend reversals and to stay ahead of changes in price action."
    
    """
    url = f"https://cryptocurrencies-technical-study.p.rapidapi.com/crypto/rsi/{symbol}/{timeframe}/{period}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cryptocurrencies-technical-study.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def simple_moving_average_sma(symbol: str, period: int, timeframe: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The Simple Moving Average (SMA) is a technical analysis indicator that is used to identify trends in the market. It is a type of Moving Average (MA) that calculates the average price of a security over a specified period of time, with each price point given equal weight.
		
		The SMA is a commonly used tool for traders as it is simple to calculate and interpret. It can be used to identify bullish and bearish trends, as well as to confirm other technical analysis indicators.
		
		For example, if a trader is using a 50-day SMA, the SMA would be calculated by adding up the closing prices of the security for the past 50 days and then dividing by 50. This value is then plotted on a chart and can be used as a reference point for potential entry and exit points.
		
		The SMA is often used in conjunction with other technical analysis tools to gain a more complete picture of the market and to make more informed trading decisions. Overall, the SMA is a useful tool for traders looking to identify trends in the market and to stay ahead of changes in price action."
    
    """
    url = f"https://cryptocurrencies-technical-study.p.rapidapi.com/crypto/sma/{symbol}/{timeframe}/{period}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cryptocurrencies-technical-study.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def average_directional_index_adx(symbol: str, timeframe: str, period: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Average Directional Index (ADX) is a technical analysis indicator that is used to measure the strength of a trend. Developed by J. Welles Wilder, ADX is a component of the Directional Movement System, which also includes the Positive Directional Indicator (+DI) and the Negative Directional Indicator (-DI).
		
		ADX is calculated by taking the difference between the smoothed moving average of the Positive Directional Indicator (+DI) and Negative Directional Indicator (-DI), and then dividing it by the smoothed moving average of the True Range (TR). The resulting value is expressed as a percentage and ranges between 0 and 100.
		
		A higher ADX value indicates a stronger trend, whether it is bullish or bearish. Traders can use ADX to identify whether a market is trending or consolidating. When the ADX value is above 25, it suggests that the market is trending, while a value below 25 suggests that the market is consolidating.
		
		Traders can also use the ADX to determine when to enter or exit a trade. For example, when the ADX value is above 25 and rising, it suggests that the trend is gaining strength, and a trader may consider entering a position. When the ADX value is above 25 and falling, it suggests that the trend is losing strength, and a trader may consider exiting a position.
		
		Overall, the Average Directional Index is a useful tool for traders looking to identify trends and determine appropriate entry and exit points for their trades. However, like all technical analysis indicators, it should be used in conjunction with other analysis tools and market information to make more informed trading decisions."
    
    """
    url = f"https://cryptocurrencies-technical-study.p.rapidapi.com/crypto/adx/{symbol}/{timeframe}/{period}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cryptocurrencies-technical-study.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def parabolic_stop_and_reverse_psar(symbol: str, timeframe: str, step: int, max: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The Parabolic Stop and Reverse (PSAR) is a technical analysis indicator that is used to identify potential trend reversals in a security's price action. It is also known as the Parabolic SAR (Stop and Reverse) or just SAR.
		
		The PSAR is calculated by plotting a series of dots above or below the price of a security. The dots move in relation to the price action, with the distance between the dots increasing as the trend strengthens and decreasing as the trend weakens.
		
		The PSAR is designed to provide traders with a stop loss level that can be adjusted as the trend changes. When a long position is taken, the PSAR dots are plotted below the price action, and the stop loss level is moved up as the price increases. When a short position is taken, the PSAR dots are plotted above the price action, and the stop loss level is moved down as the price decreases.
		
		Traders can use the PSAR in conjunction with other technical analysis tools to gain a more complete picture of the market and to make more informed trading decisions. Overall, the PSAR is a useful tool for traders looking to identify potential trend reversals and to manage their risk when trading."
    
    """
    url = f"https://cryptocurrencies-technical-study.p.rapidapi.com/crypto/psar/{symbol}/{timeframe}/{step}/{max}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cryptocurrencies-technical-study.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def volume_weighted_average_price_vwap(timeframe: str, symbol: str, period: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "VThe Volume Weighted Average Price (VWAP) is a technical analysis indicator that calculates the average price a security has traded at during the day, weighted by the volume traded at each price level.
		
		The VWAP is calculated by taking the sum of the product of each price and its corresponding volume traded, and then dividing that sum by the total volume traded for the day. This provides a more accurate representation of the average price traded for the day, as it takes into account the volume traded at each price level.
		
		The VWAP is often used by institutional traders and algorithmic traders as a benchmark for the day's trading performance, and can be used to identify support and resistance levels as well as potential entry and exit points for trades.
		
		Traders can also use the VWAP to compare a security's current price to its average traded price for the day, and use this information to determine whether the security is overvalued or undervalued. Overall, the VWAP is a useful tool for traders looking to gain insight into a security's price action and trading activity over the course of a day."
    
    """
    url = f"https://cryptocurrencies-technical-study.p.rapidapi.com/crypto/vwap/{symbol}/{timeframe}/{period}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cryptocurrencies-technical-study.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def exponential_moving_average_ema(symbol: str, timeframe: str, period: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The Exponential Moving Average (EMA) is a technical analysis indicator that is used to identify trends in the market. It is a variation of the Moving Average (MA), which assigns more weight to recent price data than older data.
		
		The EMA is calculated by taking the average of the price data over a specified period of time, with each data point weighted according to its age. This means that the more recent data points have a higher weight than older data points.
		
		The EMA is designed to provide a more responsive signal to changes in the market than other types of moving averages. It can be used to identify bullish and bearish trends, as well as to confirm other technical analysis indicators.
		
		Traders can use the EMA in conjunction with other technical analysis tools to gain a more complete picture of the market and to make more informed trading decisions. Overall, the EMA is a useful tool for traders looking to identify trends in the market and to stay ahead of changes in price action."
    
    """
    url = f"https://cryptocurrencies-technical-study.p.rapidapi.com/crypto/ema/{symbol}/{timeframe}/{period}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cryptocurrencies-technical-study.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def cryptocurrency_patterns_recognition(timeframe: str, symbol: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Experience the power of cryptocurrency pattern recognition with our cutting-edge technology.
		
		Our advanced algorithms and real-time analysis provide you with valuable insights into the latest trends and movements in the crypto market. With our user-friendly interface and technical analysis indicators, you can make informed decisions and stay ahead of the competition.
		
		Our API offers a wide range of patterns including abandoned baby, bearish engulfing pattern, bullish engulfing pattern, dark cloud cover, downside tasuki gap, doji, dragonfly doji, gravestone doji, bullish harami, bearish harami cross, bullish harami cross, bullish marubozu, bearish marubozu, evening doji star, evening star, bearish harami, piercing line, bullish spinning top, bearish spinning top, morning doji star, morning star, three black crows, three white soldiers, bullish hammer stick, bearish hammer stick, bullish inverted hammer stick, tweezertop, and tweezer bottom."
    
    """
    url = f"https://cryptocurrencies-technical-study.p.rapidapi.com/crypto/patterns/{symbol}/{timeframe}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cryptocurrencies-technical-study.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def moneyflow_index_mfi(timeframe: str, symbol: str, period: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The Money Flow Index (MFI) is a technical analysis indicator that measures buying and selling pressure in the market. It is a momentum oscillator that uses both price and volume data to identify potential trend reversals.
		
		The MFI is calculated by taking the typical price of a security over a specified period of time and multiplying it by the volume of trades. This value is then divided by the sum of the values for positive money flow and negative money flow. Positive money flow is the sum of the typical prices when the price increases, while negative money flow is the sum of the typical prices when the price decreases.
		
		The resulting MFI value is plotted on a scale of 0 to 100, with readings above 80 indicating overbought conditions and readings below 20 indicating oversold conditions. Traders can use the MFI in conjunction with other technical analysis indicators to identify potential trend reversals and to confirm other market signals.
		
		Overall, the MFI is a useful tool for traders looking to stay ahead of changes in market sentiment and to identify potential trading opportunities. However, like all technical analysis indicators, it should be used in conjunction with other analysis tools and market information to make more informed trading decisions."
    
    """
    url = f"https://cryptocurrencies-technical-study.p.rapidapi.com/crypto/mfi/{symbol}/{timeframe}/{period}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cryptocurrencies-technical-study.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def cryptocurrencies_symbols_list(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns the list of the supported cryptocurrencies symbols"
    
    """
    url = f"https://cryptocurrencies-technical-study.p.rapidapi.com/crypto/symbols"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cryptocurrencies-technical-study.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

