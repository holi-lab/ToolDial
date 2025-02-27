import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def donchian_channels(symbol: str, useheikincandles: bool, exchange: str, klineinterval: str, lookbackperiods: int=20, resulttype: str='LastCandles', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Created by Richard Donchian, Donchian Channels, also called Price Channels, are price ranges derived from highest High and lowest Low values."
    symbol: eg.  BTC-USDT
        exchange: Supported Exchanges :
BINANCE
BINANCE-FUTURES
BITFINEX
BITTREX
KRAKEN
BYBIT
HUOBI
KUCOIN
COINEX
        klineinterval: 
1m : OneMinute
3m : ThreeMinutes
5m : FiveMinutes
15m : FifteenMinutes
30m : ThirtyMinutes
1h : OneHour
4h : FourHour
6h : SixHour
12h : TwelveHour
1d : OneDay
1w : OneWeek
1mo : OneMonth
        lookbackperiods: Number of periods (N) for lookback period. Must be greater than 0 to calculate; however we suggest a larger value for an appropriate sample size. Default is 20.
        resulttype: Full : Returns all results
LastOpenCandle : Last opened candle
LastClosedCandle : Last Closed Candle
LastCandles : Last closed and last opened candles.
        
    """
    url = f"https://cryptocator.p.rapidapi.com/DonchianChannels"
    querystring = {'Symbol': symbol, 'useHeikinCandles': useheikincandles, 'Exchange': exchange, 'KlineInterval': klineinterval, }
    if lookbackperiods:
        querystring['lookbackPeriods'] = lookbackperiods
    if resulttype:
        querystring['resultType'] = resulttype
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cryptocator.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def standard_deviation_channels(exchange: str, useheikincandles: bool, klineinterval: str, symbol: str, stddeviations: int=2, resulttype: str='LastCandles', lookbackperiods: int=20, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Standard Deviation Channels are prices ranges based on an linear regression centerline and standard deviations band widths."
    exchange: Supported Exchanges :
BINANCE
BINANCE-FUTURES
BITFINEX
BITTREX
KRAKEN
BYBIT
HUOBI
KUCOIN
COINEX
        klineinterval: 
1m : OneMinute
3m : ThreeMinutes
5m : FiveMinutes
15m : FifteenMinutes
30m : ThirtyMinutes
1h : OneHour
4h : FourHour
6h : SixHour
12h : TwelveHour
1d : OneDay
1w : OneWeek
1mo : OneMonth
        symbol: eg.  BTC-USDT
        stddeviations: Width of bands. Standard deviations (D) from the regression line. Must be greater than 0. Default is 2.
        resulttype: Full : Returns all results
LastOpenCandle : Last opened candle
LastClosedCandle : Last Closed Candle
LastCandles : Last closed and last opened candles.
        lookbackperiods: Size (N) of the evaluation window. Must be null or greater than 1 to calculate. A null value will produce a full quotes evaluation window. Default is 20.
        
    """
    url = f"https://cryptocator.p.rapidapi.com/StandardDeviationChannels"
    querystring = {'Exchange': exchange, 'useHeikinCandles': useheikincandles, 'KlineInterval': klineinterval, 'Symbol': symbol, }
    if stddeviations:
        querystring['stdDeviations'] = stddeviations
    if resulttype:
        querystring['resultType'] = resulttype
    if lookbackperiods:
        querystring['lookbackPeriods'] = lookbackperiods
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cryptocator.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def starc_bands(klineinterval: str, useheikincandles: bool, symbol: str, exchange: str, smaperiods: int=7, pointtype: int=10, multiplier: int=2, resulttype: str='LastCandles', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Created by Manning Stoller, the Stoller Average Range Channel (STARC) Bands, are price ranges based on an SMA centerline and ATR band widths. See also Keltner Channels for an EMA centerline equivalent."
    klineinterval: 
1m : OneMinute
3m : ThreeMinutes
5m : FiveMinutes
15m : FifteenMinutes
30m : ThirtyMinutes
1h : OneHour
4h : FourHour
6h : SixHour
12h : TwelveHour
1d : OneDay
1w : OneWeek
1mo : OneMonth
        symbol: eg.  BTC-USDT
        exchange: Supported Exchanges :
BINANCE
BINANCE-FUTURES
BITFINEX
BITTREX
KRAKEN
BYBIT
HUOBI
KUCOIN
COINEX
        smaperiods: Number of lookback periods (S) for the center line moving average. Must be greater than 1 to calculate and is typically between 5 and 10.
        pointtype: Number of lookback periods (A) for the Average True Range. Must be greater than 1 to calculate and is typically the same value as smaPeriods. Default is 10.
        multiplier: ATR Multiplier. Must be greater than 0. Default is 2.
        resulttype: Full : Returns all results
LastOpenCandle : Last opened candle
LastClosedCandle : Last Closed Candle
LastCandles : Last closed and last opened candles.
        
    """
    url = f"https://cryptocator.p.rapidapi.com/STARCBands"
    querystring = {'KlineInterval': klineinterval, 'useHeikinCandles': useheikincandles, 'Symbol': symbol, 'Exchange': exchange, }
    if smaperiods:
        querystring['smaPeriods'] = smaperiods
    if pointtype:
        querystring['pointType'] = pointtype
    if multiplier:
        querystring['multiplier'] = multiplier
    if resulttype:
        querystring['resultType'] = resulttype
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cryptocator.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def rolling_pivot_points(symbol: str, exchange: str, klineinterval: str, useheikincandles: bool, windowperiods: int=14, offsetperiods: int=14, pointtype: str='STANDARD', resulttype: str='LastCandles', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Created by Dave Skender, Rolling Pivot Points is a modern update to traditional fixed calendar window Pivot Points. It depicts support and resistance levels, based on a defined rolling window and offset."
    symbol: eg.  BTC-USDT
        exchange: Supported Exchanges :
BINANCE
BINANCE-FUTURES
BITFINEX
BITTREX
KRAKEN
BYBIT
HUOBI
KUCOIN
COINEX
        klineinterval: 
1m : OneMinute
3m : ThreeMinutes
5m : FiveMinutes
15m : FifteenMinutes
30m : ThirtyMinutes
1h : OneHour
4h : FourHour
6h : SixHour
12h : TwelveHour
1d : OneDay
1w : OneWeek
1mo : OneMonth
        windowperiods: Number of periods (W) in the evaluation window. Must be greater than 0 to calculate; but is typically specified in the 5-20 range.
        offsetperiods: Number of periods (F) to offset the window from the current period. Must be greater than or equal to 0 and is typically less than or equal to W.
        pointtype: Type of Pivot Point. Default is STANDARD

STANDARD	: FLOOR TRADING (DEFAULT)
CAMARILLA	: CAMARILLA
DEMARK	: DEMARK
FIBONACCI	: FIBONACCI
WOODIE	: WOODIE
        resulttype: Full : Returns all results
LastOpenCandle : Last opened candle
LastClosedCandle : Last Closed Candle
LastCandles : Last closed and last opened candles.
        
    """
    url = f"https://cryptocator.p.rapidapi.com/RollingPivotPoints"
    querystring = {'Symbol': symbol, 'Exchange': exchange, 'KlineInterval': klineinterval, 'useHeikinCandles': useheikincandles, }
    if windowperiods:
        querystring['windowPeriods'] = windowperiods
    if offsetperiods:
        querystring['offsetPeriods'] = offsetperiods
    if pointtype:
        querystring['pointType'] = pointtype
    if resulttype:
        querystring['resultType'] = resulttype
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cryptocator.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def pivot_points(useheikincandles: bool, exchange: str, klineinterval: str, symbol: str, pointtype: str='STANDARD', resulttype: str='LastCandles', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Pivot Points depict support and resistance levels, based on prior calendar windows. You can specify window size (e.g. month, week, day, etc) and any of the traditional Floor Trading, Camarilla, Demark, Fibonacci, and Woodie variants."
    exchange: Supported Exchanges :
BINANCE
BINANCE-FUTURES
BITFINEX
BITTREX
KRAKEN
BYBIT
HUOBI
KUCOIN
COINEX
        klineinterval: 
1h : OneHour
4h : FourHour
6h : SixHour
12h : TwelveHour
1d : OneDay
1w : OneWeek
1mo : OneMonth
        symbol: eg.  BTC-USDT
        pointtype: Type of Pivot Point. Default is STANDARD

STANDARD	: FLOOR TRADING (DEFAULT)
CAMARILLA	: CAMARILLA
DEMARK	: DEMARK
FIBONACCI	: FIBONACCI
WOODIE	: WOODIE
        resulttype: Full : Returns all results
LastOpenCandle : Last opened candle
LastClosedCandle : Last Closed Candle
LastCandles : Last closed and last opened candles.
        
    """
    url = f"https://cryptocator.p.rapidapi.com/PivotPoints"
    querystring = {'useHeikinCandles': useheikincandles, 'Exchange': exchange, 'KlineInterval': klineinterval, 'Symbol': symbol, }
    if pointtype:
        querystring['pointType'] = pointtype
    if resulttype:
        querystring['resultType'] = resulttype
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cryptocator.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def moving_average_envelopes(symbol: str, klineinterval: str, exchange: str, useheikincandles: bool, resulttype: str='LastCandles', movingaveragetype: str='SMA', percentoffset: int=2, lookbackperiods: int=14, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Moving Average Envelopes is a price band channel overlay that is offset from the moving average of price."
    symbol: eg.  BTC-USDT
        klineinterval: 1m : OneMinute
3m : ThreeMinutes
5m : FiveMinutes
15m : FifteenMinutes
30m : ThirtyMinutes
1h : OneHour
4h : FourHour
6h : SixHour
12h : TwelveHour
1d : OneDay
1w : OneWeek
1mo : OneMonth
        exchange: Supported Exchanges :
BINANCE
BINANCE-FUTURES
BITFINEX
BITTREX
KRAKEN
BYBIT
HUOBI
KUCOIN
COINEX
        resulttype: Full : Returns all results
LastOpenCandle : Last opened candle
LastClosedCandle : Last Closed Candle
LastCandles : Last closed and last opened candles.
        movingaveragetype: Type of moving average (e.g. SMA, EMA, HMA).

Supported Types:
ALMA	Arnaud Legoux Moving Average
DEMA	Double Exponential Moving Average
EPMA	Endpoint Moving Average
EMA	Exponential Moving Average
HMA	Hull Moving Average
SMA	Simple Moving Average (default)
SMMA	Smoothed Moving Average
TEMA	Triple Exponential Moving Average
WMA	Weighted Moving Average
        percentoffset: Percent offset for envelope width. Example: 3.5% would be entered as 3.5 (not 0.035). Must be greater than 0. Typical values range from 2 to 10. Default is 2.5.
        lookbackperiods: Number of periods (N) in the moving average. Must be greater than 1.
        
    """
    url = f"https://cryptocator.p.rapidapi.com/MovingAverageEnvelopes"
    querystring = {'Symbol': symbol, 'KlineInterval': klineinterval, 'Exchange': exchange, 'useHeikinCandles': useheikincandles, }
    if resulttype:
        querystring['resultType'] = resulttype
    if movingaveragetype:
        querystring['movingAverageType'] = movingaveragetype
    if percentoffset:
        querystring['percentOffset'] = percentoffset
    if lookbackperiods:
        querystring['lookbackPeriods'] = lookbackperiods
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cryptocator.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def keltner_channels(useheikincandles: bool, symbol: str, exchange: str, klineinterval: str, emaperiods: int=20, multiplier: int=2, atrperiods: int=10, resulttype: str='LastCandles', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Created by Chester W. Keltner, Keltner Channels are based on an EMA centerline and ATR band widths. See also STARC Bands for an SMA centerline equivalent."
    symbol: eg.  BTC-USDT
        exchange: Supported Exchanges :
BINANCE
BINANCE-FUTURES
BITFINEX
BITTREX
KRAKEN
BYBIT
HUOBI
KUCOIN
COINEX
        klineinterval: 1m : OneMinute
3m : ThreeMinutes
5m : FiveMinutes
15m : FifteenMinutes
30m : ThirtyMinutes
1h : OneHour
4h : FourHour
6h : SixHour
12h : TwelveHour
1d : OneDay
1w : OneWeek
1mo : OneMonth
        emaperiods: Number of lookback periods (E) for the center line moving average. Must be greater than 1 to calculate. Default is 20.
        multiplier: ATR Multiplier. Must be greater than 0. Default is 2.
        atrperiods: Number of lookback periods (A) for the Average True Range. Must be greater than 1 to calculate. Default is 10.
        resulttype: Full : Returns all results
LastOpenCandle : Last opened candle
LastClosedCandle : Last Closed Candle
LastCandles : Last closed and last opened candles.
        
    """
    url = f"https://cryptocator.p.rapidapi.com/KeltnerChannels"
    querystring = {'useHeikinCandles': useheikincandles, 'Symbol': symbol, 'Exchange': exchange, 'KlineInterval': klineinterval, }
    if emaperiods:
        querystring['emaPeriods'] = emaperiods
    if multiplier:
        querystring['multiplier'] = multiplier
    if atrperiods:
        querystring['atrPeriods'] = atrperiods
    if resulttype:
        querystring['resultType'] = resulttype
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cryptocator.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def fractal_chaos_bands_fcb(exchange: str, symbol: str, useheikincandles: bool, klineinterval: str, windowspan: int=2, resulttype: str='LastCandles', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Created by Edward William Dreiss, Fractal Chaos Bands outline high and low price channels to depict broad less-chaotic price movements. FCB is a channelized depiction of Williams Fractal."
    exchange: Supported Exchanges :
BINANCE
BINANCE-FUTURES
BITFINEX
BITTREX
KRAKEN
BYBIT
HUOBI
KUCOIN
COINEX
        symbol: eg.  BTC-USDT
        klineinterval: 1m : OneMinute
3m : ThreeMinutes
5m : FiveMinutes
15m : FifteenMinutes
30m : ThirtyMinutes
1h : OneHour
4h : FourHour
6h : SixHour
12h : TwelveHour
1d : OneDay
1w : OneWeek
1mo : OneMonth
        windowspan: Fractal evaluation window span width (S). Must be at least 2. Default is 2.
        resulttype: Full : Returns all results
LastOpenCandle : Last opened candle
LastClosedCandle : Last Closed Candle
LastCandles : Last closed and last opened candles.
        
    """
    url = f"https://cryptocator.p.rapidapi.com/FractalChaosBands"
    querystring = {'Exchange': exchange, 'Symbol': symbol, 'useHeikinCandles': useheikincandles, 'KlineInterval': klineinterval, }
    if windowspan:
        querystring['windowSpan'] = windowspan
    if resulttype:
        querystring['resultType'] = resulttype
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cryptocator.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def bollinger_bands(symbol: str, exchange: str, useheikincandles: bool, klineinterval: str, lookbackperiods: int=20, resulttype: str='LastCandles', standarddeviations: int=2, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Created by John Bollinger, Bollinger Bands price channels depict volatility as standard deviation boundary line range from a moving average of price. Bollinger Bands® is a registered trademark of John A. Bollinger."
    symbol: eg. BTC-USDT
        exchange: Supported Exchanges :
BINANCE
BINANCE-FUTURES
BITFINEX
BITTREX
KRAKEN
BYBIT
HUOBI
KUCOIN
COINEX
        klineinterval: 1m : OneMinute
3m : ThreeMinutes
5m : FiveMinutes
15m : FifteenMinutes
30m : ThirtyMinutes
1h : OneHour
4h : FourHour
6h : SixHour
12h : TwelveHour
1d : OneDay
1w : OneWeek
1mo : OneMonth
        lookbackperiods: Number of periods (N) for the center line moving average. Must be greater than 1 to calculate; however we suggest a larger period for statistically appropriate sample size. Default is 20.
        resulttype: Full : Returns all results
LastOpenCandle : Last opened candle
LastClosedCandle : Last Closed Candle
LastCandles : Last closed and last opened candles.
        standarddeviations: Width of bands. Standard deviations (D) from the moving average. Must be greater than 0. Default is 2.
        
    """
    url = f"https://cryptocator.p.rapidapi.com/BollingerBands"
    querystring = {'Symbol': symbol, 'Exchange': exchange, 'useHeikinCandles': useheikincandles, 'KlineInterval': klineinterval, }
    if lookbackperiods:
        querystring['lookbackPeriods'] = lookbackperiods
    if resulttype:
        querystring['resultType'] = resulttype
    if standarddeviations:
        querystring['standardDeviations'] = standarddeviations
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cryptocator.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def hurst_exponent(klineinterval: str, symbol: str, exchange: str, useheikincandles: bool, resulttype: str='LastCandles', lookbackperiods: int=100, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The Hurst Exponent (H) is part of a Rescaled Range Analysis, a random-walk path analysis that measures trending and mean-reverting tendencies of incremental return values. When H is greater than 0.5 it depicts trending. When H is less than 0.5 it is is more likely to revert to the mean. When H is around 0.5 it represents a random walk."
    klineinterval: 1m : OneMinute
3m : ThreeMinutes
5m : FiveMinutes
15m : FifteenMinutes
30m : ThirtyMinutes
1h : OneHour
4h : FourHour
6h : SixHour
12h : TwelveHour
1d : OneDay
1w : OneWeek
1mo : OneMonth
        symbol: eg.  BTC-USDT
        exchange: Supported Exchanges :
BINANCE
BINANCE-FUTURES
BITFINEX
BITTREX
KRAKEN
BYBIT
HUOBI
KUCOIN
COINEX
        resulttype: Full : Returns all results
LastOpenCandle : Last opened candle
LastClosedCandle : Last Closed Candle
LastCandles : Last closed and last opened candles.
        
    """
    url = f"https://cryptocator.p.rapidapi.com/HurstExponent"
    querystring = {'KlineInterval': klineinterval, 'Symbol': symbol, 'Exchange': exchange, 'useHeikinCandles': useheikincandles, }
    if resulttype:
        querystring['resultType'] = resulttype
    if lookbackperiods:
        querystring['lookbackPeriods'] = lookbackperiods
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cryptocator.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def gator_oscillator(symbol: str, exchange: str, klineinterval: str, resulttype: str='LastCandles', useheikincandles: bool=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Created by Bill Williams, the Gator Oscillator is an expanded oscillator view of Williams Alligator’s three moving averages."
    symbol: eg.  BTC-USDT
        exchange: Supported Exchanges :
BINANCE
BINANCE-FUTURES
BITFINEX
BITTREX
KRAKEN
BYBIT
HUOBI
KUCOIN
COINEX
        klineinterval: 1m : OneMinute
3m : ThreeMinutes
5m : FiveMinutes
15m : FifteenMinutes
30m : ThirtyMinutes
1h : OneHour
4h : FourHour
6h : SixHour
12h : TwelveHour
1d : OneDay
1w : OneWeek
1mo : OneMonth
        resulttype: Full : Returns all results
LastOpenCandle : Last opened candle
LastClosedCandle : Last Closed Candle
LastCandles : Last closed and last opened candles.
        
    """
    url = f"https://cryptocator.p.rapidapi.com/GatorOscillator"
    querystring = {'Symbol': symbol, 'Exchange': exchange, 'KlineInterval': klineinterval, }
    if resulttype:
        querystring['resultType'] = resulttype
    if useheikincandles:
        querystring['useHeikinCandles'] = useheikincandles
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cryptocator.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def directional_movement_index_dmi(exchange: str, symbol: str, klineinterval: str, useheikincandles: bool, lookbackperiods: int=14, resulttype: str='LastCandles', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Created by J. Welles Wilder, the Directional Movement Index (DMI) and Average Directional Movement Index (ADX) is a measure of price directional movement. It includes upward and downward indicators, and is often used to measure strength of trend."
    exchange: Supported Exchanges :
BINANCE
BINANCE-FUTURES
BITFINEX
BITTREX
KRAKEN
BYBIT
HUOBI
KUCOIN
COINEX
        symbol: eg.  BTC-USDT
        klineinterval: 1m : OneMinute
3m : ThreeMinutes
5m : FiveMinutes
15m : FifteenMinutes
30m : ThirtyMinutes
1h : OneHour
4h : FourHour
6h : SixHour
12h : TwelveHour
1d : OneDay
1w : OneWeek
1mo : OneMonth
        lookbackperiods: Number of periods (N) for the lookback evaluation. Must be greater than 0. Default is 25.
        resulttype: Full : Returns all results
LastOpenCandle : Last opened candle
LastClosedCandle : Last Closed Candle
LastCandles : Last closed and last opened candles.
        
    """
    url = f"https://cryptocator.p.rapidapi.com/Adx"
    querystring = {'Exchange': exchange, 'Symbol': symbol, 'KlineInterval': klineinterval, 'useHeikinCandles': useheikincandles, }
    if lookbackperiods:
        querystring['lookbackPeriods'] = lookbackperiods
    if resulttype:
        querystring['resultType'] = resulttype
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cryptocator.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def average_directional_index_adx(symbol: str, exchange: str, useheikincandles: bool, klineinterval: str, lookbackperiods: int=14, resulttype: str='LastCandles', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Created by J. Welles Wilder, the Directional Movement Index (DMI) and Average Directional Movement Index (ADX) is a measure of price directional movement. It includes upward and downward indicators, and is often used to measure strength of trend."
    symbol: eg.  BTC-USDT
        exchange: Supported Exchanges :
BINANCE
BINANCE-FUTURES
BITFINEX
BITTREX
KRAKEN
BYBIT
HUOBI
KUCOIN
COINEX
        klineinterval: 1m : OneMinute
3m : ThreeMinutes
5m : FiveMinutes
15m : FifteenMinutes
30m : ThirtyMinutes
1h : OneHour
4h : FourHour
6h : SixHour
12h : TwelveHour
1d : OneDay
1w : OneWeek
1mo : OneMonth
        lookbackperiods: Number of periods (N) for the lookback evaluation. Must be greater than 0. Default is 25.
        resulttype: Full : Returns all results
LastOpenCandle : Last opened candle
LastClosedCandle : Last Closed Candle
LastCandles : Last closed and last opened candles.
        
    """
    url = f"https://cryptocator.p.rapidapi.com/Adx"
    querystring = {'Symbol': symbol, 'Exchange': exchange, 'useHeikinCandles': useheikincandles, 'KlineInterval': klineinterval, }
    if lookbackperiods:
        querystring['lookbackPeriods'] = lookbackperiods
    if resulttype:
        querystring['resultType'] = resulttype
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cryptocator.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def aroon(exchange: str, symbol: str, klineinterval: str, useheikincandles: bool, resulttype: str='LastCandles', lookbackperiods: int=25, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Created by Tushar Chande, Aroon is a oscillator view of how long ago the new high or low price occurred."
    exchange: Supported Exchanges :
BINANCE
BINANCE-FUTURES
BITFINEX
BITTREX
KRAKEN
BYBIT
HUOBI
KUCOIN
COINEX
        symbol: eg.  BTC-USDT
        klineinterval: 
1m : OneMinute
3m : ThreeMinutes
5m : FiveMinutes
15m : FifteenMinutes
30m : ThirtyMinutes
1h : OneHour
4h : FourHour
6h : SixHour
12h : TwelveHour
1d : OneDay
1w : OneWeek
1mo : OneMonth
        resulttype: Full : Returns all results
LastOpenCandle : Last opened candle
LastClosedCandle : Last Closed Candle
LastCandles : Last closed and last opened candles.
        lookbackperiods: Number of periods (N) for the lookback evaluation. Must be greater than 0. Default is 25.
        
    """
    url = f"https://cryptocator.p.rapidapi.com/Aroon"
    querystring = {'Exchange': exchange, 'Symbol': symbol, 'KlineInterval': klineinterval, 'useHeikinCandles': useheikincandles, }
    if resulttype:
        querystring['resultType'] = resulttype
    if lookbackperiods:
        querystring['lookbackPeriods'] = lookbackperiods
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cryptocator.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def ulcer_index_ui(exchange: str, symbol: str, useheikincandles: bool, klineinterval: str, lookbackperiods: int=14, resulttype: str='LastCandles', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Created by Peter Martin, the Ulcer Index is a measure of downside price volatility over a lookback window. Often called the “heart attack” score, it measures the amount of pain seen from drawdowns in financial market prices and portfolio value."
    exchange: Supported Exchanges :
BINANCE
BINANCE-FUTURES
BITFINEX
BITTREX
KRAKEN
BYBIT
HUOBI
KUCOIN
COINEX
        symbol: eg. BTC-USDT
        klineinterval: 
1m : OneMinute
3m : ThreeMinutes
5m : FiveMinutes
15m : FifteenMinutes
30m : ThirtyMinutes
1h : OneHour
4h : FourHour
6h : SixHour
12h : TwelveHour
1d : OneDay
1w : OneWeek
1mo : OneMonth
        lookbackperiods: Number of periods (N) for review. Must be greater than 0. Default is 14.

        resulttype: Full : Returns all results
LastOpenCandle : Last opened candle
LastClosedCandle : Last Closed Candle
LastCandles : Last closed and last opened candles.
        
    """
    url = f"https://cryptocator.p.rapidapi.com/UI"
    querystring = {'Exchange': exchange, 'Symbol': symbol, 'useHeikinCandles': useheikincandles, 'KlineInterval': klineinterval, }
    if lookbackperiods:
        querystring['lookbackPeriods'] = lookbackperiods
    if resulttype:
        querystring['resultType'] = resulttype
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cryptocator.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def true_strength_index_tsi(klineinterval: str, exchange: str, useheikincandles: bool, symbol: str, smoothperiods: int=13, lookbackperiods: int=25, signalperiods: int=7, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Created by William Blau, the True Strength Index is a momentum oscillator that uses a series of exponential moving averages to depicts trends in price changes."
    klineinterval: 
1m : OneMinute
3m : ThreeMinutes
5m : FiveMinutes
15m : FifteenMinutes
30m : ThirtyMinutes
1h : OneHour
4h : FourHour
6h : SixHour
12h : TwelveHour
1d : OneDay
1w : OneWeek
1mo : OneMonth
        exchange: Supported Exchanges :
BINANCE
BINANCE-FUTURES
BITFINEX
BITTREX
KRAKEN
BYBIT
HUOBI
KUCOIN
COINEX
        symbol: eg. BTC-USDT
        smoothperiods: Number of periods (M) for the second smoothing. Must be greater than 0. Default is 13.

        lookbackperiods: Number of periods (N) for the first EMA. Must be greater than 0. Default is 25.

        signalperiods: Number of periods (S) in the TSI moving average. Must be greater than or equal to 0. Default is 7.

        
    """
    url = f"https://cryptocator.p.rapidapi.com/TSI"
    querystring = {'KlineInterval': klineinterval, 'Exchange': exchange, 'useHeikinCandles': useheikincandles, 'Symbol': symbol, }
    if smoothperiods:
        querystring['smoothPeriods'] = smoothperiods
    if lookbackperiods:
        querystring['lookbackPeriods'] = lookbackperiods
    if signalperiods:
        querystring['signalPeriods'] = signalperiods
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cryptocator.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def rate_of_change_roc(symbol: str, exchange: str, klineinterval: str, useheikincandles: bool, lookbackperiods: int=14, smaperiods: int=7, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Rate of Change, also known as Momentum Oscillator, is the percent change of price over a lookback window. A Rate of Change with Bands variant, created by Vitali Apirine, is also included."
    symbol: eg. BTC-USDT
        exchange: Supported Exchanges :
BINANCE
BINANCE-FUTURES
BITFINEX
BITTREX
KRAKEN
BYBIT
HUOBI
KUCOIN
COINEX
        klineinterval: 
1m : OneMinute
3m : ThreeMinutes
5m : FiveMinutes
15m : FifteenMinutes
30m : ThirtyMinutes
1h : OneHour
4h : FourHour
6h : SixHour
12h : TwelveHour
1d : OneDay
1w : OneWeek
1mo : OneMonth
        lookbackperiods: Number of periods (N) to go back. Must be greater than 0.

        smaperiods: Optional. Number of periods in the moving average of ROC. Must be greater than 0, if specified.

        
    """
    url = f"https://cryptocator.p.rapidapi.com/ROC"
    querystring = {'Symbol': symbol, 'Exchange': exchange, 'KlineInterval': klineinterval, 'useHeikinCandles': useheikincandles, }
    if lookbackperiods:
        querystring['lookbackPeriods'] = lookbackperiods
    if smaperiods:
        querystring['smaPeriods'] = smaperiods
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cryptocator.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def on_balance_volume_obv(useheikincandles: bool, symbol: str, exchange: str, klineinterval: str, smaperiods: int=14, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Popularized by Joseph Granville, On-balance Volume is a rolling accumulation of volume based on Close price direction."
    symbol: eg. BTC-USDT
        exchange: Supported Exchanges :
BINANCE
BINANCE-FUTURES
BITFINEX
BITTREX
KRAKEN
BYBIT
HUOBI
KUCOIN
COINEX
        klineinterval: 
1m : OneMinute
3m : ThreeMinutes
5m : FiveMinutes
15m : FifteenMinutes
30m : ThirtyMinutes
1h : OneHour
4h : FourHour
6h : SixHour
12h : TwelveHour
1d : OneDay
1w : OneWeek
1mo : OneMonth
        smaperiods: Optional. Number of periods (N) in the moving average of OBV. Must be greater than 0, if specified.

        
    """
    url = f"https://cryptocator.p.rapidapi.com/OBV"
    querystring = {'useHeikinCandles': useheikincandles, 'Symbol': symbol, 'Exchange': exchange, 'KlineInterval': klineinterval, }
    if smaperiods:
        querystring['smaPeriods'] = smaperiods
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cryptocator.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def standard_deviation_volatility(exchange: str, symbol: str, useheikincandles: bool, klineinterval: str, lookbackperiods: int=14, smaperiods: int=7, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Standard Deviation of price over a rolling lookback window. Also known as Historical Volatility (HV). Z-Score is also returned."
    exchange: Supported Exchanges :
BINANCE
BINANCE-FUTURES
BITFINEX
BITTREX
KRAKEN
BYBIT
HUOBI
KUCOIN
COINEX
        symbol: eg. BTC-USDT
        klineinterval: 
1m : OneMinute
3m : ThreeMinutes
5m : FiveMinutes
15m : FifteenMinutes
30m : ThirtyMinutes
1h : OneHour
4h : FourHour
6h : SixHour
12h : TwelveHour
1d : OneDay
1w : OneWeek
1mo : OneMonth
        lookbackperiods: Optional. Number of periods in the moving average of StdDev. Must be greater than 0, if specified.

        
    """
    url = f"https://cryptocator.p.rapidapi.com/StandardDeviation"
    querystring = {'Exchange': exchange, 'Symbol': symbol, 'useHeikinCandles': useheikincandles, 'KlineInterval': klineinterval, }
    if lookbackperiods:
        querystring['lookbackPeriods'] = lookbackperiods
    if smaperiods:
        querystring['smaPeriods'] = smaperiods
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cryptocator.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def choppiness_index(symbol: str, exchange: str, useheikincandles: bool, klineinterval: str, lookbackperiods: int=14, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Created by E.W. Dreiss, the Choppiness Index measures the trendiness or choppiness on a scale of 0 to 100, to depict steady trends versus conditions of choppiness."
    symbol: eg. BTC-USDT
        exchange: Supported Exchanges :
BINANCE
BINANCE-FUTURES
BITFINEX
BITTREX
KRAKEN
BYBIT
HUOBI
KUCOIN
COINEX
        klineinterval: 
1m : OneMinute
3m : ThreeMinutes
5m : FiveMinutes
15m : FifteenMinutes
30m : ThirtyMinutes
1h : OneHour
4h : FourHour
6h : SixHour
12h : TwelveHour
1d : OneDay
1w : OneWeek
1mo : OneMonth
        lookbackperiods: Number of periods (N) for the lookback evaluation. Must be greater than 1. Default is 14.

        
    """
    url = f"https://cryptocator.p.rapidapi.com/ChoppinessIndex"
    querystring = {'Symbol': symbol, 'Exchange': exchange, 'useHeikinCandles': useheikincandles, 'KlineInterval': klineinterval, }
    if lookbackperiods:
        querystring['lookbackPeriods'] = lookbackperiods
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cryptocator.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def balance_of_power_bop(symbol: str, useheikincandles: bool, exchange: str, klineinterval: str, smoothperiods: int=14, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Created by Igor Levshin, the Balance of Power (aka Balance of Market Power) is a momentum oscillator that depicts the strength of buying and selling pressure."
    symbol: eg. BTC-USDT
        exchange: Supported Exchanges :
BINANCE
BINANCE-FUTURES
BITFINEX
BITTREX
KRAKEN
BYBIT
HUOBI
KUCOIN
COINEX
        klineinterval: 
1m : OneMinute
3m : ThreeMinutes
5m : FiveMinutes
15m : FifteenMinutes
30m : ThirtyMinutes
1h : OneHour
4h : FourHour
6h : SixHour
12h : TwelveHour
1d : OneDay
1w : OneWeek
1mo : OneMonth
        smoothperiods: Number of periods (N) for smoothing. Must be greater than 0. Default is 14.

        
    """
    url = f"https://cryptocator.p.rapidapi.com/BOP"
    querystring = {'Symbol': symbol, 'useHeikinCandles': useheikincandles, 'Exchange': exchange, 'KlineInterval': klineinterval, }
    if smoothperiods:
        querystring['smoothPeriods'] = smoothperiods
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cryptocator.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def average_true_range_atr(symbol: str, useheikincandles: bool, klineinterval: str, exchange: str, lookbackperiods: int=14, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Created by J. Welles Wilder, True Range and Average True Range is a measure of volatility that captures gaps and limits between periods."
    symbol: eg. BTC-USDT
        klineinterval: 
1m : OneMinute
3m : ThreeMinutes
5m : FiveMinutes
15m : FifteenMinutes
30m : ThirtyMinutes
1h : OneHour
4h : FourHour
6h : SixHour
12h : TwelveHour
1d : OneDay
1w : OneWeek
1mo : OneMonth
        exchange: Supported Exchanges :
BINANCE
BINANCE-FUTURES
BITFINEX
BITTREX
KRAKEN
BYBIT
HUOBI
KUCOIN
COINEX
        lookbackperiods: Number of periods (N) in the moving average. Must be greater than 0.

        
    """
    url = f"https://cryptocator.p.rapidapi.com/ATR"
    querystring = {'Symbol': symbol, 'useHeikinCandles': useheikincandles, 'KlineInterval': klineinterval, 'Exchange': exchange, }
    if lookbackperiods:
        querystring['lookbackPeriods'] = lookbackperiods
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cryptocator.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def zig_zag(symbol: str, klineinterval: str, useheikincandles: bool, exchange: str, endtype: str='CLOSE', percentchange: int=5, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Zig Zag is a price chart overlay that simplifies the up and down movements and transitions based on a percent change smoothing threshold."
    symbol: eg. BTC-USDT
        klineinterval: 
1m : OneMinute
3m : ThreeMinutes
5m : FiveMinutes
15m : FifteenMinutes
30m : ThirtyMinutes
1h : OneHour
4h : FourHour
6h : SixHour
12h : TwelveHour
1d : OneDay
1w : OneWeek
1mo : OneMonth
        exchange: Supported Exchanges :
BINANCE
BINANCE-FUTURES
BITFINEX
BITTREX
KRAKEN
BYBIT
HUOBI
KUCOIN
COINEX
        endtype: CLOSE : Brick change threshold measured from Close price (default)
HIGHLOW :Brick change threshold measured from High and Low price
        percentchange: Percent change required to establish a line endpoint. Example: 3.5% would be entered as 3.5 (not 0.035). Must be greater than 0. Typical values range from 3 to 10. Default is 5.
        
    """
    url = f"https://cryptocator.p.rapidapi.com/ZigZag"
    querystring = {'Symbol': symbol, 'KlineInterval': klineinterval, 'useHeikinCandles': useheikincandles, 'Exchange': exchange, }
    if endtype:
        querystring['endType'] = endtype
    if percentchange:
        querystring['percentChange'] = percentchange
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cryptocator.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def renko_chart(exchange: str, symbol: str, useheikincandles: bool, klineinterval: str, bricksize: int=0, endtype: str='CLOSE', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The Renko Chart is a Japanese price transformed candlestick pattern that uses “bricks” to show a defined increment of change over a non-linear time series. Transitions can use either Close or High/Low price values. An ATR variant is also provided where brick size is determined by current Average True Range values."
    exchange: Supported Exchanges :
BINANCE
BINANCE-FUTURES
BITFINEX
BITTREX
KRAKEN
BYBIT
HUOBI
KUCOIN
COINEX
        symbol: eg. BTC-USDT
        klineinterval: 
1m : OneMinute
3m : ThreeMinutes
5m : FiveMinutes
15m : FifteenMinutes
30m : ThirtyMinutes
1h : OneHour
4h : FourHour
6h : SixHour
12h : TwelveHour
1d : OneDay
1w : OneWeek
1mo : OneMonth
        bricksize: Brick size. Must be greater than 0.

        endtype: CLOSE : Brick change threshold measured from Close price (default)
HIGHLOW :Brick change threshold measured from High and Low price
        
    """
    url = f"https://cryptocator.p.rapidapi.com/RenkoChart"
    querystring = {'Exchange': exchange, 'Symbol': symbol, 'useHeikinCandles': useheikincandles, 'KlineInterval': klineinterval, }
    if bricksize:
        querystring['brickSize'] = bricksize
    if endtype:
        querystring['endType'] = endtype
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cryptocator.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def heikin_ashi(exchange: str, useheikincandles: bool, klineinterval: str, symbol: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Created by Munehisa Homma, Heikin-Ashi is a modified candlestick pattern based on prior period prices for smoothing."
    exchange: Supported Exchanges :
BINANCE
BINANCE-FUTURES
BITFINEX
BITTREX
KRAKEN
BYBIT
HUOBI
KUCOIN
COINEX
        klineinterval: 
1m : OneMinute
3m : ThreeMinutes
5m : FiveMinutes
15m : FifteenMinutes
30m : ThirtyMinutes
1h : OneHour
4h : FourHour
6h : SixHour
12h : TwelveHour
1d : OneDay
1w : OneWeek
1mo : OneMonth
        symbol: eg. BTC-USDT
        
    """
    url = f"https://cryptocator.p.rapidapi.com/HeikinAshi"
    querystring = {'Exchange': exchange, 'useHeikinCandles': useheikincandles, 'KlineInterval': klineinterval, 'Symbol': symbol, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cryptocator.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def ehlers_fisher_transform(exchange: str, symbol: str, useheikincandles: bool, klineinterval: str, lookbackperiods: int=10, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Created by John Ehlers, the Fisher Transform converts prices into a Gaussian normal distribution."
    exchange: Supported Exchanges :
BINANCE
BINANCE-FUTURES
BITFINEX
BITTREX
KRAKEN
BYBIT
HUOBI
KUCOIN
COINEX
        symbol: eg. BTC-USDT
        klineinterval: 
1m : OneMinute
3m : ThreeMinutes
5m : FiveMinutes
15m : FifteenMinutes
30m : ThirtyMinutes
1h : OneHour
4h : FourHour
6h : SixHour
12h : TwelveHour
1d : OneDay
1w : OneWeek
1mo : OneMonth
        lookbackperiods: Number of periods (N) in the lookback window. Must be greater than 0. Default is 10.

        
    """
    url = f"https://cryptocator.p.rapidapi.com/FisherTransform"
    querystring = {'Exchange': exchange, 'Symbol': symbol, 'useHeikinCandles': useheikincandles, 'KlineInterval': klineinterval, }
    if lookbackperiods:
        querystring['lookbackPeriods'] = lookbackperiods
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cryptocator.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def basic_quote_transforms(symbol: str, useheikincandles: bool, klineinterval: str, exchange: str, candlepart: str='OPEN', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a basic quote transform (e.g. HL2, OHL3, etc.) and isolation of individual price quote candle parts from a full OHLCV quote."
    symbol: eg. BTC-USDT
        klineinterval: 
1m : OneMinute
3m : ThreeMinutes
5m : FiveMinutes
15m : FifteenMinutes
30m : ThirtyMinutes
1h : OneHour
4h : FourHour
6h : SixHour
12h : TwelveHour
1d : OneDay
1w : OneWeek
1mo : OneMonth
        exchange: Supported Exchanges :
BINANCE
BINANCE-FUTURES
BITFINEX
BITTREX
KRAKEN
BYBIT
HUOBI
KUCOIN
COINEX
        candlepart: OPEN	:OPEN PRICE
HIGH	:HIGH PRICE
LOW	:LOW PRICE
CLOSE	:CLOSE PRICE
VOLUME	:VOLUME
HL2	:(HIGH+LOW)/2
HLC3	:(HIGH+LOW+CLOSE)/3
OC2	:(OPEN+CLOSE)/2
OHL3	:(OPEN+HIGH+LOW)/3
OHLC4	:(OPEN+HIGH+LOW+CLOSE)/4
        
    """
    url = f"https://cryptocator.p.rapidapi.com/BasicQuoteTransforms"
    querystring = {'Symbol': symbol, 'useHeikinCandles': useheikincandles, 'KlineInterval': klineinterval, 'Exchange': exchange, }
    if candlepart:
        querystring['candlePart'] = candlepart
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cryptocator.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def weighted_moving_average_wma(klineinterval: str, exchange: str, useheikincandles: bool, symbol: str, lookbackperiods: int=14, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Weighted Moving Average is the linear weighted average of price over a lookback window. This also called Linear Weighted Moving Average (LWMA)."
    klineinterval: 
1m : OneMinute
3m : ThreeMinutes
5m : FiveMinutes
15m : FifteenMinutes
30m : ThirtyMinutes
1h : OneHour
4h : FourHour
6h : SixHour
12h : TwelveHour
1d : OneDay
1w : OneWeek
1mo : OneMonth
        exchange: Supported Exchanges :
BINANCE
BINANCE-FUTURES
BITFINEX
BITTREX
KRAKEN
BYBIT
HUOBI
KUCOIN
COINEX
        symbol: eg. BTC-USDT
        lookbackperiods: Number of periods (N) in the moving average. Must be greater than 0.

        
    """
    url = f"https://cryptocator.p.rapidapi.com/WMA"
    querystring = {'KlineInterval': klineinterval, 'Exchange': exchange, 'useHeikinCandles': useheikincandles, 'Symbol': symbol, }
    if lookbackperiods:
        querystring['lookbackPeriods'] = lookbackperiods
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cryptocator.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def volume_weighted_moving_average_vwma(klineinterval: str, useheikincandles: bool, symbol: str, exchange: str, lookbackperiods: int=14, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Volume Weighted Moving Average is the volume adjusted average price over a lookback window."
    klineinterval: 
1m : OneMinute
3m : ThreeMinutes
5m : FiveMinutes
15m : FifteenMinutes
30m : ThirtyMinutes
1h : OneHour
4h : FourHour
6h : SixHour
12h : TwelveHour
1d : OneDay
1w : OneWeek
1mo : OneMonth
        symbol: eg. BTC-USDT
        exchange: Supported Exchanges :
BINANCE
BINANCE-FUTURES
BITFINEX
BITTREX
KRAKEN
BYBIT
HUOBI
KUCOIN
COINEX
        lookbackperiods: Number of periods (N) in the moving average. Must be greater than 0.

        
    """
    url = f"https://cryptocator.p.rapidapi.com/VWMA"
    querystring = {'KlineInterval': klineinterval, 'useHeikinCandles': useheikincandles, 'Symbol': symbol, 'Exchange': exchange, }
    if lookbackperiods:
        querystring['lookbackPeriods'] = lookbackperiods
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cryptocator.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def volume_weighted_average_price_vwap(useheikincandles: bool, symbol: str, exchange: str, klineinterval: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The Volume Weighted Average Price is a Volume weighted average of price, typically used on intraday data."
    symbol: eg. BTC-USDT
        exchange: Supported Exchanges :
BINANCE
BINANCE-FUTURES
BITFINEX
BITTREX
KRAKEN
BYBIT
HUOBI
KUCOIN
COINEX
        klineinterval: 
1m : OneMinute
3m : ThreeMinutes
5m : FiveMinutes
15m : FifteenMinutes
30m : ThirtyMinutes
1h : OneHour
4h : FourHour
6h : SixHour
12h : TwelveHour
1d : OneDay
1w : OneWeek
1mo : OneMonth
        
    """
    url = f"https://cryptocator.p.rapidapi.com/VWAP"
    querystring = {'useHeikinCandles': useheikincandles, 'Symbol': symbol, 'Exchange': exchange, 'KlineInterval': klineinterval, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cryptocator.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def triple_exponential_moving_average_tema(useheikincandles: bool, exchange: str, symbol: str, klineinterval: str, lookbackperiods: int=14, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Created by Patrick G. Mulloy, the Triple exponential moving average is a faster multi-smoothed EMA of the price over a lookback window."
    exchange: Supported Exchanges :
BINANCE
BINANCE-FUTURES
BITFINEX
BITTREX
KRAKEN
BYBIT
HUOBI
KUCOIN
COINEX
        symbol: eg. BTC-USDT
        klineinterval: 
1m : OneMinute
3m : ThreeMinutes
5m : FiveMinutes
15m : FifteenMinutes
30m : ThirtyMinutes
1h : OneHour
4h : FourHour
6h : SixHour
12h : TwelveHour
1d : OneDay
1w : OneWeek
1mo : OneMonth
        lookbackperiods: Number of periods (N) in the moving average. Must be greater than 0.

        
    """
    url = f"https://cryptocator.p.rapidapi.com/TEMA"
    querystring = {'useHeikinCandles': useheikincandles, 'Exchange': exchange, 'Symbol': symbol, 'KlineInterval': klineinterval, }
    if lookbackperiods:
        querystring['lookbackPeriods'] = lookbackperiods
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cryptocator.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def tillson_t3_moving_average(symbol: str, klineinterval: str, exchange: str, useheikincandles: bool, lookbackperiods: int=3, volumefactor: int=0, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Created by Tim Tillson, the T3 indicator is a smooth moving average that reduces both lag and overshooting."
    symbol: eg. BTC-USDT
        klineinterval: 
1m : OneMinute
3m : ThreeMinutes
5m : FiveMinutes
15m : FifteenMinutes
30m : ThirtyMinutes
1h : OneHour
4h : FourHour
6h : SixHour
12h : TwelveHour
1d : OneDay
1w : OneWeek
1mo : OneMonth
        exchange: Supported Exchanges :
BINANCE
BINANCE-FUTURES
BITFINEX
BITTREX
KRAKEN
BYBIT
HUOBI
KUCOIN
COINEX
        lookbackperiods: Number of periods (N) for the EMA smoothing. Must be greater than 0 and is usually less than 63. Default is 5.

        volumefactor: Size of the Volume Factor. Must be greater than 0 and is usually less than 2. Default is 0.7

        
    """
    url = f"https://cryptocator.p.rapidapi.com/T3"
    querystring = {'Symbol': symbol, 'KlineInterval': klineinterval, 'Exchange': exchange, 'useHeikinCandles': useheikincandles, }
    if lookbackperiods:
        querystring['lookbackPeriods'] = lookbackperiods
    if volumefactor:
        querystring['volumeFactor'] = volumefactor
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cryptocator.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def simple_moving_average_sma(exchange: str, symbol: str, useheikincandles: bool, klineinterval: str, lookbackperiods: int=14, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Simple Moving Average is the average price over a lookback window. An extended analysis option includes mean absolute deviation (MAD), mean square error (MSE), and mean absolute percentage error (MAPE)."
    exchange: Supported Exchanges :
BINANCE
BINANCE-FUTURES
BITFINEX
BITTREX
KRAKEN
BYBIT
HUOBI
KUCOIN
COINEX
        symbol: eg. BTC-USDT
        klineinterval: 
1m : OneMinute
3m : ThreeMinutes
5m : FiveMinutes
15m : FifteenMinutes
30m : ThirtyMinutes
1h : OneHour
4h : FourHour
6h : SixHour
12h : TwelveHour
1d : OneDay
1w : OneWeek
1mo : OneMonth
        lookbackperiods: Number of periods (N) in the moving average. Must be greater than 0.

        
    """
    url = f"https://cryptocator.p.rapidapi.com/SMA"
    querystring = {'Exchange': exchange, 'Symbol': symbol, 'useHeikinCandles': useheikincandles, 'KlineInterval': klineinterval, }
    if lookbackperiods:
        querystring['lookbackPeriods'] = lookbackperiods
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cryptocator.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def smoothed_moving_average_smma(symbol: str, exchange: str, useheikincandles: bool, klineinterval: str, lookbackperiods: int=14, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Smoothed Moving Average is the average of price over a lookback window using a smoothing method. SMMA is also known as modified moving average (MMA) and running moving average (RMA)."
    symbol: eg. BTC-USDT
        exchange: Supported Exchanges :
BINANCE
BINANCE-FUTURES
BITFINEX
BITTREX
KRAKEN
BYBIT
HUOBI
KUCOIN
COINEX
        klineinterval: 
1m : OneMinute
3m : ThreeMinutes
5m : FiveMinutes
15m : FifteenMinutes
30m : ThirtyMinutes
1h : OneHour
4h : FourHour
6h : SixHour
12h : TwelveHour
1d : OneDay
1w : OneWeek
1mo : OneMonth
        lookbackperiods: Number of periods (N) in the moving average. Must be greater than 0.

        
    """
    url = f"https://cryptocator.p.rapidapi.com/SMMA"
    querystring = {'Symbol': symbol, 'Exchange': exchange, 'useHeikinCandles': useheikincandles, 'KlineInterval': klineinterval, }
    if lookbackperiods:
        querystring['lookbackPeriods'] = lookbackperiods
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cryptocator.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def mcginley_dynamic(exchange: str, useheikincandles: bool, symbol: str, klineinterval: str, lookbackperiods: int=14, kfactor: int=0, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Created by John R. McGinley, the McGinley Dynamic is a more responsive variant of exponential moving average."
    exchange: Supported Exchanges :
BINANCE
BINANCE-FUTURES
BITFINEX
BITTREX
KRAKEN
BYBIT
HUOBI
KUCOIN
COINEX
        symbol: eg. BTC-USDT
        klineinterval: 
1m : OneMinute
3m : ThreeMinutes
5m : FiveMinutes
15m : FifteenMinutes
30m : ThirtyMinutes
1h : OneHour
4h : FourHour
6h : SixHour
12h : TwelveHour
1d : OneDay
1w : OneWeek
1mo : OneMonth
        lookbackperiods: Number of periods (N) in the moving average. Must be greater than 0.

        kfactor: Optional. Range adjustment factor (K). Must be greater than 0. Default is 0.6

        
    """
    url = f"https://cryptocator.p.rapidapi.com/McGinleyDynamic"
    querystring = {'Exchange': exchange, 'useHeikinCandles': useheikincandles, 'Symbol': symbol, 'KlineInterval': klineinterval, }
    if lookbackperiods:
        querystring['lookbackPeriods'] = lookbackperiods
    if kfactor:
        querystring['kFactor'] = kfactor
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cryptocator.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def mesa_adaptive_moving_average_mama(symbol: str, exchange: str, useheikincandles: bool, klineinterval: str, fastlimit: int=0, slowlimit: int=0, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Created by John Ehlers, the MAMA indicator is a 5-period adaptive moving average of high/low price that uses classic electrical radio-frequency signal processing algorithms to reduce noise."
    symbol: eg. BTC-USDT
        exchange: Supported Exchanges :
BINANCE
BINANCE-FUTURES
BITFINEX
BITTREX
KRAKEN
BYBIT
HUOBI
KUCOIN
COINEX
        klineinterval: 
1m : OneMinute
3m : ThreeMinutes
5m : FiveMinutes
15m : FifteenMinutes
30m : ThirtyMinutes
1h : OneHour
4h : FourHour
6h : SixHour
12h : TwelveHour
1d : OneDay
1w : OneWeek
1mo : OneMonth
        fastlimit: Fast limit threshold. Must be greater than slowLimit and less than 1. Default is 0.5.

        slowlimit: Slow limit threshold. Must be greater than 0. Default is 0.05.

        
    """
    url = f"https://cryptocator.p.rapidapi.com/MAMA"
    querystring = {'Symbol': symbol, 'Exchange': exchange, 'useHeikinCandles': useheikincandles, 'KlineInterval': klineinterval, }
    if fastlimit:
        querystring['fastLimit'] = fastlimit
    if slowlimit:
        querystring['slowLimit'] = slowlimit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cryptocator.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def kaufman_s_adaptive_moving_average_kama(symbol: str, useheikincandles: bool, exchange: str, klineinterval: str, erperiods: int=10, fastperiods: int=2, slowperiods: int=30, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Created by Perry Kaufman, KAMA is an volatility adaptive moving average of price over configurable lookback periods."
    symbol: eg. BTC-USDT
        exchange: Supported Exchanges :
BINANCE
BINANCE-FUTURES
BITFINEX
BITTREX
KRAKEN
BYBIT
HUOBI
KUCOIN
COINEX
        klineinterval: 
1m : OneMinute
3m : ThreeMinutes
5m : FiveMinutes
15m : FifteenMinutes
30m : ThirtyMinutes
1h : OneHour
4h : FourHour
6h : SixHour
12h : TwelveHour
1d : OneDay
1w : OneWeek
1mo : OneMonth
        erperiods: Number of Efficiency Ratio (volatility) periods (E). Must be greater than 0. Default is 10.
        fastperiods: Number of Fast EMA periods. Must be greater than 0. Default is 2.

        slowperiods: Number of Slow EMA periods. Must be greater than fastPeriods. Default is 30.

        
    """
    url = f"https://cryptocator.p.rapidapi.com/KAMA"
    querystring = {'Symbol': symbol, 'useHeikinCandles': useheikincandles, 'Exchange': exchange, 'KlineInterval': klineinterval, }
    if erperiods:
        querystring['erPeriods'] = erperiods
    if fastperiods:
        querystring['fastPeriods'] = fastperiods
    if slowperiods:
        querystring['slowPeriods'] = slowperiods
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cryptocator.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def hull_moving_average_hma(symbol: str, klineinterval: str, exchange: str, useheikincandles: bool, lookbackperiods: int=14, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Created by Alan Hull, the Hull Moving Average is a modified weighted average of price that reduces lag."
    symbol: eg. BTC-USDT
        klineinterval: 
1m : OneMinute
3m : ThreeMinutes
5m : FiveMinutes
15m : FifteenMinutes
30m : ThirtyMinutes
1h : OneHour
4h : FourHour
6h : SixHour
12h : TwelveHour
1d : OneDay
1w : OneWeek
1mo : OneMonth
        exchange: Supported Exchanges :
BINANCE
BINANCE-FUTURES
BITFINEX
BITTREX
KRAKEN
BYBIT
HUOBI
KUCOIN
COINEX
        lookbackperiods: Number of periods (N) in the moving average. Must be greater than 1.

        
    """
    url = f"https://cryptocator.p.rapidapi.com/HMA"
    querystring = {'Symbol': symbol, 'KlineInterval': klineinterval, 'Exchange': exchange, 'useHeikinCandles': useheikincandles, }
    if lookbackperiods:
        querystring['lookbackPeriods'] = lookbackperiods
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cryptocator.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def hilbert_transform_instantaneous_trendline(symbol: str, klineinterval: str, exchange: str, useheikincandles: bool, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Created by John Ehlers, the Hilbert Transform Instantaneous Trendline is a 5-period trendline of high/low price that that uses classic electrical radio-frequency signal processing algorithms reduce noise. Dominant Cycle Periods information is also provided."
    symbol: eg. BTC-USDT
        klineinterval: 
1m : OneMinute
3m : ThreeMinutes
5m : FiveMinutes
15m : FifteenMinutes
30m : ThirtyMinutes
1h : OneHour
4h : FourHour
6h : SixHour
12h : TwelveHour
1d : OneDay
1w : OneWeek
1mo : OneMonth
        exchange: Supported Exchanges :
BINANCE
BINANCE-FUTURES
BITFINEX
BITTREX
KRAKEN
BYBIT
HUOBI
KUCOIN
COINEX
        
    """
    url = f"https://cryptocator.p.rapidapi.com/HTL"
    querystring = {'Symbol': symbol, 'KlineInterval': klineinterval, 'Exchange': exchange, 'useHeikinCandles': useheikincandles, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cryptocator.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def exponential_moving_average_ema(klineinterval: str, exchange: str, useheikincandles: bool, symbol: str, resulttype: str='LastCandles', lookbackperiods: int=20, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Exponentially weighted moving average is a rolling moving average that puts more weight on current price."
    klineinterval: 
1m : OneMinute
3m : ThreeMinutes
5m : FiveMinutes
15m : FifteenMinutes
30m : ThirtyMinutes
1h : OneHour
4h : FourHour
6h : SixHour
12h : TwelveHour
1d : OneDay
1w : OneWeek
1mo : OneMonth
        exchange: Supported Exchanges :
BINANCE
BINANCE-FUTURES
BITFINEX
BITTREX
KRAKEN
BYBIT
HUOBI
KUCOIN
COINEX
        symbol: eg. BTC-USDT
        resulttype: Full : Returns all results
LastOpenCandle : Last opened candle
LastClosedCandle : Last Closed Candle
LastCandles : Last closed and last opened candles.
        lookbackperiods: Number of periods (N) in the moving average. Must be greater than 0.

        
    """
    url = f"https://cryptocator.p.rapidapi.com/EMA"
    querystring = {'KlineInterval': klineinterval, 'Exchange': exchange, 'useHeikinCandles': useheikincandles, 'Symbol': symbol, }
    if resulttype:
        querystring['resultType'] = resulttype
    if lookbackperiods:
        querystring['lookbackPeriods'] = lookbackperiods
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cryptocator.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def endpoint_moving_average_epma(klineinterval: str, exchange: str, symbol: str, useheikincandles: bool, lookbackperiods: int=14, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Endpoint Moving Average (EPMA), also known as Least Squares Moving Average (LSMA), plots the projected last point of a defined retrospective linear regression."
    klineinterval: 
1m : OneMinute
3m : ThreeMinutes
5m : FiveMinutes
15m : FifteenMinutes
30m : ThirtyMinutes
1h : OneHour
4h : FourHour
6h : SixHour
12h : TwelveHour
1d : OneDay
1w : OneWeek
1mo : OneMonth
        exchange: Supported Exchanges :
BINANCE
BINANCE-FUTURES
BITFINEX
BITTREX
KRAKEN
BYBIT
HUOBI
KUCOIN
COINEX
        symbol: eg. BTC-USDT
        lookbackperiods: Number of periods (N) in the moving average. Must be greater than 0.

        
    """
    url = f"https://cryptocator.p.rapidapi.com/EPMA"
    querystring = {'KlineInterval': klineinterval, 'Exchange': exchange, 'Symbol': symbol, 'useHeikinCandles': useheikincandles, }
    if lookbackperiods:
        querystring['lookbackPeriods'] = lookbackperiods
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cryptocator.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def double_exponential_moving_average_dema(useheikincandles: bool, exchange: str, symbol: str, klineinterval: str, lookbackperiods: int=14, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Created by Patrick G. Mulloy, the Double exponential moving average is a faster smoothed EMA of the price over a lookback window."
    exchange: Supported Exchanges :
BINANCE
BINANCE-FUTURES
BITFINEX
BITTREX
KRAKEN
BYBIT
HUOBI
KUCOIN
COINEX
        symbol: eg. BTC-USDT
        klineinterval: 
1m : OneMinute
3m : ThreeMinutes
5m : FiveMinutes
15m : FifteenMinutes
30m : ThirtyMinutes
1h : OneHour
4h : FourHour
6h : SixHour
12h : TwelveHour
1d : OneDay
1w : OneWeek
1mo : OneMonth
        lookbackperiods: Number of periods (N) in the moving average. Must be greater than 0.

        
    """
    url = f"https://cryptocator.p.rapidapi.com/DEMA"
    querystring = {'useHeikinCandles': useheikincandles, 'Exchange': exchange, 'Symbol': symbol, 'KlineInterval': klineinterval, }
    if lookbackperiods:
        querystring['lookbackPeriods'] = lookbackperiods
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cryptocator.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def accumulation_distribution_line_adl(klineinterval: str, symbol: str, useheikincandles: bool, exchange: str, smaperiods: int=14, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Created by Marc Chaikin, the Accumulation/Distribution Line/Index is a rolling accumulation of Chaikin Money Flow Volume."
    klineinterval: 
1m : OneMinute
3m : ThreeMinutes
5m : FiveMinutes
15m : FifteenMinutes
30m : ThirtyMinutes
1h : OneHour
4h : FourHour
6h : SixHour
12h : TwelveHour
1d : OneDay
1w : OneWeek
1mo : OneMonth
        symbol: eg. BTC-USDT
        exchange: Supported Exchanges :
BINANCE
BINANCE-FUTURES
BITFINEX
BITTREX
KRAKEN
BYBIT
HUOBI
KUCOIN
COINEX
        smaperiods: Optional. Number of periods (N) in the moving average of ADL. Must be greater than 0, if specified.

        
    """
    url = f"https://cryptocator.p.rapidapi.com/ADL"
    querystring = {'KlineInterval': klineinterval, 'Symbol': symbol, 'useHeikinCandles': useheikincandles, 'Exchange': exchange, }
    if smaperiods:
        querystring['smaPeriods'] = smaperiods
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cryptocator.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def williams_fractal(exchange: str, klineinterval: str, symbol: str, useheikincandles: bool, endtype: str='HIGHLOW', windowspan: int=2, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Created by Larry Williams, Fractal is a retrospective price pattern that identifies a central high or low point chevron."
    exchange: Supported Exchanges :
BINANCE
BINANCE-FUTURES
BITFINEX
BITTREX
KRAKEN
BYBIT
HUOBI
KUCOIN
COINEX
        klineinterval: 
1m : OneMinute
3m : ThreeMinutes
5m : FiveMinutes
15m : FifteenMinutes
30m : ThirtyMinutes
1h : OneHour
4h : FourHour
6h : SixHour
12h : TwelveHour
1d : OneDay
1w : OneWeek
1mo : OneMonth
        symbol: eg. BTC-USDT
        endtype: CLOSE : 	Chevron point identified from Close price
HIGHLOW: 	Chevron point identified from High and Low price (default)
        windowspan: Evaluation window span width (S). Must be at least 2. Default is 2.
        
    """
    url = f"https://cryptocator.p.rapidapi.com/WilliamsFractal"
    querystring = {'Exchange': exchange, 'KlineInterval': klineinterval, 'Symbol': symbol, 'useHeikinCandles': useheikincandles, }
    if endtype:
        querystring['endType'] = endtype
    if windowspan:
        querystring['windowSpan'] = windowspan
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cryptocator.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def pivots(useheikincandles: bool, klineinterval: str, exchange: str, symbol: str, leftspan: int=2, endtype: str='HIGHLOW', rightspan: int=2, maxtrendperiods: int=20, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Pivots is an extended customizable version of Williams Fractal that includes identification of Higher High, Lower Low, Higher Low, and Lower Low trends between pivots in a lookback window."
    klineinterval: 
1m : OneMinute
3m : ThreeMinutes
5m : FiveMinutes
15m : FifteenMinutes
30m : ThirtyMinutes
1h : OneHour
4h : FourHour
6h : SixHour
12h : TwelveHour
1d : OneDay
1w : OneWeek
1mo : OneMonth
        exchange: Supported Exchanges :
BINANCE
BINANCE-FUTURES
BITFINEX
BITTREX
KRAKEN
BYBIT
HUOBI
KUCOIN
COINEX
        symbol: eg. BTC-USDT
        leftspan: Left evaluation window span width (L). Must be at least 2. Default is 2.

        endtype: CLOSE : 	Chevron point identified from Close price
HIGHLOW: 	Chevron point identified from High and Low price (default)
        rightspan: Right evaluation window span width (R). Must be at least 2. Default is 2.

        maxtrendperiods: Number of periods (N) in evaluation window. Must be greater than leftSpan. Default is 20.

        
    """
    url = f"https://cryptocator.p.rapidapi.com/Pivots"
    querystring = {'useHeikinCandles': useheikincandles, 'KlineInterval': klineinterval, 'Exchange': exchange, 'Symbol': symbol, }
    if leftspan:
        querystring['leftSpan'] = leftspan
    if endtype:
        querystring['endType'] = endtype
    if rightspan:
        querystring['rightSpan'] = rightspan
    if maxtrendperiods:
        querystring['maxTrendPeriods'] = maxtrendperiods
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cryptocator.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def marubozu(exchange: str, useheikincandles: bool, klineinterval: str, symbol: str, minbodypercent: int=95, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Marubozu is a single-bar candlestick pattern that has no wicks, representing consistent directional movement."
    exchange: Supported Exchanges :
BINANCE
BINANCE-FUTURES
BITFINEX
BITTREX
KRAKEN
BYBIT
HUOBI
KUCOIN
COINEX
        klineinterval: 
1m : OneMinute
3m : ThreeMinutes
5m : FiveMinutes
15m : FifteenMinutes
30m : ThirtyMinutes
1h : OneHour
4h : FourHour
6h : SixHour
12h : TwelveHour
1d : OneDay
1w : OneWeek
1mo : OneMonth
        symbol: eg. BTC-USDT
        minbodypercent: Optional. Minimum body size as a percent of total candle size. Example: 85% would be entered as 85 (not 0.85). Must be between 80 and 100, if specified. Default is 95 (95%).
        
    """
    url = f"https://cryptocator.p.rapidapi.com/Marubozu"
    querystring = {'Exchange': exchange, 'useHeikinCandles': useheikincandles, 'KlineInterval': klineinterval, 'Symbol': symbol, }
    if minbodypercent:
        querystring['minBodyPercent'] = minbodypercent
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cryptocator.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def doji(exchange: str, symbol: str, useheikincandles: bool, klineinterval: str, maxpricechangepercent: int=0, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Doji is a single-bar candlestick pattern where open and close price are virtually identical, representing market indecision."
    exchange: Supported Exchanges :
BINANCE
BINANCE-FUTURES
BITFINEX
BITTREX
KRAKEN
BYBIT
HUOBI
KUCOIN
COINEX
        symbol: eg. BTC-USDT
        klineinterval: 
1m : OneMinute
3m : ThreeMinutes
5m : FiveMinutes
15m : FifteenMinutes
30m : ThirtyMinutes
1h : OneHour
4h : FourHour
6h : SixHour
12h : TwelveHour
1d : OneDay
1w : OneWeek
1mo : OneMonth
        maxpricechangepercent: Optional. Maximum absolute percent difference in open and close price. Example: 0.3% would be entered as 0.3 (not 0.003). Must be between 0 and 0.5 percent, if specified. Default is 0.1 (0.1%).
        
    """
    url = f"https://cryptocator.p.rapidapi.com/Doji"
    querystring = {'Exchange': exchange, 'Symbol': symbol, 'useHeikinCandles': useheikincandles, 'KlineInterval': klineinterval, }
    if maxpricechangepercent:
        querystring['maxPriceChangePercent'] = maxpricechangepercent
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cryptocator.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def volatility_stop(useheikincandles: bool, exchange: str, klineinterval: str, symbol: str, lookbackperiods: int=7, multiplier: int=3, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Created by J. Welles Wilder, Volatility Stop, also known his Volatility System, is an ATR based indicator used to determine trend direction, stops, and reversals. It is similar to Wilder’s Parabolic SAR and"
    exchange: Supported Exchanges :
BINANCE
BINANCE-FUTURES
BITFINEX
BITTREX
KRAKEN
BYBIT
HUOBI
KUCOIN
COINEX
        klineinterval: 
1m : OneMinute
3m : ThreeMinutes
5m : FiveMinutes
15m : FifteenMinutes
30m : ThirtyMinutes
1h : OneHour
4h : FourHour
6h : SixHour
12h : TwelveHour
1d : OneDay
1w : OneWeek
1mo : OneMonth
        symbol: eg. BTC-USDT
        lookbackperiods: Number of periods (N) ATR lookback window. Must be greater than 1. Default is 7.

        multiplier: ATR multiplier for the offset. Must be greater than 0. Default is 3.0.

        
    """
    url = f"https://cryptocator.p.rapidapi.com/VolatilityStop"
    querystring = {'useHeikinCandles': useheikincandles, 'Exchange': exchange, 'KlineInterval': klineinterval, 'Symbol': symbol, }
    if lookbackperiods:
        querystring['lookbackPeriods'] = lookbackperiods
    if multiplier:
        querystring['multiplier'] = multiplier
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cryptocator.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def parabolic_sar(symbol: str, useheikincandles: bool, klineinterval: str, exchange: str, accelerationstep: int=0, maxaccelerationfactor: int=0, initialfactor: int=0, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Created by J. Welles Wilder, Parabolic SAR (stop and reverse) is a price-time based indicator used to determine trend direction and reversals."
    symbol: eg.  BTC-USDT
        klineinterval: 1m : OneMinute
3m : ThreeMinutes
5m : FiveMinutes
15m : FifteenMinutes
30m : ThirtyMinutes
1h : OneHour
4h : FourHour
6h : SixHour
12h : TwelveHour
1d : OneDay
1w : OneWeek
1mo : OneMonth
        exchange: Supported Exchanges :
BINANCE
BINANCE-FUTURES
BITFINEX
BITTREX
KRAKEN
BYBIT
HUOBI
KUCOIN
COINEX
        accelerationstep: Incremental step size for the Acceleration Factor. Must be greater than 0. Default is 0.02
        maxaccelerationfactor: Maximum factor limit. Must be greater than accelerationStep. Default is 0.2

        initialfactor: Optional. Initial Acceleration Factor. Must be greater than 0. Default is accelerationStep.

        
    """
    url = f"https://cryptocator.p.rapidapi.com/ParabolicSAR"
    querystring = {'Symbol': symbol, 'useHeikinCandles': useheikincandles, 'KlineInterval': klineinterval, 'Exchange': exchange, }
    if accelerationstep:
        querystring['accelerationStep'] = accelerationstep
    if maxaccelerationfactor:
        querystring['maxAccelerationFactor'] = maxaccelerationfactor
    if initialfactor:
        querystring['initialFactor'] = initialfactor
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cryptocator.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def chandelier_exit(symbol: str, exchange: str, klineinterval: str, useheikincandles: bool, type: str='LONG', lookbackperiods: int=22, multiplier: int=3, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Created by Charles Le Beau, the Chandelier Exit is an adjusted Average True Range (ATR) offset from price that is is typically used for stop-loss and can be computed for both long or short types."
    symbol: eg.  BTC-USDT
        exchange: Supported Exchanges :
BINANCE
BINANCE-FUTURES
BITFINEX
BITTREX
KRAKEN
BYBIT
HUOBI
KUCOIN
COINEX
        klineinterval: 1m : OneMinute
3m : ThreeMinutes
5m : FiveMinutes
15m : FifteenMinutes
30m : ThirtyMinutes
1h : OneHour
4h : FourHour
6h : SixHour
12h : TwelveHour
1d : OneDay
1w : OneWeek
1mo : OneMonth
        type: LONG : Intended as stop loss value for long positions. (default)
SHORT : Intended as stop loss value for short positions.
        lookbackperiods: Number of periods (N) for the lookback evaluation. Default is 22.

        multiplier: Multiplier number must be a positive value. Default is 3.

        
    """
    url = f"https://cryptocator.p.rapidapi.com/ChandelierExit"
    querystring = {'Symbol': symbol, 'Exchange': exchange, 'KlineInterval': klineinterval, 'useHeikinCandles': useheikincandles, }
    if type:
        querystring['type'] = type
    if lookbackperiods:
        querystring['lookbackPeriods'] = lookbackperiods
    if multiplier:
        querystring['multiplier'] = multiplier
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cryptocator.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def williams_r(exchange: str, useheikincandles: bool, klineinterval: str, symbol: str, lookbackperiods: int=14, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Created by Larry Williams, the Williams %R momentum oscillator compares current price with recent highs and lows and is presented on scale of -100 to 0. It is exactly the same as the fast variant of Stochastic Oscillator, but with a different scaling."
    exchange: Supported Exchanges :
BINANCE
BINANCE-FUTURES
BITFINEX
BITTREX
KRAKEN
BYBIT
HUOBI
KUCOIN
COINEX
        klineinterval: 
1m : OneMinute
3m : ThreeMinutes
5m : FiveMinutes
15m : FifteenMinutes
30m : ThirtyMinutes
1h : OneHour
4h : FourHour
6h : SixHour
12h : TwelveHour
1d : OneDay
1w : OneWeek
1mo : OneMonth
        symbol: eg.  BTC-USDT
        lookbackperiods: Number of periods (N) in the lookback period. Must be greater than 0. Default is 14.

        
    """
    url = f"https://cryptocator.p.rapidapi.com/WilliamsR"
    querystring = {'Exchange': exchange, 'useHeikinCandles': useheikincandles, 'KlineInterval': klineinterval, 'Symbol': symbol, }
    if lookbackperiods:
        querystring['lookbackPeriods'] = lookbackperiods
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cryptocator.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def ultimate_oscillator(symbol: str, useheikincandles: bool, klineinterval: str, exchange: str, shortperiods: int=7, middleperiods: int=14, longperiods: int=28, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Created by Larry Williams, the Ultimate Oscillator uses several moving averages to weigh buying power against true range price to produce on oversold / overbought oscillator."
    symbol: eg.  BTC-USDT
        klineinterval: 
1m : OneMinute
3m : ThreeMinutes
5m : FiveMinutes
15m : FifteenMinutes
30m : ThirtyMinutes
1h : OneHour
4h : FourHour
6h : SixHour
12h : TwelveHour
1d : OneDay
1w : OneWeek
1mo : OneMonth
        exchange: Supported Exchanges :
BINANCE
BINANCE-FUTURES
BITFINEX
BITTREX
KRAKEN
BYBIT
HUOBI
KUCOIN
COINEX
        shortperiods: Number of periods (S) in the short lookback. Must be greater than 0. Default is 7.
        middleperiods: Number of periods (M) in the middle lookback. Must be greater than S. Default is 14.
        longperiods: Number of periods (L) in the long lookback. Must be greater than M. Default is 28.
        
    """
    url = f"https://cryptocator.p.rapidapi.com/UltimateOscillator"
    querystring = {'Symbol': symbol, 'useHeikinCandles': useheikincandles, 'KlineInterval': klineinterval, 'Exchange': exchange, }
    if shortperiods:
        querystring['shortPeriods'] = shortperiods
    if middleperiods:
        querystring['middlePeriods'] = middleperiods
    if longperiods:
        querystring['longPeriods'] = longperiods
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cryptocator.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def triple_ema_oscillator_trix(useheikincandles: bool, klineinterval: str, exchange: str, symbol: str, signalperiods: int=3, lookbackperiods: int=14, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Created by Jack Hutson, TRIX is the rate of change for a 3 EMA smoothing of the price over a lookback window. TRIX is often confused with TEMA."
    klineinterval: 
1m : OneMinute
3m : ThreeMinutes
5m : FiveMinutes
15m : FifteenMinutes
30m : ThirtyMinutes
1h : OneHour
4h : FourHour
6h : SixHour
12h : TwelveHour
1d : OneDay
1w : OneWeek
1mo : OneMonth
        exchange: Supported Exchanges :
BINANCE
BINANCE-FUTURES
BITFINEX
BITTREX
KRAKEN
BYBIT
HUOBI
KUCOIN
COINEX
        symbol: eg.  BTC-USDT
        signalperiods: Optional. Number of periods in the moving average of TRIX. Must be greater than 0, if specified.

        lookbackperiods: Number of periods (N) in each of the the exponential moving averages. Must be greater than 0.

        
    """
    url = f"https://cryptocator.p.rapidapi.com/TRIX"
    querystring = {'useHeikinCandles': useheikincandles, 'KlineInterval': klineinterval, 'Exchange': exchange, 'Symbol': symbol, }
    if signalperiods:
        querystring['signalPeriods'] = signalperiods
    if lookbackperiods:
        querystring['lookbackPeriods'] = lookbackperiods
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cryptocator.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def stochastic_rsi(useheikincandles: bool, klineinterval: str, symbol: str, exchange: str, rsiperiods: int=14, stochperiods: int=14, smoothperiods: int=1, signalperiods: int=3, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Created by by Tushar Chande and Stanley Kroll, Stochastic RSI is a Stochastic interpretation of the Relative Strength Index. It is different from, and often confused with the more traditional Stochastic"
    klineinterval: 
1m : OneMinute
3m : ThreeMinutes
5m : FiveMinutes
15m : FifteenMinutes
30m : ThirtyMinutes
1h : OneHour
4h : FourHour
6h : SixHour
12h : TwelveHour
1d : OneDay
1w : OneWeek
1mo : OneMonth
        symbol: eg.  BTC-USDT
        exchange: Supported Exchanges :
BINANCE
BINANCE-FUTURES
BITFINEX
BITTREX
KRAKEN
BYBIT
HUOBI
KUCOIN
COINEX
        rsiperiods: Number of periods (R) in the lookback period. Must be greater than 0. Standard is 14.
        stochperiods: Number of periods (S) in the lookback period. Must be greater than 0. Typically the same value as rsiPeriods.
        smoothperiods: Smoothing periods (M) for the Stochastic. Must be greater than 0. Default is 1 (Fast variant).
        signalperiods: Number of periods (G) in the signal line (SMA of the StochRSI). Must be greater than 0. Typically 3-5.
        
    """
    url = f"https://cryptocator.p.rapidapi.com/StochasticRSI"
    querystring = {'useHeikinCandles': useheikincandles, 'KlineInterval': klineinterval, 'Symbol': symbol, 'Exchange': exchange, }
    if rsiperiods:
        querystring['rsiPeriods'] = rsiperiods
    if stochperiods:
        querystring['stochPeriods'] = stochperiods
    if smoothperiods:
        querystring['smoothPeriods'] = smoothperiods
    if signalperiods:
        querystring['signalPeriods'] = signalperiods
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cryptocator.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def stochastic_oscillator(exchange: str, symbol: str, useheikincandles: bool, klineinterval: str, lookbackperiods: int=14, signalperiods: int=3, kfactor: int=3, dfactor: int=2, movingaveragetype: str='SMA', smoothperiods: int=3, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Created by George Lane, the Stochastic Oscillator, also known as KDJ Index, is a momentum oscillator that compares current price with recent highs and lows and is presented on a scale of 0 to 100."
    exchange: Supported Exchanges :
BINANCE
BINANCE-FUTURES
BITFINEX
BITTREX
KRAKEN
BYBIT
HUOBI
KUCOIN
COINEX
        symbol: eg.  BTC-USDT
        klineinterval: 
1m : OneMinute
3m : ThreeMinutes
5m : FiveMinutes
15m : FifteenMinutes
30m : ThirtyMinutes
1h : OneHour
4h : FourHour
6h : SixHour
12h : TwelveHour
1d : OneDay
1w : OneWeek
1mo : OneMonth
        lookbackperiods: Lookback period (N) for the oscillator (%K). Must be greater than 0. Default is 14.
        signalperiods: Smoothing period for the signal (%D). Must be greater than 0. Default is 3.
        kfactor: Optional. Weight of %K in the %J calculation. Must be greater than 0. Default is 3.
        dfactor: Optional. Weight of %D in the %J calculation. Must be greater than 0. Default is 2.
        movingaveragetype: Type of moving average (e.g. SMA, EMA, HMA).

Supported Types:
ALMA Arnaud Legoux Moving Average
DEMA Double Exponential Moving Average
EPMA Endpoint Moving Average
EMA Exponential Moving Average
HMA Hull Moving Average
SMA Simple Moving Average (default)
SMMA Smoothed Moving Average
TEMA Triple Exponential Moving Average
WMA Weighted Moving Average
        smoothperiods: Smoothing period (S) for the Oscillator (%K). “Slow” stochastic uses 3, “Fast” stochastic uses 1. Must be greater than 0. Default is 3.
        
    """
    url = f"https://cryptocator.p.rapidapi.com/StochasticOscillator"
    querystring = {'Exchange': exchange, 'Symbol': symbol, 'useHeikinCandles': useheikincandles, 'KlineInterval': klineinterval, }
    if lookbackperiods:
        querystring['lookbackPeriods'] = lookbackperiods
    if signalperiods:
        querystring['signalPeriods'] = signalperiods
    if kfactor:
        querystring['kFactor'] = kfactor
    if dfactor:
        querystring['dFactor'] = dfactor
    if movingaveragetype:
        querystring['movingAverageType'] = movingaveragetype
    if smoothperiods:
        querystring['smoothPeriods'] = smoothperiods
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cryptocator.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def stochastic_momentum_index_smi(klineinterval: str, exchange: str, symbol: str, useheikincandles: bool, firstsmoothperiods: int=14, secondsmoothperiods: int=7, lookbackperiods: int=14, signalperiods: int=3, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Created by William Blau, the Stochastic Momentum Index (SMI) oscillator is a double-smoothed variant of the Stochastic Oscillator, depicted on a scale from -100 to 100."
    klineinterval: 
1m : OneMinute
3m : ThreeMinutes
5m : FiveMinutes
15m : FifteenMinutes
30m : ThirtyMinutes
1h : OneHour
4h : FourHour
6h : SixHour
12h : TwelveHour
1d : OneDay
1w : OneWeek
1mo : OneMonth
        exchange: Supported Exchanges :
BINANCE
BINANCE-FUTURES
BITFINEX
BITTREX
KRAKEN
BYBIT
HUOBI
KUCOIN
COINEX
        symbol: eg.  BTC-USDT
        firstsmoothperiods: First smoothing factor lookback. Must be greater than 0.

        secondsmoothperiods: Second smoothing factor lookback. Must be greater than 0.

        lookbackperiods: Lookback period (N) for the stochastic. Must be greater than 0.

        signalperiods: EMA of SMI lookback periods. Must be greater than 0. Default is 3.

        
    """
    url = f"https://cryptocator.p.rapidapi.com/SMI"
    querystring = {'KlineInterval': klineinterval, 'Exchange': exchange, 'Symbol': symbol, 'useHeikinCandles': useheikincandles, }
    if firstsmoothperiods:
        querystring['firstSmoothPeriods'] = firstsmoothperiods
    if secondsmoothperiods:
        querystring['secondSmoothPeriods'] = secondsmoothperiods
    if lookbackperiods:
        querystring['lookbackPeriods'] = lookbackperiods
    if signalperiods:
        querystring['signalPeriods'] = signalperiods
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cryptocator.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def schaff_trend_cycle(klineinterval: str, exchange: str, useheikincandles: bool, symbol: str, slowperiods: int=50, cycleperiods: int=10, fastperiods: int=23, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Created by Doug Schaff, the Schaff Trend Cycle is a stochastic oscillator view of two converging/diverging exponential moving averages. In other words, it’s a Stochastic Oscillator of Moving Average Convergence / Divergence (MACD)."
    klineinterval: 
1m : OneMinute
3m : ThreeMinutes
5m : FiveMinutes
15m : FifteenMinutes
30m : ThirtyMinutes
1h : OneHour
4h : FourHour
6h : SixHour
12h : TwelveHour
1d : OneDay
1w : OneWeek
1mo : OneMonth
        exchange: Supported Exchanges :
BINANCE
BINANCE-FUTURES
BITFINEX
BITTREX
KRAKEN
BYBIT
HUOBI
KUCOIN
COINEX
        symbol: eg.  BTC-USDT
        slowperiods: Number of periods (S) for the slower moving average. Must be greater than fastPeriods. Default is 50.
        cycleperiods: Number of periods (C) for the Trend Cycle. Must be greater than or equal to 0. Default is 10.
        fastperiods: Number of periods (F) for the faster moving average. Must be greater than 0. Default is 23.
        
    """
    url = f"https://cryptocator.p.rapidapi.com/SchaffTrendCycle"
    querystring = {'KlineInterval': klineinterval, 'Exchange': exchange, 'useHeikinCandles': useheikincandles, 'Symbol': symbol, }
    if slowperiods:
        querystring['slowPeriods'] = slowperiods
    if cycleperiods:
        querystring['cyclePeriods'] = cycleperiods
    if fastperiods:
        querystring['fastPeriods'] = fastperiods
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cryptocator.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def relative_strength_index_rsi(useheikincandles: bool, exchange: str, symbol: str, klineinterval: str, lookbackperiods: int=14, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Created by J. Welles Wilder, the Relative Strength Index is an oscillator that measures strength of the winning/losing streak over N lookback periods on a scale of 0 to 100, to depict overbought and oversold conditions."
    exchange: Supported Exchanges :
BINANCE
BINANCE-FUTURES
BITFINEX
BITTREX
KRAKEN
BYBIT
HUOBI
KUCOIN
COINEX
        symbol: eg.  BTC-USDT
        klineinterval: 
1m : OneMinute
3m : ThreeMinutes
5m : FiveMinutes
15m : FifteenMinutes
30m : ThirtyMinutes
1h : OneHour
4h : FourHour
6h : SixHour
12h : TwelveHour
1d : OneDay
1w : OneWeek
1mo : OneMonth
        lookbackperiods: Number of periods (N) in the lookback period. Must be greater than 0. Default is 14.

        
    """
    url = f"https://cryptocator.p.rapidapi.com/RSI"
    querystring = {'useHeikinCandles': useheikincandles, 'Exchange': exchange, 'Symbol': symbol, 'KlineInterval': klineinterval, }
    if lookbackperiods:
        querystring['lookbackPeriods'] = lookbackperiods
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cryptocator.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def kdj_index(klineinterval: str, symbol: str, exchange: str, useheikincandles: bool, kfactor: int=3, lookbackperiods: int=14, smoothperiods: int=3, signalperiods: int=3, dfactor: int=2, movingaveragetype: str='SMA', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Created by George Lane, the Stochastic Oscillator, also known as KDJ Index, is a momentum oscillator that compares current price with recent highs and lows and is presented on a scale of 0 to 100."
    klineinterval: 
1m : OneMinute
3m : ThreeMinutes
5m : FiveMinutes
15m : FifteenMinutes
30m : ThirtyMinutes
1h : OneHour
4h : FourHour
6h : SixHour
12h : TwelveHour
1d : OneDay
1w : OneWeek
1mo : OneMonth
        symbol: eg.  BTC-USDT
        exchange: Supported Exchanges :
BINANCE
BINANCE-FUTURES
BITFINEX
BITTREX
KRAKEN
BYBIT
HUOBI
KUCOIN
COINEX
        kfactor: Optional. Weight of %K in the %J calculation. Must be greater than 0. Default is 3.
        lookbackperiods: Lookback period (N) for the oscillator (%K). Must be greater than 0. Default is 14.
        smoothperiods: Smoothing period (S) for the Oscillator (%K). “Slow” stochastic uses 3, “Fast” stochastic uses 1. Must be greater than 0. Default is 3.
        signalperiods: Smoothing period for the signal (%D). Must be greater than 0. Default is 3.
        dfactor: Optional. Weight of %D in the %J calculation. Must be greater than 0. Default is 2.
        movingaveragetype: Type of moving average (e.g. SMA, EMA, HMA).

Supported Types:
ALMA Arnaud Legoux Moving Average
DEMA Double Exponential Moving Average
EPMA Endpoint Moving Average
EMA Exponential Moving Average
HMA Hull Moving Average
SMA Simple Moving Average (default)
SMMA Smoothed Moving Average
TEMA Triple Exponential Moving Average
WMA Weighted Moving Average
        
    """
    url = f"https://cryptocator.p.rapidapi.com/KDJ"
    querystring = {'KlineInterval': klineinterval, 'Symbol': symbol, 'Exchange': exchange, 'useHeikinCandles': useheikincandles, }
    if kfactor:
        querystring['kFactor'] = kfactor
    if lookbackperiods:
        querystring['lookbackPeriods'] = lookbackperiods
    if smoothperiods:
        querystring['smoothPeriods'] = smoothperiods
    if signalperiods:
        querystring['signalPeriods'] = signalperiods
    if dfactor:
        querystring['dFactor'] = dfactor
    if movingaveragetype:
        querystring['movingAverageType'] = movingaveragetype
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cryptocator.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def detrended_price_oscillator_dpo(useheikincandles: bool, exchange: str, symbol: str, klineinterval: str, lookbackperiods: int=14, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Detrended Price Oscillator depicts the difference between price and an offset simple moving average. It is used to identify trend cycles and duration."
    exchange: Supported Exchanges :
BINANCE
BINANCE-FUTURES
BITFINEX
BITTREX
KRAKEN
BYBIT
HUOBI
KUCOIN
COINEX
        symbol: eg.  BTC-USDT
        klineinterval: 
1m : OneMinute
3m : ThreeMinutes
5m : FiveMinutes
15m : FifteenMinutes
30m : ThirtyMinutes
1h : OneHour
4h : FourHour
6h : SixHour
12h : TwelveHour
1d : OneDay
1w : OneWeek
1mo : OneMonth
        lookbackperiods: Number of periods (N) in the moving average. Must be greater than 0.
        
    """
    url = f"https://cryptocator.p.rapidapi.com/DPO"
    querystring = {'useHeikinCandles': useheikincandles, 'Exchange': exchange, 'Symbol': symbol, 'KlineInterval': klineinterval, }
    if lookbackperiods:
        querystring['lookbackPeriods'] = lookbackperiods
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cryptocator.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def connorsrsi(exchange: str, useheikincandles: bool, klineinterval: str, symbol: str, rsiperiods: int=3, rankperiods: int=100, streakperiods: int=2, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Created by Laurence Connors, the ConnorsRSI is a composite oscillator that incorporates RSI, winning/losing streaks, and percentile gain metrics on scale of 0 to 100."
    exchange: Supported Exchanges :
BINANCE
BINANCE-FUTURES
BITFINEX
BITTREX
KRAKEN
BYBIT
HUOBI
KUCOIN
COINEX
        klineinterval: 
1m : OneMinute
3m : ThreeMinutes
5m : FiveMinutes
15m : FifteenMinutes
30m : ThirtyMinutes
1h : OneHour
4h : FourHour
6h : SixHour
12h : TwelveHour
1d : OneDay
1w : OneWeek
1mo : OneMonth
        symbol: eg.  BTC-USDT
        rsiperiods: Lookback period (R) for the price RSI. Must be greater than 1. Default is 3.

        rankperiods: Lookback period (P) for the Percentile Rank. Must be greater than 1. Default is 100.

        streakperiods: Lookback period (S) for the streak RSI. Must be greater than 1. Default is 2.

        
    """
    url = f"https://cryptocator.p.rapidapi.com/ConnorsRSI"
    querystring = {'Exchange': exchange, 'useHeikinCandles': useheikincandles, 'KlineInterval': klineinterval, 'Symbol': symbol, }
    if rsiperiods:
        querystring['rsiPeriods'] = rsiperiods
    if rankperiods:
        querystring['rankPeriods'] = rankperiods
    if streakperiods:
        querystring['streakPeriods'] = streakperiods
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cryptocator.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def commodity_channel_index_cci(exchange: str, useheikincandles: bool, klineinterval: str, symbol: str, lookbackperiods: int=20, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Created by Donald Lambert, the Commodity Channel Index is an oscillator depicting deviation from typical price range, often used to identify cyclical trends."
    exchange: Supported Exchanges :
BINANCE
BINANCE-FUTURES
BITFINEX
BITTREX
KRAKEN
BYBIT
HUOBI
KUCOIN
COINEX
        klineinterval: 
1m : OneMinute
3m : ThreeMinutes
5m : FiveMinutes
15m : FifteenMinutes
30m : ThirtyMinutes
1h : OneHour
4h : FourHour
6h : SixHour
12h : TwelveHour
1d : OneDay
1w : OneWeek
1mo : OneMonth
        symbol: eg.  BTC-USDT
        lookbackperiods: Number of periods (N) in the moving average. Must be greater than 0. Default is 20.
        
    """
    url = f"https://cryptocator.p.rapidapi.com/CCI"
    querystring = {'Exchange': exchange, 'useHeikinCandles': useheikincandles, 'KlineInterval': klineinterval, 'Symbol': symbol, }
    if lookbackperiods:
        querystring['lookbackPeriods'] = lookbackperiods
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cryptocator.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def chande_momentum_oscillator_cmo(exchange: str, useheikincandles: bool, symbol: str, klineinterval: str, lookbackperiods: int=14, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Created by Tushar Chande, the Chande Momentum Oscillator is a weighted percent of higher prices over a lookback window."
    exchange: Supported Exchanges :
BINANCE
BINANCE-FUTURES
BITFINEX
BITTREX
KRAKEN
BYBIT
HUOBI
KUCOIN
COINEX
        symbol: eg.  BTC-USDT
        klineinterval: 
1m : OneMinute
3m : ThreeMinutes
5m : FiveMinutes
15m : FifteenMinutes
30m : ThirtyMinutes
1h : OneHour
4h : FourHour
6h : SixHour
12h : TwelveHour
1d : OneDay
1w : OneWeek
1mo : OneMonth
        lookbackperiods: Number of periods (N) in the lookback window. Must be greater than 0.
        
    """
    url = f"https://cryptocator.p.rapidapi.com/CMO"
    querystring = {'Exchange': exchange, 'useHeikinCandles': useheikincandles, 'Symbol': symbol, 'KlineInterval': klineinterval, }
    if lookbackperiods:
        querystring['lookbackPeriods'] = lookbackperiods
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cryptocator.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def awesome_oscillator_ao(exchange: str, symbol: str, klineinterval: str, useheikincandles: bool, slowperiods: int=34, fastperiods: int=5, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Created by Bill Williams, the Awesome Oscillator (aka Super AO) is a measure of the gap between a fast and slow period modified moving average."
    exchange: Supported Exchanges :
BINANCE
BINANCE-FUTURES
BITFINEX
BITTREX
KRAKEN
BYBIT
HUOBI
KUCOIN
COINEX
        symbol: eg.  BTC-USDT
        klineinterval: 
1m : OneMinute
3m : ThreeMinutes
5m : FiveMinutes
15m : FifteenMinutes
30m : ThirtyMinutes
1h : OneHour
4h : FourHour
6h : SixHour
12h : TwelveHour
1d : OneDay
1w : OneWeek
1mo : OneMonth
        slowperiods: Number of periods (S) for the slower moving average. Must be greater than fastPeriods. Default is 34.
        fastperiods: Number of periods (F) for the faster moving average. Must be greater than 0. Default is 5.

        
    """
    url = f"https://cryptocator.p.rapidapi.com/AwesomeOscillator"
    querystring = {'Exchange': exchange, 'Symbol': symbol, 'KlineInterval': klineinterval, 'useHeikinCandles': useheikincandles, }
    if slowperiods:
        querystring['slowPeriods'] = slowperiods
    if fastperiods:
        querystring['fastPeriods'] = fastperiods
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cryptocator.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def williams_alligator(exchange: str, symbol: str, klineinterval: str, useheikincandles: bool, jawperiods: int=13, jawoffset: int=8, lipsoffset: int=3, teethoffset: int=5, teethperiods: int=8, resulttype: str='LastCandles', lipsperiods: int=5, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Created by Bill Williams, Alligator is a depiction of three smoothed moving averages of median price, showing chart patterns that compared to an alligator’s feeding habits when describing market movement. The moving averages are known as the Jaw, Teeth, and Lips, which are calculated using lookback and offset periods."
    exchange: 
Supported Exchanges :
BINANCE
BINANCE-FUTURES
BITFINEX
BITTREX
KRAKEN
BYBIT
HUOBI
KUCOIN
COINEX
        symbol: eg. BTC-USDT
        klineinterval: 
1m : OneMinute
3m : ThreeMinutes
5m : FiveMinutes
15m : FifteenMinutes
30m : ThirtyMinutes
1h : OneHour
4h : FourHour
6h : SixHour
12h : TwelveHour
1d : OneDay
1w : OneWeek
1mo : OneMonth
        jawperiods: Number of periods (JP) for the Jaw moving average. Must be greater than teethPeriods. Default is 13.
        jawoffset: Number of periods (JO) for the Jaw offset. Must be greater than 0. Default is 8.
        lipsoffset: Number of periods (LO) for the Lips offset. Must be greater than 0. Default is 3.
        teethoffset: Number of periods (TO) for the Teeth offset. Must be greater than 0. Default is 5.
        teethperiods: Number of periods (TP) for the Teeth moving average. Must be greater than lipsPeriods. Default is 8.
        resulttype: Full : Returns all results
LastOpenCandle : Last opened candle
LastClosedCandle : Last Closed Candle
LastCandles : Last closed and last opened candles.
        lipsperiods: Number of periods (LP) for the Lips moving average. Must be greater than 0. Default is 5.
        
    """
    url = f"https://cryptocator.p.rapidapi.com/WilliamsAlligator"
    querystring = {'Exchange': exchange, 'Symbol': symbol, 'KlineInterval': klineinterval, 'useHeikinCandles': useheikincandles, }
    if jawperiods:
        querystring['jawPeriods'] = jawperiods
    if jawoffset:
        querystring['jawOffset'] = jawoffset
    if lipsoffset:
        querystring['lipsOffset'] = lipsoffset
    if teethoffset:
        querystring['teethOffset'] = teethoffset
    if teethperiods:
        querystring['teethPeriods'] = teethperiods
    if resulttype:
        querystring['resultType'] = resulttype
    if lipsperiods:
        querystring['lipsPeriods'] = lipsperiods
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cryptocator.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def vortex_indicator_vi(symbol: str, klineinterval: str, useheikincandles: bool, exchange: str, lookbackperiods: int=14, resulttype: str='LastCandles', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Created by Etienne Botes and Douglas Siepman, the Vortex Indicator is a measure of price directional movement. It includes positive and negative indicators, and is often used to identify trends and reversals."
    symbol: eg.  BTC-USDT
        klineinterval: 1m : OneMinute
3m : ThreeMinutes
5m : FiveMinutes
15m : FifteenMinutes
30m : ThirtyMinutes
1h : OneHour
4h : FourHour
6h : SixHour
12h : TwelveHour
1d : OneDay
1w : OneWeek
1mo : OneMonth
        exchange: Supported Exchanges :
BINANCE
BINANCE-FUTURES
BITFINEX
BITTREX
KRAKEN
BYBIT
HUOBI
KUCOIN
COINEX
        lookbackperiods: Number of periods (N) to consider. Must be greater than 1 and is usually between 14 and 30.
        resulttype: Full : Returns all results
LastOpenCandle : Last opened candle
LastClosedCandle : Last Closed Candle
LastCandles : Last closed and last opened candles.
        
    """
    url = f"https://cryptocator.p.rapidapi.com/VortexIndicator"
    querystring = {'Symbol': symbol, 'KlineInterval': klineinterval, 'useHeikinCandles': useheikincandles, 'Exchange': exchange, }
    if lookbackperiods:
        querystring['lookbackPeriods'] = lookbackperiods
    if resulttype:
        querystring['resultType'] = resulttype
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cryptocator.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def supertrend(exchange: str, symbol: str, useheikincandles: bool, klineinterval: str, lookbackperiods: int=10, multiplier: int=3, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Created by Oliver Seban, the SuperTrend indicator attempts to determine the primary trend of prices by using Average True Range (ATR) band thresholds around an HL2 midline. It can indicate a buy/sell signal or a trailing stop when the trend changes."
    exchange: Supported Exchanges :
BINANCE
BINANCE-FUTURES
BITFINEX
BITTREX
KRAKEN
BYBIT
HUOBI
KUCOIN
COINEX
        symbol: eg. BTC-USDT
        klineinterval: 
1m : OneMinute
3m : ThreeMinutes
5m : FiveMinutes
15m : FifteenMinutes
30m : ThirtyMinutes
1h : OneHour
4h : FourHour
6h : SixHour
12h : TwelveHour
1d : OneDay
1w : OneWeek
1mo : OneMonth
        lookbackperiods: Number of periods (N) for the ATR evaluation. Must be greater than 1 and is usually set between 7 and 14. Default is 10.
        multiplier: Multiplier sets the ATR band width. Must be greater than 0 and is usually set around 2 to 3. Default is 3.
        
    """
    url = f"https://cryptocator.p.rapidapi.com/SuperTrend"
    querystring = {'Exchange': exchange, 'Symbol': symbol, 'useHeikinCandles': useheikincandles, 'KlineInterval': klineinterval, }
    if lookbackperiods:
        querystring['lookbackPeriods'] = lookbackperiods
    if multiplier:
        querystring['multiplier'] = multiplier
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cryptocator.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def macd(symbol: str, exchange: str, klineinterval: str, useheikincandles: bool, slowperiods: int=26, fastperiods: int=12, signalperiods: int=9, resulttype: str='LastCandles', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Created by Gerald Appel, MACD is a simple oscillator view of two converging / diverging exponential moving averages and their differences."
    symbol: eg. BTC-USDT
        exchange: Supported Exchanges :
BINANCE
BINANCE-FUTURES
BITFINEX
BITTREX
KRAKEN
BYBIT
HUOBI
KUCOIN
COINEX
        klineinterval: 1m : OneMinute
3m : ThreeMinutes
5m : FiveMinutes
15m : FifteenMinutes
30m : ThirtyMinutes
1h : OneHour
4h : FourHour
6h : SixHour
12h : TwelveHour
1d : OneDay
1w : OneWeek
1mo : OneMonth
        slowperiods: Number of periods (S) for the slower moving average. Must be greater than fastPeriods. Default is 26.
        fastperiods: Number of periods (F) for the faster moving average. Must be greater than 0. Default is 12.
        signalperiods: Number of periods (P) for the moving average of MACD. Must be greater than or equal to 0. Default is 9.
        resulttype: Full : Returns all results
LastOpenCandle : Last opened candle
LastClosedCandle : Last Closed Candle
LastCandles : Last closed and last opened candles.
        
    """
    url = f"https://cryptocator.p.rapidapi.com/Macd"
    querystring = {'Symbol': symbol, 'Exchange': exchange, 'KlineInterval': klineinterval, 'useHeikinCandles': useheikincandles, }
    if slowperiods:
        querystring['slowPeriods'] = slowperiods
    if fastperiods:
        querystring['fastPeriods'] = fastperiods
    if signalperiods:
        querystring['signalPeriods'] = signalperiods
    if resulttype:
        querystring['resultType'] = resulttype
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cryptocator.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def ichimoku_cloud(symbol: str, exchange: str, klineinterval: str, useheikincandles: bool, kijunperiods: int=26, senkoubperiods: int=52, tenkanperiods: int=9, resulttype: str='LastCandles', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Created by Goichi Hosoda (細田悟一, Hosoda Goichi), Ichimoku Cloud, also known as Ichimoku Kinkō Hyō, is a collection of indicators that depict support and resistance, momentum, and trend direction."
    symbol: eg. BTC-USDT
        exchange: Supported Exchanges :
BINANCE
BINANCE-FUTURES
BITFINEX
BITTREX
KRAKEN
BYBIT
HUOBI
KUCOIN
COINEX
        klineinterval: 1m : OneMinute
3m : ThreeMinutes
5m : FiveMinutes
15m : FifteenMinutes
30m : ThirtyMinutes
1h : OneHour
4h : FourHour
6h : SixHour
12h : TwelveHour
1d : OneDay
1w : OneWeek
1mo : OneMonth
        kijunperiods: Number of periods (K) in the shorter Kijun-sen midpoint evaluation. Must be greater than 0. Default is 26.
        senkoubperiods: Number of periods (S) in the longer Senkou leading span B midpoint evaluation. Must be greater than K. Default is 52.
        tenkanperiods: Number of periods (T) in the Tenkan-sen midpoint evaluation. Must be greater than 0. Default is 9.
        resulttype: Full : Returns all results
LastOpenCandle : Last opened candle
LastClosedCandle : Last Closed Candle
LastCandles : Last closed and last opened candles.
        
    """
    url = f"https://cryptocator.p.rapidapi.com/IchimokuCloud"
    querystring = {'Symbol': symbol, 'Exchange': exchange, 'KlineInterval': klineinterval, 'useHeikinCandles': useheikincandles, }
    if kijunperiods:
        querystring['kijunPeriods'] = kijunperiods
    if senkoubperiods:
        querystring['SenkouBPeriods'] = senkoubperiods
    if tenkanperiods:
        querystring['tenkanPeriods'] = tenkanperiods
    if resulttype:
        querystring['resultType'] = resulttype
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cryptocator.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def elder_ray_index(exchange: str, symbol: str, klineinterval: str, useheikincandles: bool, resulttype: str='LastCandles', lookbackperiods: int=13, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Created by Alexander Elder, the Elder-ray Index, also known as Bull and Bear Power, is an oscillator that depicts buying and selling pressure. It compares current high/low prices against an Exponential Moving Average."
    exchange: Supported Exchanges :
BINANCE
BINANCE-FUTURES
BITFINEX
BITTREX
KRAKEN
BYBIT
HUOBI
KUCOIN
COINEX
        symbol: eg.  BTC-USDT
        klineinterval: 
1m : OneMinute
3m : ThreeMinutes
5m : FiveMinutes
15m : FifteenMinutes
30m : ThirtyMinutes
1h : OneHour
4h : FourHour
6h : SixHour
12h : TwelveHour
1d : OneDay
1w : OneWeek
1mo : OneMonth
        resulttype: Full : Returns all results
LastOpenCandle : Last opened candle
LastClosedCandle : Last Closed Candle
LastCandles : Last closed and last opened candles.
        lookbackperiods: Number of periods (N) for the lookback evaluation. Must be greater than 0. Default is 25.
        
    """
    url = f"https://cryptocator.p.rapidapi.com/ElderRayIndex"
    querystring = {'Exchange': exchange, 'Symbol': symbol, 'KlineInterval': klineinterval, 'useHeikinCandles': useheikincandles, }
    if resulttype:
        querystring['resultType'] = resulttype
    if lookbackperiods:
        querystring['lookbackPeriods'] = lookbackperiods
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cryptocator.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def atr_trailing_stop(klineinterval: str, useheikincandles: bool, exchange: str, symbol: str, lookbackperiods: int=14, multiplier: int=2, endtype: str='CLOSE', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Created by Welles Wilder, the ATR Trailing Stop indicator attempts to determine the primary trend of Close prices by using Average True Range (ATR) band thresholds. It can indicate a buy/sell signal or a trailing stop when the trend changes."
    klineinterval: 1m : OneMinute
3m : ThreeMinutes
5m : FiveMinutes
15m : FifteenMinutes
30m : ThirtyMinutes
1h : OneHour
4h : FourHour
6h : SixHour
12h : TwelveHour
1d : OneDay
1w : OneWeek
1mo : OneMonth
        exchange: Supported Exchanges :
BINANCE
BINANCE-FUTURES
BITFINEX
BITTREX
KRAKEN
BYBIT
HUOBI
KUCOIN
COINEX
        symbol: eg.  BTC-USDT
        lookbackperiods: Number of periods (N) for the ATR evaluation. Must be greater than 1. Default is 21.
        multiplier: Multiplier sets the ATR band width. Must be greater than 0 and is usually set around 2 to 3. Default is 3.
        endtype: Determines whether Close or High/Low is used as basis for stop offset. See EndType options below. Default is CLOSE.

EndType options
CLOSE : Stop offset from Close price (default)
HIGHLOW : Stop offset from High or Low price
        
    """
    url = f"https://cryptocator.p.rapidapi.com/AtrTrailingStop"
    querystring = {'KlineInterval': klineinterval, 'useHeikinCandles': useheikincandles, 'Exchange': exchange, 'Symbol': symbol, }
    if lookbackperiods:
        querystring['lookbackPeriods'] = lookbackperiods
    if multiplier:
        querystring['multiplier'] = multiplier
    if endtype:
        querystring['endType'] = endtype
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cryptocator.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

