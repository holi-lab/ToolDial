import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_2_crows(symbol: str, exchange: str, market: str, interval: str, is_from: str='1683895800', backtracks: int=1, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "2 Crows indicator"
    market: Available markets are:

spot
usdt-perpetual
inverse-perpetual
        
    """
    url = f"https://qvantana.p.rapidapi.com/2crows"
    querystring = {'symbol': symbol, 'exchange': exchange, 'market': market, 'interval': interval, }
    if is_from:
        querystring['from'] = is_from
    if backtracks:
        querystring['backtracks'] = backtracks
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "qvantana.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def bollinger_bands(exchange: str, symbol: str, interval: str, market: str, length: int=20, is_from: str='1683895800', backtracks: int=1, stddev: int=2, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Bollinger Bands indicator"
    market: Available markets are:

spot
usdt-perpetual
inverse-perpetual
        length: Default 20
        stddev: Default 2
        
    """
    url = f"https://qvantana.p.rapidapi.com/bollingerbands"
    querystring = {'exchange': exchange, 'symbol': symbol, 'interval': interval, 'market': market, }
    if length:
        querystring['length'] = length
    if is_from:
        querystring['from'] = is_from
    if backtracks:
        querystring['backtracks'] = backtracks
    if stddev:
        querystring['stdDev'] = stddev
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "qvantana.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def fibonacci_retracement(exchange: str, symbol: str, interval: str, market: str, length: int, is_from: str='1683895800', backtracks: int=1, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Fibonacci Retracement"
    market: Available markets are:

spot
usdt-perpetual
inverse-perpetual
        length: Default 50. Lengths equals how many candlesticks back to do the calculation on.
        
    """
    url = f"https://qvantana.p.rapidapi.com/fibonacciretracement"
    querystring = {'exchange': exchange, 'symbol': symbol, 'interval': interval, 'market': market, 'length': length, }
    if is_from:
        querystring['from'] = is_from
    if backtracks:
        querystring['backtracks'] = backtracks
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "qvantana.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def doji(symbol: str, market: str, exchange: str, interval: str, is_from: str='1683895800', backtracks: int=1, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Doji indicator"
    market: Available markets are:

spot
usdt-perpetual
inverse-perpetual
        
    """
    url = f"https://qvantana.p.rapidapi.com/doji"
    querystring = {'symbol': symbol, 'market': market, 'exchange': exchange, 'interval': interval, }
    if is_from:
        querystring['from'] = is_from
    if backtracks:
        querystring['backtracks'] = backtracks
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "qvantana.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def abandoned_baby(market: str, interval: str, exchange: str, symbol: str, backtracks: int=1, is_from: str='1683895800', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Abandoned Baby indicator"
    market: Available markets are:

spot
usdt-perpetual
inverse-perpetual
        
    """
    url = f"https://qvantana.p.rapidapi.com/abandonedbaby"
    querystring = {'market': market, 'interval': interval, 'exchange': exchange, 'symbol': symbol, }
    if backtracks:
        querystring['backtracks'] = backtracks
    if is_from:
        querystring['from'] = is_from
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "qvantana.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_3_line_strike(symbol: str, interval: str, exchange: str, market: str, backtracks: int=1, is_from: str='1683895800', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "3 Line Strike indicator"
    market: Available markets are:

spot
usdt-perpetual
inverse-perpetual
        
    """
    url = f"https://qvantana.p.rapidapi.com/3linestrike"
    querystring = {'symbol': symbol, 'interval': interval, 'exchange': exchange, 'market': market, }
    if backtracks:
        querystring['backtracks'] = backtracks
    if is_from:
        querystring['from'] = is_from
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "qvantana.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def three_inside_up_down(exchange: str, interval: str, market: str, symbol: str, backtracks: int=1, is_from: str='1683895800', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Three Inside Up/Down indicator"
    market: Available markets are:

spot
usdt-perpetual
inverse-perpetual
        
    """
    url = f"https://qvantana.p.rapidapi.com/3inside"
    querystring = {'exchange': exchange, 'interval': interval, 'market': market, 'symbol': symbol, }
    if backtracks:
        querystring['backtracks'] = backtracks
    if is_from:
        querystring['from'] = is_from
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "qvantana.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_3_black_crows(exchange: str, interval: str, symbol: str, market: str, backtracks: int=1, is_from: str='1683895800', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "3 Black Crows indicator"
    market: Available markets are:

spot
usdt-perpetual
inverse-perpetual
        
    """
    url = f"https://qvantana.p.rapidapi.com/3blackcrows"
    querystring = {'exchange': exchange, 'interval': interval, 'symbol': symbol, 'market': market, }
    if backtracks:
        querystring['backtracks'] = backtracks
    if is_from:
        querystring['from'] = is_from
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "qvantana.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def kline(exchange: str, market: str, interval: str, symbol: str, to: str='1683896400', is_from: str='1683895800', limit: int=100, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint retrieves historical k-line (candlestick) data for a specific cryptocurrency pair on a chosen exchange and market. The endpoint accepts the following parameters:
		
		symbol (required): The trading pair (e.g., 'ethusdt'). The availability of symbols depends on the chosen exchange.
		market (required): The market type. Options include 'usdt-perpetual', 'spot', and 'inverse-perpetual'.
		interval (required): The timeframe for each k-line data point. The available timeframes depend on the chosen exchange.
		limit (optional): The number of data points to return. The maximum limit is 5000.
		from & to (optional): The start and end time for the data, respectively. They can be in UNIX timestamp format (seconds) or 'yyyy-MM-dd HH:mm:ss' format.
		exchange (required): The exchange to fetch data from. Available options are 'bybit', 'binance', 'bitstamp', 'gateio', and 'kraken'.
		This endpoint offers comprehensive historical trading data tailored to your specific needs, providing a valuable tool for market analysis."
    market: Available markets are:

spot
usdt-perpetual
inverse-perpetual
        to: Can be in UNIX timestamp format (seconds) or 'yyyy-MM-dd HH:mm:ss' format.
        is_from: Can be in UNIX timestamp format (seconds) or 'yyyy-MM-dd HH:mm:ss' format.
        
    """
    url = f"https://qvantana.p.rapidapi.com/kline"
    querystring = {'exchange': exchange, 'market': market, 'interval': interval, 'symbol': symbol, }
    if to:
        querystring['to'] = to
    if is_from:
        querystring['from'] = is_from
    if limit:
        querystring['limit'] = limit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "qvantana.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def backtrack_kline(interval: str, market: str, symbol: str, exchange: str, limit: int=100, is_from: str='1683895800', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint retrieves historical k-line (candlestick) data for a specific cryptocurrency pair on a chosen exchange and market. This endpoint is equal to the Kline endpoint except it returns the result up until the "from" parameter. The endpoint accepts the following parameters:
		
		symbol (required): The trading pair (e.g., 'ethusdt'). The availability of symbols depends on the chosen exchange.
		market (required): The market type. Options include 'usdt-perpetual', 'spot', and 'inverse-perpetual'.
		interval (required): The timeframe for each k-line data point. The available timeframes depend on the chosen exchange.
		limit (optional): The number of data points to return. The maximum limit is 5000.
		from & to (optional): The start and end time for the data, respectively. They can be in UNIX timestamp format (seconds) or 'yyyy-MM-dd HH:mm:ss' format.
		exchange (required): The exchange to fetch data from. Available options are 'bybit', 'binance', 'bitstamp', 'gateio', and 'kraken'.
		This endpoint offers comprehensive historical trading data tailored to your specific needs, providing a valuable tool for market analysis."
    market: Available markets are:

spot
usdt-perpetual
inverse-perpetual
        is_from: Can be in UNIX timestamp format (seconds) or 'yyyy-MM-dd HH:mm:ss' format.
        
    """
    url = f"https://qvantana.p.rapidapi.com/kline/backtrack"
    querystring = {'interval': interval, 'market': market, 'symbol': symbol, 'exchange': exchange, }
    if limit:
        querystring['limit'] = limit
    if is_from:
        querystring['from'] = is_from
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "qvantana.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def exchanges(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Endpoint for all the available exchanges"
    
    """
    url = f"https://qvantana.p.rapidapi.com/exchanges"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "qvantana.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def moving_average_ma_sma(market: str, interval: str, symbol: str, exchange: str, is_from: str='1683895800', length: int=9, backtracks: int=1, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Moving average (MA/SMA) indicator"
    market: Available markets are:

spot
usdt-perpetual
inverse-perpetual
        length: Default 9
        
    """
    url = f"https://qvantana.p.rapidapi.com/ma"
    querystring = {'market': market, 'interval': interval, 'symbol': symbol, 'exchange': exchange, }
    if is_from:
        querystring['from'] = is_from
    if length:
        querystring['length'] = length
    if backtracks:
        querystring['backtracks'] = backtracks
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "qvantana.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def exponential_moving_average_ema(market: str, exchange: str, symbol: str, interval: str, backtracks: int=1, length: int=9, is_from: str='1683895800', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Exponential Moving Average (EMA) indicator"
    market: Available markets are:

spot
usdt-perpetual
inverse-perpetual
        length: Default 9
        
    """
    url = f"https://qvantana.p.rapidapi.com/ema"
    querystring = {'market': market, 'exchange': exchange, 'symbol': symbol, 'interval': interval, }
    if backtracks:
        querystring['backtracks'] = backtracks
    if length:
        querystring['length'] = length
    if is_from:
        querystring['from'] = is_from
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "qvantana.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def relative_strength_index_rsi(market: str, exchange: str, symbol: str, interval: str, backtracks: int=1, length: int=14, is_from: str='1683895800', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Stochastic Relative Strength Index (RSI) indicator"
    market: Available markets are:

spot
usdt-perpetual
inverse-perpetual
        length: Default 14
        
    """
    url = f"https://qvantana.p.rapidapi.com/rsi"
    querystring = {'market': market, 'exchange': exchange, 'symbol': symbol, 'interval': interval, }
    if backtracks:
        querystring['backtracks'] = backtracks
    if length:
        querystring['length'] = length
    if is_from:
        querystring['from'] = is_from
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "qvantana.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def parabolic_sar(exchange: str, interval: str, symbol: str, market: str, is_from: str='1683895800', max: int=0, backtracks: int=1, increment: int=0, start: int=0, af: int=2, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Parabolic SAR indicator"
    market: Available markets are:

spot
usdt-perpetual
inverse-perpetual
        max: Default 0.2
        increment: Default 0.02
        start: Default 0.02
        af: Default 2
        
    """
    url = f"https://qvantana.p.rapidapi.com/sar"
    querystring = {'exchange': exchange, 'interval': interval, 'symbol': symbol, 'market': market, }
    if is_from:
        querystring['from'] = is_from
    if max:
        querystring['max'] = max
    if backtracks:
        querystring['backtracks'] = backtracks
    if increment:
        querystring['increment'] = increment
    if start:
        querystring['start'] = start
    if af:
        querystring['af'] = af
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "qvantana.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def momentum_mom(exchange: str, market: str, symbol: str, interval: str, is_from: str='1683895800', backtracks: int=1, length: int=10, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Momentum (MOM) indicator"
    market: Available markets are:

spot
usdt-perpetual
inverse-perpetual
        length: Default 10
        
    """
    url = f"https://qvantana.p.rapidapi.com/mom"
    querystring = {'exchange': exchange, 'market': market, 'symbol': symbol, 'interval': interval, }
    if is_from:
        querystring['from'] = is_from
    if backtracks:
        querystring['backtracks'] = backtracks
    if length:
        querystring['length'] = length
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "qvantana.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def moving_average_convergence_divergence_macd(exchange: str, market: str, symbol: str, interval: str, slowperiod: int=26, is_from: str='1683895800', fastperiod: int=12, backtracks: int=1, signalperiod: int=9, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Moving Average Convergence Divergence (MACD) indicator"
    market: Available markets are:

spot
usdt-perpetual
inverse-perpetual
        slowperiod: Default 26
        fastperiod: Default 12
        signalperiod: Default 9
        
    """
    url = f"https://qvantana.p.rapidapi.com/macd"
    querystring = {'exchange': exchange, 'market': market, 'symbol': symbol, 'interval': interval, }
    if slowperiod:
        querystring['slowPeriod'] = slowperiod
    if is_from:
        querystring['from'] = is_from
    if fastperiod:
        querystring['fastPeriod'] = fastperiod
    if backtracks:
        querystring['backtracks'] = backtracks
    if signalperiod:
        querystring['signalPeriod'] = signalperiod
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "qvantana.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def ichimoku_cloud(market: str, symbol: str, exchange: str, interval: str, is_from: str='1683895800', tenkansenperiod: int=9, kijunsenperiod: int=26, chikouspanperiod: int=26, backtracks: int=1, senkouspanbperiod: int=52, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Ichimoku Cloud indicator"
    market: Available markets are:

spot
usdt-perpetual
inverse-perpetual
        tenkansenperiod: Default 9
        kijunsenperiod: Default 26
        chikouspanperiod: Default 26
        senkouspanbperiod: Default 52
        
    """
    url = f"https://qvantana.p.rapidapi.com/ichimoku"
    querystring = {'market': market, 'symbol': symbol, 'exchange': exchange, 'interval': interval, }
    if is_from:
        querystring['from'] = is_from
    if tenkansenperiod:
        querystring['tenkanSenPeriod'] = tenkansenperiod
    if kijunsenperiod:
        querystring['kijunSenPeriod'] = kijunsenperiod
    if chikouspanperiod:
        querystring['chikouSpanPeriod'] = chikouspanperiod
    if backtracks:
        querystring['backtracks'] = backtracks
    if senkouspanbperiod:
        querystring['senkouSpanBPeriod'] = senkouspanbperiod
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "qvantana.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def chaikin_money_flow_cmf(exchange: str, interval: str, market: str, symbol: str, is_from: str='1683895800', length: str='20', backtracks: int=1, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Chaikin Money Flow (CMF) indicator"
    market: Available markets are:

spot
usdt-perpetual
inverse-perpetual
        length: Default 20
        
    """
    url = f"https://qvantana.p.rapidapi.com/cmf"
    querystring = {'exchange': exchange, 'interval': interval, 'market': market, 'symbol': symbol, }
    if is_from:
        querystring['from'] = is_from
    if length:
        querystring['length'] = length
    if backtracks:
        querystring['backtracks'] = backtracks
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "qvantana.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def average_true_range_atr(exchange: str, market: str, symbol: str, interval: str, backtracks: int=1, length: int=14, is_from: str='1683895800', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Average True Range (ATR) indicator"
    market: Available markets are:

spot
usdt-perpetual
inverse-perpetual
        length: Default 14
        
    """
    url = f"https://qvantana.p.rapidapi.com/atr"
    querystring = {'exchange': exchange, 'market': market, 'symbol': symbol, 'interval': interval, }
    if backtracks:
        querystring['backtracks'] = backtracks
    if length:
        querystring['length'] = length
    if is_from:
        querystring['from'] = is_from
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "qvantana.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def commodity_channel_index_cci(exchange: str, market: str, interval: str, symbol: str, is_from: str='1683895800', length: str='20', backtracks: int=1, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Commodity Channel Index (CCI) indicator"
    market: Available markets are:

spot
usdt-perpetual
inverse-perpetual
        length: Default 20
        
    """
    url = f"https://qvantana.p.rapidapi.com/cci"
    querystring = {'exchange': exchange, 'market': market, 'interval': interval, 'symbol': symbol, }
    if is_from:
        querystring['from'] = is_from
    if length:
        querystring['length'] = length
    if backtracks:
        querystring['backtracks'] = backtracks
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "qvantana.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def three_advancing_white_soldiers(market: str, interval: str, exchange: str, symbol: str, is_from: str='1683895800', backtracks: int=1, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Three Advancing White Soldiers indicator"
    market: Available markets are:

spot
usdt-perpetual
inverse-perpetual
        
    """
    url = f"https://qvantana.p.rapidapi.com/3whitesoldiers"
    querystring = {'market': market, 'interval': interval, 'exchange': exchange, 'symbol': symbol, }
    if is_from:
        querystring['from'] = is_from
    if backtracks:
        querystring['backtracks'] = backtracks
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "qvantana.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def three_stars_in_the_south(interval: str, exchange: str, symbol: str, market: str, is_from: str='1683895800', backtracks: int=1, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Three Stars In The South indicator"
    market: Available markets are:

spot
usdt-perpetual
inverse-perpetual
        
    """
    url = f"https://qvantana.p.rapidapi.com/3starsinsouth"
    querystring = {'interval': interval, 'exchange': exchange, 'symbol': symbol, 'market': market, }
    if is_from:
        querystring['from'] = is_from
    if backtracks:
        querystring['backtracks'] = backtracks
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "qvantana.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def three_outside_up_down(symbol: str, exchange: str, interval: str, market: str, is_from: str='1683895800', backtracks: int=1, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Three Outside Up/Down indicator"
    market: Available markets are:

spot
usdt-perpetual
inverse-perpetual
        
    """
    url = f"https://qvantana.p.rapidapi.com/3outside"
    querystring = {'symbol': symbol, 'exchange': exchange, 'interval': interval, 'market': market, }
    if is_from:
        querystring['from'] = is_from
    if backtracks:
        querystring['backtracks'] = backtracks
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "qvantana.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def volume_weighted_average_price_vwap(exchange: str, symbol: str, interval: str, market: str, backtracks: int=1, is_from: str='1683895800', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Volume Weighted Average Price (VWAP) indicator"
    market: Available markets are:

spot
usdt-perpetual
inverse-perpetual
        
    """
    url = f"https://qvantana.p.rapidapi.com/vwap"
    querystring = {'exchange': exchange, 'symbol': symbol, 'interval': interval, 'market': market, }
    if backtracks:
        querystring['backtracks'] = backtracks
    if is_from:
        querystring['from'] = is_from
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "qvantana.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def hull_moving_average(exchange: str, market: str, symbol: str, interval: str, is_from: str='1683895800', backtracks: int=1, length: int=9, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Hull Moving Average indicator"
    market: Available markets are:

spot
usdt-perpetual
inverse-perpetual
        length: Default 9
        
    """
    url = f"https://qvantana.p.rapidapi.com/hma"
    querystring = {'exchange': exchange, 'market': market, 'symbol': symbol, 'interval': interval, }
    if is_from:
        querystring['from'] = is_from
    if backtracks:
        querystring['backtracks'] = backtracks
    if length:
        querystring['length'] = length
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "qvantana.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def ultimateoscillator(exchange: str, interval: str, market: str, symbol: str, is_from: str='1683895800', shortperiod: int=7, mediumperiod: int=14, longperiod: int=28, backtracks: int=1, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "UltimateOscillator indicator"
    market: Available markets are:

spot
usdt-perpetual
inverse-perpetual
        shortperiod: Default 7
        mediumperiod: Default 14
        longperiod: Default 28
        
    """
    url = f"https://qvantana.p.rapidapi.com/uo"
    querystring = {'exchange': exchange, 'interval': interval, 'market': market, 'symbol': symbol, }
    if is_from:
        querystring['from'] = is_from
    if shortperiod:
        querystring['shortPeriod'] = shortperiod
    if mediumperiod:
        querystring['mediumPeriod'] = mediumperiod
    if longperiod:
        querystring['longPeriod'] = longperiod
    if backtracks:
        querystring['backtracks'] = backtracks
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "qvantana.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def typical_price(market: str, exchange: str, symbol: str, interval: str, backtracks: int=1, length: int=18, is_from: str='1683895800', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Typical price indicator"
    market: Available markets are:

spot
usdt-perpetual
inverse-perpetual
        length: Default 18
        
    """
    url = f"https://qvantana.p.rapidapi.com/tp"
    querystring = {'market': market, 'exchange': exchange, 'symbol': symbol, 'interval': interval, }
    if backtracks:
        querystring['backtracks'] = backtracks
    if length:
        querystring['length'] = length
    if is_from:
        querystring['from'] = is_from
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "qvantana.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def trix(market: str, interval: str, exchange: str, symbol: str, is_from: str='1683895800', length: int=18, backtracks: int=1, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "TRIX indicator"
    market: Available markets are:

spot
usdt-perpetual
inverse-perpetual
        length: Default 18
        
    """
    url = f"https://qvantana.p.rapidapi.com/trix"
    querystring = {'market': market, 'interval': interval, 'exchange': exchange, 'symbol': symbol, }
    if is_from:
        querystring['from'] = is_from
    if length:
        querystring['length'] = length
    if backtracks:
        querystring['backtracks'] = backtracks
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "qvantana.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def true_range(market: str, symbol: str, exchange: str, interval: str, is_from: str='1683895800', backtracks: int=1, length: int=100, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "True range indicator"
    market: Available markets are:

spot
usdt-perpetual
inverse-perpetual
        length: Default 100
        
    """
    url = f"https://qvantana.p.rapidapi.com/truerange"
    querystring = {'market': market, 'symbol': symbol, 'exchange': exchange, 'interval': interval, }
    if is_from:
        querystring['from'] = is_from
    if backtracks:
        querystring['backtracks'] = backtracks
    if length:
        querystring['length'] = length
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "qvantana.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def stochastic_rsi(exchange: str, symbol: str, market: str, interval: str, is_from: str='1683895800', rsilength: int=14, backtracks: int=1, stochlength: int=14, smoothk: int=3, smoothd: int=3, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Stochastic RSI indicator"
    market: Available markets are:

spot
usdt-perpetual
inverse-perpetual
        rsilength: Default 14
        stochlength: Default 14
        smoothk: Default 3
        smoothd: Default 3
        
    """
    url = f"https://qvantana.p.rapidapi.com/stochrsi"
    querystring = {'exchange': exchange, 'symbol': symbol, 'market': market, 'interval': interval, }
    if is_from:
        querystring['from'] = is_from
    if rsilength:
        querystring['rsiLength'] = rsilength
    if backtracks:
        querystring['backtracks'] = backtracks
    if stochlength:
        querystring['stochLength'] = stochlength
    if smoothk:
        querystring['smoothK'] = smoothk
    if smoothd:
        querystring['smoothD'] = smoothd
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "qvantana.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def stochastic(symbol: str, interval: str, market: str, exchange: str, is_from: str='1683895800', backtracks: int=1, kperiod: int=14, dperiod: int=3, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Stochastic indicator"
    market: Available markets are:

spot
usdt-perpetual
inverse-perpetual
        kperiod: Default 14
        dperiod: Default 3
        
    """
    url = f"https://qvantana.p.rapidapi.com/stochastic"
    querystring = {'symbol': symbol, 'interval': interval, 'market': market, 'exchange': exchange, }
    if is_from:
        querystring['from'] = is_from
    if backtracks:
        querystring['backtracks'] = backtracks
    if kperiod:
        querystring['kPeriod'] = kperiod
    if dperiod:
        querystring['dPeriod'] = dperiod
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "qvantana.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def standard_deviation(exchange: str, market: str, interval: str, symbol: str, backtracks: int=1, is_from: str='1683895800', length: int=20, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Standard deviation indicator"
    market: Available markets are:

spot
usdt-perpetual
inverse-perpetual
        length: Default 20
        
    """
    url = f"https://qvantana.p.rapidapi.com/std"
    querystring = {'exchange': exchange, 'market': market, 'interval': interval, 'symbol': symbol, }
    if backtracks:
        querystring['backtracks'] = backtracks
    if is_from:
        querystring['from'] = is_from
    if length:
        querystring['length'] = length
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "qvantana.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def stalled_pattern(interval: str, exchange: str, symbol: str, market: str, length: int=10, is_from: str='1683895800', backtracks: int=1, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Stalled pattern indicator"
    market: Available markets are:

spot
usdt-perpetual
inverse-perpetual
        length: Default 10
        
    """
    url = f"https://qvantana.p.rapidapi.com/stalled"
    querystring = {'interval': interval, 'exchange': exchange, 'symbol': symbol, 'market': market, }
    if length:
        querystring['length'] = length
    if is_from:
        querystring['from'] = is_from
    if backtracks:
        querystring['backtracks'] = backtracks
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "qvantana.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

