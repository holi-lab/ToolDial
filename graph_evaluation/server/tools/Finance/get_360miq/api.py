import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def market_breadth(exchange: str, series: str, bar: int=10, format: str='JSON', to: str='2023-01-31', is_from: str='2023-01-01', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Stock Market Breadth API for the US and Global Stock Markets"
    exchange: NYSE, Nasdaq, TSX, LSE, ASX, NSE, TYO, HKEX, SHSE, SZSE
        series: C>MA20, 
C>MA50, 
C>MA200, 
C>MA250, 
MA3>MA18, 
MA5>MA20, 
MA10>MA50, 
MA50>MA200, 
MA50>MA250, 
ADV, 
ADV-DEC, 
ADV-DEC_CUM, 
RSI14D<30, 
RSI14D>50, 
RSI14D>70, 
RSI14W<30, 
RSI14W>50, 
RSI14W>70, 
HIGH250-LOW250, 
HIGH250-LOW250_CUM, 
MCCLELLANOSC, 
MCCLELLANSUM
        bar: Number of the most recent bar
        format: CSV, JSON. Default is CSV
        to: To date
        is_from: From date
        
    """
    url = f"https://360miq1.p.rapidapi.com/"
    querystring = {'exchange': exchange, 'series': series, }
    if bar:
        querystring['bar'] = bar
    if format:
        querystring['format'] = format
    if to:
        querystring['to'] = to
    if is_from:
        querystring['from'] = is_from
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "360miq1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

