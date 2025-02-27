import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_rsi_by_pairs(pairs: str, timeframe: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get RSI indicator for all trading pairs on Binance.com"
    pairs: Trading pairs for quote USDT, BTC, ETH or BNB in uppercase separated by commas
        timeframe: Timeframe for calculate RSI is available: 

- **15m** - 15 minutes 
- **1h** - 1 hour 
- **4h** - 4 hours
- **1d** - 1 day
        
    """
    url = f"https://binance-rsi-indicator.p.rapidapi.com/"
    querystring = {'pairs': pairs, 'timeframe': timeframe, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "binance-rsi-indicator.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

