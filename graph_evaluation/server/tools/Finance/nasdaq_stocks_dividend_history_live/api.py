import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def yahoo_finance_api_alternative_with_dividends_data(stockcode: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint returns all the Dividend History of a Nasdaq Stock Quote from past 20-30 years to present day"
    stockcode: For Nasdaq Stocks, accepted Params are the stock quotes like: 
- MSFT [for Microsoft]
- AAPL [for Apple]
- TSLA [for Tesla]
as long as you have the designated/official stock quote/code of the target stock, you can search it using this API

more details on searching KLSE and SGX counters, you may click the link provided below;
[API Documentation](https://rapidapi.com/moneygoddess888/api/nasdaq-stocks-dividend-history-live/details)


        
    """
    url = f"https://nasdaq-stocks-dividend-history-live.p.rapidapi.com/rapid-dividend/{stockcode}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "nasdaq-stocks-dividend-history-live.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

