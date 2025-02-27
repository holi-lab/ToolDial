import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_euro_exchange_rates(date: str, currency: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The value returned is the exchange rate for the specified date. If there is no data (as in closing days) then the value is empty and there is an error text."
    date: Date in the format YYYY-MM-DD (year-month-day).
        currency: The currency code (3 letters).
Choose one from this 41 currencies: USD, JPY, BGN, CYP, CZK, DKK, EEK, GBP, HUF, LTL, LVL, MTL, PLN, ROL, RON, SEK, SIT, SKK, CHF, ISK, NOK, HRK, RUB, TRL, TRY, AUD, BRL, CAD, CNY, HKD, IDR, ILS, INR, KRW, MXN, MYR, NZD, PHP, SGD, THB, ZAR.
        
    """
    url = f"https://euro-exchange-rates-ecb.p.rapidapi.com/ecb-eur-exchange/v1"
    querystring = {'date': date, 'currency': currency, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "euro-exchange-rates-ecb.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

