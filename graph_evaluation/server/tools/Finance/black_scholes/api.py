import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def price(interest_rate: int=None, volatility: int=None, strike: int=100, spot: int=100, dividend_yield: int=None, valuation_date: str='2022-04-27', option_type: str='Call', maturity: str='2023-04-27', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Calculate the Price & Greeks of Vanilla Call or Put"
    
    """
    url = f"https://black-scholes.p.rapidapi.com/price"
    querystring = {}
    if interest_rate:
        querystring['interest_rate'] = interest_rate
    if volatility:
        querystring['volatility'] = volatility
    if strike:
        querystring['strike'] = strike
    if spot:
        querystring['spot'] = spot
    if dividend_yield:
        querystring['dividend_yield'] = dividend_yield
    if valuation_date:
        querystring['valuation_date'] = valuation_date
    if option_type:
        querystring['option_type'] = option_type
    if maturity:
        querystring['maturity'] = maturity
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "black-scholes.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

