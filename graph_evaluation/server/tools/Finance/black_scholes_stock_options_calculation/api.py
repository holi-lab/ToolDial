import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def calculate_options_value(k: int, s: int, q: int, t: int, r: int, sigma: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Pass the parameter like Spot Price, Strike Price, Time to expiry, Implied Volatility, Risk-free return, and stocks divided and it will calculate all of the parameters and the values with the change of each underlying value like time, volatility, risk-free return, or change in the spot price."
    k: Strike Price
        s: Spot Price
        q: Stock's dividend yield
        t: Time to expiry
        r: Risk free return
        sigma: Stocks implied volatility
        
    """
    url = f"https://black-scholes-stock-options-calculation.p.rapidapi.com/api/v1/options"
    querystring = {'k': k, 's': s, 'q': q, 't': t, 'r': r, 'sigma': sigma, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "black-scholes-stock-options-calculation.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

