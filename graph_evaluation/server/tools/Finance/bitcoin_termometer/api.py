import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def planb_stock_to_flow_modeling_prediction(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Return the followings variables:
		
		Current date: date the data was processed.
		Current price: market price in dollars
		Predicted price: price predicted by the model.
		Predicted × 1SE:	multiplied by 1 standard error
		Predicted ÷ 1SE:	divided by 1 standard error
		Predicted × 2SE:	multiplied by 2 standard errors
		Predicted ÷ 2SE: divided by 2 standard errors
		
		PlanB: https://twitter.com/100trillionusd"
    
    """
    url = f"https://bitcoin-termometer.p.rapidapi.com/s2f"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "bitcoin-termometer.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def btc_30_days_volatility(period_of_volatility: int=30, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint returns last 30 days volatility of Bitcoin's price."
    
    """
    url = f"https://bitcoin-termometer.p.rapidapi.com/btc_vt"
    querystring = {}
    if period_of_volatility:
        querystring['period_of_volatility'] = period_of_volatility
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "bitcoin-termometer.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def btc_moving_average(btc_average_days: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Moving Average for Bitcoin Price for the last 100 or 200 days"
    btc_average_days: Valid parameters are "100" or "200". These parameters correspond to the number of the days for Moving Average of Bitcoin Price to be calculated.
        
    """
    url = f"https://bitcoin-termometer.p.rapidapi.com/btc_av"
    querystring = {'btc_average_days': btc_average_days, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "bitcoin-termometer.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

