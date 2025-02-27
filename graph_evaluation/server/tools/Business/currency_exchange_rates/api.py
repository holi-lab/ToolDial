import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def getting_the_latest_exchange_rate_s(base: str, target: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The /live/ endpoint returns the most recent exchange rate(s) for a given set of currencies. The base currency is the currency you're converting FROM, while the target currency or currencies are what you're converting TO.
		
		If no target currency or currencies are specified, then all available currencies are returned. You can see a full list of supported currencies at the bottom of this documentation."
    base: The base currency used to get the latest exchange rate(s) for. Uses the ISO 4217 currency standard (e.g., USD for United States Dollars), like all currency parameters in this API. You can see a full list of supported currencies here.
        target: The target currency or currencies to get the exchange rate of versus the base currency. Like the base parameters, any currency passed here follows the ISO 4217 standard. If multiple currencies are passed in the target, they should be separated by commas (e.g., &target=EUR,CAD,AUD).
        
    """
    url = f"https://currency-exchange-rates3.p.rapidapi.com/v1/live"
    querystring = {'base': base, }
    if target:
        querystring['target'] = target
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "currency-exchange-rates3.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def converting_currencies(base: str, target: str, date: str='2020-01-01', base_amount: str='500', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The /convert/ endpoint is similar to the /live/ and /historical/ endpoints, except it allows you to convert an arbitrary amount of currency.
		
		For example, it calculates how many EUR you would get for 1,337.99 USD today, or how many GBP and CAD you'd get for that amount on January 1, 2020."
    base: The base currency used to get the latest exchange rate(s) for. Uses the ISO 4217 currency standard (e.g., USD for United States Dollars), like all currency parameters in this API. You can see a full list of supported currencies here.
        target: The target currency or currencies to get the exchange rate of versus the base currency. Like the base parameters, any currency passed here follows the ISO 4217 standard. Note that unlinke the other endpoints, /convert/ only accepts one target currency at a time.
        date: The historical date you'd like to get rates from, in the format of YYYY-MM-DD. If you leave this blank, it will use the latest available rate(s).
        base_amount: The amount of the base currency you would like to convert to the target currency.
        
    """
    url = f"https://currency-exchange-rates3.p.rapidapi.com/v1/convert"
    querystring = {'base': base, 'target': target, }
    if date:
        querystring['date'] = date
    if base_amount:
        querystring['base_amount'] = base_amount
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "currency-exchange-rates3.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getting_historical_exchange_rate_s(date: str, base: str, target: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The /historical/ endpoint functions almost exactly like the /live/ endpoint, except it requires you to request a date parameter, and subsequently returns the returns the most recent exchange rate(s) for a given set of currencies.
		
		As with the /live/ endpoint, the base currency is the currency you're converting FROM, while the target currency or currencies are what you're converting TO. Also, if no target currency or currencies are specified, then all available currencies are returned. You can see a full list of supported currencies at the bottom of this documentation."
    date: The historical date you'd like to get rates from, in the format of YYYY-MM-DD
        target: The target currency or currencies to get the exchange rate of versus the base currency. Like the base parameters, any currency passed here follows the ISO 4217 standard. If multiple currencies are passed in the target, they should be separated by commas (e.g., &target=EUR,CAD,AUD).
        
    """
    url = f"https://currency-exchange-rates3.p.rapidapi.com/v1/historical"
    querystring = {'date': date, 'base': base, }
    if target:
        querystring['target'] = target
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "currency-exchange-rates3.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

