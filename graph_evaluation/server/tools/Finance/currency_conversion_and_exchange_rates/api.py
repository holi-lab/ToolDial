import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def recent_exchange_rates(is_from: str='USD', to: str='EUR,GBP', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieve the latest exchange rate data. Refresh rate will depend on your subscription: updated every 60 minutes, every 10 minutes or every 60 seconds."
    is_from: Three-letter currency code of the currency would would like to convert from. This currency will be converted into the code of the currency provided in the `to` parameter
        to: A comma-separated list of currency codes to convert the `from` parameter into.
        
    """
    url = f"https://currency-conversion-and-exchange-rates.p.rapidapi.com/latest"
    querystring = {}
    if is_from:
        querystring['from'] = is_from
    if to:
        querystring['to'] = to
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "currency-conversion-and-exchange-rates.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def historical_exchange_rates(date: str, is_from: str='USD', to: str='EUR,GBP', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieve historical exchange rate data. Data is available for most currencies all the way back to the year of 1999."
    date: The date to retrieve the historical exchange rates from.
        is_from: Three-letter currency code of the currency would would like to convert from. This currency will be converted into the code of the currency provided in the `to` parameter
        to: A comma-separated list of currency codes to convert the `from` parameter into.
        
    """
    url = f"https://currency-conversion-and-exchange-rates.p.rapidapi.com/{date}"
    querystring = {}
    if is_from:
        querystring['from'] = is_from
    if to:
        querystring['to'] = to
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "currency-conversion-and-exchange-rates.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def convert(to: str, amount: str, is_from: str, date: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "In addition to providing converstion rates, our API provides a dedicated endpoint to easily do conversion on a specific amount of the currency."
    to: The three-letter currency code of the currency you would like to convert to.
        amount: The amount to be converted.
        from: The three-letter currency code of the currency you would like to convert from.
        date: Optionally, provide a specific date (format YYYY-MM-DD) to use historical rates for this conversion.
        
    """
    url = f"https://currency-conversion-and-exchange-rates.p.rapidapi.com/convert"
    querystring = {'to': to, 'amount': amount, 'from': is_from, }
    if date:
        querystring['date'] = date
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "currency-conversion-and-exchange-rates.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def symbols(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieve a list of all currently available currency symbols"
    
    """
    url = f"https://currency-conversion-and-exchange-rates.p.rapidapi.com/symbols"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "currency-conversion-and-exchange-rates.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def time_series_endpoint(start_date: str, end_date: str, is_from: str='USD', to: str='EUR,GBP', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieve historical rates between two specified dates.
		
		`Maximum of 365 day time range`"
    is_from: Three-letter currency code of the currency would would like to convert from. This currency will be converted into the code of the currency provided in the `to` parameter
        to: A comma-separated list of currency codes to convert the `from` parameter into.
        
    """
    url = f"https://currency-conversion-and-exchange-rates.p.rapidapi.com/timeseries"
    querystring = {'start_date': start_date, 'end_date': end_date, }
    if is_from:
        querystring['from'] = is_from
    if to:
        querystring['to'] = to
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "currency-conversion-and-exchange-rates.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

