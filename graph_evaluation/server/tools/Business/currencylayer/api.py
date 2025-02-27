import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def list_endpoint_obtain_a_json_list_of_all_supported_currencies(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    ""
    
    """
    url = f"https://apilayer-currencylayer-v1.p.rapidapi.com/list"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "apilayer-currencylayer-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def change_endpoint_request_a_currency_s_change_parameters(currencies: str=None, start_date: str=None, end_date: str=None, source: str='USD', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Using the "change" API endpoint, you can obtain the "change" parameters (margin and percentage) of one or more currencies, optionally within a specific time-frame. Important: If you do not specify a time-frame, change parameters for "yesterday to now" will be displayed."
    currencies: "currencies" - request only specific currencies (e.g. EUR,CAD,GBP)
        start_date: "start_date" - optional - the starting date of the time-frame you want to specify (format: YYYY-MM-DD)
        end_date: "end_date" - optional - the end date of the time-frame you want to specify (format: YYYY-MM-DD)
        source: "source" - the currency to which all other exchange rates are relative ("Source Currency")
        
    """
    url = f"https://apilayer-currencylayer-v1.p.rapidapi.com/change"
    querystring = {}
    if currencies:
        querystring['currencies'] = currencies
    if start_date:
        querystring['start_date'] = start_date
    if end_date:
        querystring['end_date'] = end_date
    if source:
        querystring['source'] = source
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "apilayer-currencylayer-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def historical_endpoint_request_exchange_rates_for_a_specific_date(date: str, currencies: str=None, source: str='USD', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Make a request to the API's "historical" endpoint providing a valid "date" in the format "YYYY-MM-DD""
    date: The date you want to obtain exchange rates for (e.g. 2014-12-31)
        currencies: "currencies" - request only specific currencies (e.g. EUR,CAD,GBP)
        source: "source" - the currency to which all other exchange rates are relative ("Source Currency")
        
    """
    url = f"https://apilayer-currencylayer-v1.p.rapidapi.com/historical"
    querystring = {'date': date, }
    if currencies:
        querystring['currencies'] = currencies
    if source:
        querystring['source'] = source
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "apilayer-currencylayer-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def convert_endpoint_convert_any_amount_from_one_currency_to_another(is_from: str, to: str, amount: str, date: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The "convert" endpoint lets you convert any amount from one currency to another using real-time exchange rates. If you append an additional "date" parameter, exchange rates of the specified dates will be used for your conversion."
    from: "from" - the currency your conversion is based on
        to: "to" - the currency you convert the specified amount to
        amount: "amount" - the amount to be converted
        date: "date" (format: YYYY-MM-DD) - perform the conversion with historical data from the date you specify
        
    """
    url = f"https://apilayer-currencylayer-v1.p.rapidapi.com/convert"
    querystring = {'from': is_from, 'to': to, 'amount': amount, }
    if date:
        querystring['date'] = date
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "apilayer-currencylayer-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def timeframe_endpoint_request_exchange_rates_for_a_specified_timeframe(start_date: str, end_date: str, currencies: str=None, source: str='USD', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Appending a valid "start_date" and "end_date", the API's "timeframe" endpoint will return exchange rates for this specified period."
    start_date: "start_date" - the starting date of the time-frame you want to specify
        end_date: "end_date" - the end date of the time-frame you want to specify
        currencies: "currencies" - request only specific currencies (e.g. EUR,CAD,GBP)
        source: "source" - the currency to which all other exchange rates are relative ("Source Currency")
        
    """
    url = f"https://apilayer-currencylayer-v1.p.rapidapi.com/timeframe"
    querystring = {'start_date': start_date, 'end_date': end_date, }
    if currencies:
        querystring['currencies'] = currencies
    if source:
        querystring['source'] = source
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "apilayer-currencylayer-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

