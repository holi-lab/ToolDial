import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def historic_bitcoin_price_for_a_specific_date_and_time(time: str, date: str, timezone: str, setting_flag: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a Bitcoin Price for a specified date and time in history starting in 2012.  There are two types of data, historic (BTCUSD_from_HIS field in the JSON return package) or current (BTCUSD_from_COI, GEM, KRA, BIN).  Price snapshot data is generally available in minute increments."
    
    """
    url = f"https://historic-bitcoin-price.p.rapidapi.com/api.php?action=price_retrieval"
    querystring = {'time': time, 'date': date, 'timezone': timezone, 'setting_flag': setting_flag, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "historic-bitcoin-price.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def data_verification(verify_timezone: str, number_of_minutes: int, time_delay_in_hours: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Reviews the stored bitcoin price data to see how far back in time data is stored within a specified gap (number_of_minutes)."
    
    """
    url = f"https://historic-bitcoin-price.p.rapidapi.com/api.php?action=verify"
    querystring = {'verify_timezone': verify_timezone, 'number_of_minutes': number_of_minutes, 'time_delay_in_hours': time_delay_in_hours, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "historic-bitcoin-price.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

