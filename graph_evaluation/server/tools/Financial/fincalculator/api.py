import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def recurring_deposit(rate: str, year: int, month: int, mon_inv: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Calculate Recurring Deposit"
    rate: Interest Rate
        year: No. of Years
        month: No. of Months
        mon_inv: Monthly Investment
        
    """
    url = f"https://fincalculator.p.rapidapi.com/rd"
    querystring = {'rate': rate, 'year': year, 'month': month, 'mon_inv': mon_inv, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "fincalculator.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def fixed_deposit(year: int, month: int, pa: str, day: int, rate: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Calculate Fixed Deposit"
    year: No. of Years
        month: No. of Months
        pa: Principal Amount
        day: No. of Days
        rate: Interest Rate
        
    """
    url = f"https://fincalculator.p.rapidapi.com/fd"
    querystring = {'year': year, 'month': month, 'pa': pa, 'day': day, 'rate': rate, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "fincalculator.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

