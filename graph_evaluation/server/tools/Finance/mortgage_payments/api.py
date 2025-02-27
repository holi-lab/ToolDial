import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def payments(price: int, downpayment: int, interestrate: int, assessment: int=None, taxrate: int=None, type: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Calculate monthly mortgage payments"
    price: Total price of property
        downpayment: Amount paid at closing
        interestrate: Annual interest rate expressed as a decimal, e.g. 0.039
        assessment: Assessment for tax purposes; if omitted, assumes assessment equal to price.
        taxrate: Annual tax rate as a decimal
        type: Type of mortgage (fix30 or fix15): 30-year or 15-year fixed
        
    """
    url = f"https://shaisachs-mortgage-payments-v1.p.rapidapi.com/payments"
    querystring = {'price': price, 'downPayment': downpayment, 'interestRate': interestrate, }
    if assessment:
        querystring['assessment'] = assessment
    if taxrate:
        querystring['taxRate'] = taxrate
    if type:
        querystring['type'] = type
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "shaisachs-mortgage-payments-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

