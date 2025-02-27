import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def calculate_repayment_schedule(installmentcount: str, principalamount: str, annualinterestrate: str, startdate: str, repaymentinterval: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint calculates repayment schedule."
    repaymentinterval: default: 'month'
        
    """
    url = f"https://loan-amortization-calculator.p.rapidapi.com/loan-amortization"
    querystring = {'installmentCount': installmentcount, 'principalAmount': principalamount, 'annualInterestRate': annualinterestrate, 'startDate': startdate, }
    if repaymentinterval:
        querystring['repaymentInterval'] = repaymentinterval
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "loan-amortization-calculator.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

