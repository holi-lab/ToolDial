import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def calculate_repayment_schedule(annualinterestrate: int, installmentcount: int, startdate: str, principalamount: int, repaymentinterval: str='month', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns detailed loan repayment schedule"
    annualinterestrate: Decimal number: (annual interest rate in percent) / 100
        installmentcount: Number of installments (payments)
        startdate: Start date of the schedule
        principalamount: Principal amount of the loan
        repaymentinterval: Allowed values: year, month, week, biweekly
        
    """
    url = f"https://loan-amortization-schedule-calculator.p.rapidapi.com/v1/"
    querystring = {'annualInterestRate': annualinterestrate, 'installmentCount': installmentcount, 'startDate': startdate, 'principalAmount': principalamount, }
    if repaymentinterval:
        querystring['repaymentInterval'] = repaymentinterval
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "loan-amortization-schedule-calculator.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

