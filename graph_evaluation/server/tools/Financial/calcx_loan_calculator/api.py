import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def calculate_loan_get(term: int, rate: int, loan: int, currency: str='USD', type: str='student', extra: int=0, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint calculates the total cost of a loan given the loan amount, interest rate, and repayment term."
    term: The repayment term in months as an integer. **Example: term=12**.
        rate: The annual interest rate as a float. **Example: rate=5.5**.
        loan: The loan amount as a float. **Example: loan=10000**.
        currency: The desired currency for the response. Defaults to \\\"**USD**\\\".
        type: The loan type as a string. This parameter is case-insensitive and can take one of the following values:  **mortgage**, **auto**, **business**, **student**, **medical** or **personal**. (optional)
        extra: The extra payment per month as a float. **Example: extra=500.0**. **Default value: 0.0**. (optional)
        
    """
    url = f"https://calcx-loan-calculator.p.rapidapi.com/calculate"
    querystring = {'term': term, 'rate': rate, 'loan': loan, }
    if currency:
        querystring['currency'] = currency
    if type:
        querystring['type'] = type
    if extra:
        querystring['extra'] = extra
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "calcx-loan-calculator.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

