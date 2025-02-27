import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def monthly_payment(interestrate: int, terms: int, loanamount: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "calculate monthly payment. Add all input to the query parameter string.
		loanAmount, interestRate, and terms."
    
    """
    url = f"https://mortgage-monthly-payment-calculator.p.rapidapi.com/revotek-finance/mortgage/monthly-payment"
    querystring = {'interestRate': interestrate, 'terms': terms, 'loanAmount': loanamount, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "mortgage-monthly-payment-calculator.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

