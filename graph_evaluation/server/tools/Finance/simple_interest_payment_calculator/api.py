import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def paymentcalculatorsimple_calculate(downpayment: int, loantermmonths: int, interestrate: int, purchaseprice: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://simple-interest-payment-calculator.p.rapidapi.com/PaymentCalculatorSimple/Calculate"
    querystring = {'downPayment': downpayment, 'loanTermMonths': loantermmonths, 'interestRate': interestrate, 'purchasePrice': purchaseprice, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "simple-interest-payment-calculator.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def paymentcalculatorsimple_calculate(downpayment: int, loantermmonths: int, interestrate: int, purchaseprice: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://simple-interest-payment-calculator.p.rapidapi.com/PaymentCalculatorSimple/Calculate"
    querystring = {'downPayment': downpayment, 'loanTermMonths': loantermmonths, 'interestRate': interestrate, 'purchasePrice': purchaseprice, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "simple-interest-payment-calculator.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

