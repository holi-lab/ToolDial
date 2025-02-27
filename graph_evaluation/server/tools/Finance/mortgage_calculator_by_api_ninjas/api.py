import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def v1_mortgagecalculator(interest_rate: int, downpayment: int=None, home_value: int=None, monthly_hoa: int=None, annual_property_tax: str=None, duration_years: int=None, loan_amount: int=200000, annual_home_insurance: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "API Ninjas Mortgage Calculator API endpoint. Either **loan_amount** or (**home_value** + **downpayment**) parameters must be set."
    interest_rate: annual interest rate (in %). For example, a 3.5% interest rate would be 3.5. Cannot exceed 10000.
        downpayment: downpayment on the home or asset. Cannot exceed home_value.
        home_value: total value of the home or asset. Must be greater than downpayment.
        monthly_hoa: monthly homeowner association fees.
        annual_property_tax: annual property tax owed.
        duration_years: duration of the loan in years. Must be between 1 and 10000. If not set, default value is 30 years.
        loan_amount: principle loan amount.
        annual_home_insurance: annual homeowner's insurance bill.
        
    """
    url = f"https://mortgage-calculator-by-api-ninjas.p.rapidapi.com/v1/mortgagecalculator"
    querystring = {'interest_rate': interest_rate, }
    if downpayment:
        querystring['downpayment'] = downpayment
    if home_value:
        querystring['home_value'] = home_value
    if monthly_hoa:
        querystring['monthly_hoa'] = monthly_hoa
    if annual_property_tax:
        querystring['annual_property_tax'] = annual_property_tax
    if duration_years:
        querystring['duration_years'] = duration_years
    if loan_amount:
        querystring['loan_amount'] = loan_amount
    if annual_home_insurance:
        querystring['annual_home_insurance'] = annual_home_insurance
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "mortgage-calculator-by-api-ninjas.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

