import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def income_statement(stock: str, apikey: str, period: str='quarter', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Company Income Statement In JSON format"
    apikey: get your apikey at https://fmpcloud.io/register
        period: annual / quarter
        
    """
    url = f"https://fmpcloud.p.rapidapi.com/income-statement/{stock}"
    querystring = {'apikey': apikey, }
    if period:
        querystring['period'] = period
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "fmpcloud.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def balance_sheet_statement(apikey: str, stock: str, period: str='quarter', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Companies Balance Sheet Statement in JSON Format"
    apikey: get your apikey at https://fmpcloud.io/register
        period: quarter / annual
        
    """
    url = f"https://fmpcloud.p.rapidapi.com/balance-sheet-statement/{stock}"
    querystring = {'apikey': apikey, }
    if period:
        querystring['period'] = period
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "fmpcloud.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def cash_flow_statement(stock: str, apikey: str, period: str='quarter', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Cash Flow Statement in JSON Format"
    apikey: get your apikey at https://fmpcloud.io/register
        period: period / annual
        
    """
    url = f"https://fmpcloud.p.rapidapi.com/cash-flow-statement/{stock}"
    querystring = {'apikey': apikey, }
    if period:
        querystring['period'] = period
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "fmpcloud.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

