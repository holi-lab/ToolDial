import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def time_series_single_maturity(strip: str, startdate: str=None, enddate: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "single maturity for a date range"
    strip: one of py_3m, py_4m, py_6m, py_9m, py_1y, py_2y, py_5y, py_7y, py_10y, py_15y, py_30y
        startdate: yyyy-mm-dd
        enddate: yyyy-mm-dd
        
    """
    url = f"https://eur-risk-free-rate.p.rapidapi.com/euro/curve/single"
    querystring = {'strip': strip, }
    if startdate:
        querystring['startdate'] = startdate
    if enddate:
        querystring['enddate'] = enddate
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "eur-risk-free-rate.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def time_series_yield_curve(startdate: str=None, enddate: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "yield curve for a date range"
    startdate: yyyy-mm-dd
        enddate: yyyy-mm-dd
        
    """
    url = f"https://eur-risk-free-rate.p.rapidapi.com/euro/curve"
    querystring = {}
    if startdate:
        querystring['startdate'] = startdate
    if enddate:
        querystring['enddate'] = enddate
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "eur-risk-free-rate.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

