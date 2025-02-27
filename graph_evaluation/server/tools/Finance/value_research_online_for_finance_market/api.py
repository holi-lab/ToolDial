import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def funds_fundvsindex_asp(sch: int, obj: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    ""
    sch: Scheme Code
        obj: Object Type Like equity or debt
        
    """
    url = f"https://valueresearchonline.p.rapidapi.com/funds/fundVSindex.asp"
    querystring = {'Sch': sch, }
    if obj:
        querystring['obj'] = obj
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "valueresearchonline.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def mutual_funds(x_mashape_key: str, sch: int, amt: int=None, freq: str=None, startdt: str=None, enddt: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get SIP returns on Mutual Funds"
    sch: Scheme Code of MF
        amt: SIP Amount
        freq: Frequency Of SIP Amount
        startdt: Start Date
        enddt: End Date
        
    """
    url = f"https://valueresearchonline.p.rapidapi.com/funds/fundSIPReturn.asp"
    querystring = {'X-Mashape-Key': x_mashape_key, 'sch': sch, }
    if amt:
        querystring['amt'] = amt
    if freq:
        querystring['freq'] = freq
    if startdt:
        querystring['startDt'] = startdt
    if enddt:
        querystring['endDt'] = enddt
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "valueresearchonline.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

