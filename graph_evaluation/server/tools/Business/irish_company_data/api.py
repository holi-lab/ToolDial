import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def search(address: str='clontibret', company_num: int=389820, company_name: str='ryanair', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Here, you can search by either company name, company id, or company address in Ireland. This is sourced directly from government data, and ideal for KYB purposes."
    
    """
    url = f"https://irish-company-data3.p.rapidapi.com/default/IE/cws/companies"
    querystring = {}
    if address:
        querystring['address'] = address
    if company_num:
        querystring['company_num'] = company_num
    if company_name:
        querystring['company_name'] = company_name
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "irish-company-data3.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

