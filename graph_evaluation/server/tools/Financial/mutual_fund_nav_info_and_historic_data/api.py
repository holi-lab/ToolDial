import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_latest_business_news(fromindex: str='2', limit: str='10', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Latest Business News"
    fromindex: Start From this Index
        limit: Limit size of News array
        
    """
    url = f"https://mutual-fund-info.p.rapidapi.com/ri/v1/investment/scheme/news"
    querystring = {}
    if fromindex:
        querystring['fromIndex'] = fromindex
    if limit:
        querystring['limit'] = limit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "mutual-fund-info.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_scheme_nav_using_schemecode_and_date_single_schemecode_supported(date: str, schemecode: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Scheme NAV using schemeCode(AMFI Scheme Code) and date - single schemeCode supported"
    date: Date in dd-MMM-YYYY format
        
    """
    url = f"https://mutual-fund-info.p.rapidapi.com/ri/v1/investment/scheme/{schemecode}"
    querystring = {'date': date, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "mutual-fund-info.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

