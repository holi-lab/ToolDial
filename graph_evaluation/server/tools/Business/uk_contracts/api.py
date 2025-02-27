import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def find_contracts(include: str, sendby: str='2021-11-01', status: str='active', without: str='hardware', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "find the UK gov's contracts"
    
    """
    url = f"https://uk-contracts.p.rapidapi.com/v1/find"
    querystring = {'include': include, }
    if sendby:
        querystring['sendby'] = sendby
    if status:
        querystring['status'] = status
    if without:
        querystring['without'] = without
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "uk-contracts.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

