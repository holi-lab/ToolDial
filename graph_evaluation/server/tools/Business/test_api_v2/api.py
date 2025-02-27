import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def api(apistatus: str=None, apistatuslist: str=None, backendid: str=None, applicationid: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "12"
    
    """
    url = f"https://test_api87.p.rapidapi.com/application/x-www-form-urlencoded"
    querystring = {}
    if apistatus:
        querystring['apiStatus'] = apistatus
    if apistatuslist:
        querystring['apiStatusList'] = apistatuslist
    if backendid:
        querystring['backEndId'] = backendid
    if applicationid:
        querystring['applicationId'] = applicationid
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "test_api87.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

