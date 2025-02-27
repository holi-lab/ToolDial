import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def testtest1(aaa5: str='aaa', ddd2: str='ddd', ccc4: str='bbb', bbb3: str='ccc', eee1: str='eee', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This is test the test endpoint for Rakuten support.
		This is test the test endpoint for Rakuten support."
    
    """
    url = f"https://testchinen.p.rapidapi.com/"
    querystring = {}
    if aaa5:
        querystring['aaa5'] = aaa5
    if ddd2:
        querystring['ddd2'] = ddd2
    if ccc4:
        querystring['ccc4'] = ccc4
    if bbb3:
        querystring['bbb3'] = bbb3
    if eee1:
        querystring['eee1'] = eee1
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "testchinen.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def testtest0(param1: str='param1', param4: str='param3', param2: str=None, param3: str='param2', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This is test the test endpoint for Rakuten support.
		This is test the test endpoint for Rakuten support."
    
    """
    url = f"https://testchinen.p.rapidapi.com/"
    querystring = {}
    if param1:
        querystring['param1'] = param1
    if param4:
        querystring['param4'] = param4
    if param2:
        querystring['param2'] = param2
    if param3:
        querystring['param3'] = param3
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "testchinen.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

