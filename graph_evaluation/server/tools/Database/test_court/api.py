import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def test_court_get(testcases: int=100, testname: str='case', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Test court to get"
    
    """
    url = f"https://test-court.p.rapidapi.com/"
    querystring = {}
    if testcases:
        querystring['testcases'] = testcases
    if testname:
        querystring['testname'] = testname
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "test-court.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

