import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def statistics_from_users_choices(type: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The winning"
    type: 'type' is the statistic type you want to query. Now only supports two: 'take5max' and 'numbersmax'. Also provide the real result from NY Lottery for each day. You can setup 'type' as "take5max_real" or "numbersmax_real" to query the numbers which win most times, it is being updated everyday
        
    """
    url = f"https://ychen11-test2.p.rapidapi.com/statis"
    querystring = {'type': type, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ychen11-test2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def ny_lottery_picker(category: str, param1: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "request lottery numbers"
    category: category of ny lottery, ex. take5
        param1: parameter, ex, in "take 5", this parameter can be 5 as the round number
        
    """
    url = f"https://ychen11-test2.p.rapidapi.com/query"
    querystring = {'category': category, 'param1': param1, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ychen11-test2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

