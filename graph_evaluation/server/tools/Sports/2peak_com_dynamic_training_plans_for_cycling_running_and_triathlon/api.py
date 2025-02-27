import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def api_v2_activities_before_2014_07_24t18_00_00z_after_2014_07_14t18_00_00z(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Activities"
    
    """
    url = f"https://sebpeak-2peak-com-dynamic-training-plans-for-cycling-running-and-triathlon-v1.p.rapidapi.com/api/v2/activities?before=2014-07-24T18:00:00Z&after=2014-07-14T18:00:00Z"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sebpeak-2peak-com-dynamic-training-plans-for-cycling-running-and-triathlon-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

