import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_railway_period_for_utc_date(millis: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint will return railway period between:
		{"millis":"0","railwayPeriod":"1969-1970 P10"} (01/01/1970 00:00:00)
		And
		{"millis":"8640000000000","railwayPeriod":"2243-2244 P8"} (17/10/2243 00:00:00)"
    
    """
    url = f"https://railway-periods.p.rapidapi.com/millis/{millis}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "railway-periods.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

