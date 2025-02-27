import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def date_info(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "/date_info/"gmt_offset": Returns the current date information, including the timestamp, month, year, day, weekday, week, hour, minute, second, and number of days in the month, based on the specified GMT offset (in hours)."
    
    """
    url = f"https://gmt-timestamp-and-date-information-api.p.rapidapi.com/date_info/0"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "gmt-timestamp-and-date-information-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def timestamp(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "/timestamp/"gmt_offset": Returns the current timestamp based on the specified GMT offset (in hours)."
    
    """
    url = f"https://gmt-timestamp-and-date-information-api.p.rapidapi.com/timestamp/0"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "gmt-timestamp-and-date-information-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

