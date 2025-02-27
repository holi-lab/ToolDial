import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def getchineselunarcalendar(date: str, timezone: str='480', simplified: str='1', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Convert a solar calendar date into its corresponding Chinese lunar calendar date"
    date: A date in western calendar(Gregorian calendar), in a format of YYYYMMDD. For example, 20230325.
        timezone: The timezone of the date parameter. It only impacts the result of solar term. All other properties are all based on the solar date, regardless whichever time zone it is.

The default value is 480 , which stands for the Shanghai/Singapore/Hong Kong/Taibei timezone (UTC+8:00). 

How to set the timezone value? It is how many minutes it deviates from UTC+0:00. Suppose your timezone is UTC+5:30, so the value of the timezone parameter is 5 * 60 + 30 = 330.
        simplified: Indicate whether the returned results should be in Simplified Chinese which people in mainland China use, or Traditional Chinese which HK and Taiwan people use.

The default value is 1. 1 stand for Simplified Chinese, 0 stands for Traditional Chinese.

        
    """
    url = f"https://chinese-lunar-calendar.p.rapidapi.com/"
    querystring = {'date': date, }
    if timezone:
        querystring['timezone'] = timezone
    if simplified:
        querystring['simplified'] = simplified
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "chinese-lunar-calendar.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

