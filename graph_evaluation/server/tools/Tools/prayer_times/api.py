import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def calendar_by_address(address: str, year: int, month: int, method: int=None, school: int=None, latitudeadjustmentmethod: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a prayer times calendar for a month by address"
    year: 4 digit year - example 2017
        month: 2 digit month, example 03 for March
        method: Any of the prayer time calculation methods specified on https://aladhan.com/calculation-methods
        school: 1 for Hanfi. 0 for all others, including, Shafi, Hanbali, etc.
        latitudeadjustmentmethod: Method for adjusting times higher latitudes - for instance, if you are checking timings in the UK or Sweden. 1 - Middle of the Night 2 - One Seventh 3 - Angle Based
        
    """
    url = f"https://aladhan.p.rapidapi.com/calendarByAddress"
    querystring = {'address': address, 'year': year, 'month': month, }
    if method:
        querystring['method'] = method
    if school:
        querystring['school'] = school
    if latitudeadjustmentmethod:
        querystring['latitudeAdjustmentMethod'] = latitudeadjustmentmethod
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aladhan.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

