import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def v1_holidays(year: int, country: str, type: str='federal_holiday', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "API Ninjas Holidays endpoint"
    year: Calendar year between 2010 and 2030 (inclusive). Note: not all countries are guaranteed contain data going back to 2010.
        country: Country name or ISO 3166-2 country code (preferred).
        type: Holiday type filter. Possible values are:

- public_holiday
- observance
- national_holiday
- season
- state_holiday
- optional_holiday
- clock_change_daylight_saving_time
- local_holiday
- united_nations_observance
- observance_christian
- bank_holiday
- common_local_holiday
- national_holiday_christian
- christian
- observance_hebrew
- jewish_holiday
- muslim
- hindu_holiday
- restricted_holiday
- official_holiday
- national_holiday_orthodox
- local_observance
- 
        
    """
    url = f"https://holidays-by-api-ninjas.p.rapidapi.com/v1/holidays"
    querystring = {'year': year, 'country': country, }
    if type:
        querystring['type'] = type
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "holidays-by-api-ninjas.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

