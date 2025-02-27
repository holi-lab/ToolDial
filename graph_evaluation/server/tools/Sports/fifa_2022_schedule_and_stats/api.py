import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_matches_on_a_specific_date(date: str, utc_offset: int=8, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Parameters:
		1.  date - Date to query the matches
		2. utc_offset - Change the utc offset of the date to suit your timezone.
		
		Returns the matches on the specified date and timezone.
		Works for both historical, live, and future dates.
		Match data consists of date, time, home team, away team, and many more, see the example response for an example."
    utc_offset: UTC Offset for timezone (Must be between -12 to 14)
        
    """
    url = f"https://fifa-2022-schedule-and-stats.p.rapidapi.com/schedule"
    querystring = {'date': date, }
    if utc_offset:
        querystring['utc_offset'] = utc_offset
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "fifa-2022-schedule-and-stats.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

