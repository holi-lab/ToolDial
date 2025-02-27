import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_list_of_trending_pages(time_filter: str, service_filter: str='facebook', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "A real-time feed of the latest 20 pages that are growing in popularity on the Acrostic.me website."
    time_filter: Show urls for a specific time range, starting from yesterday going back a number of days equal to the specified period. Valid values are day, week, month with the default value of month.
        service_filter: Show urls that were shared or clicked to/from a specific service.
        
    """
    url = f"https://acrosticme-acrosticme-trending.p.rapidapi.com/trending/{time_filter}/{service_filter}"
    querystring = {}
    if service_filter:
        querystring['service_filter'] = service_filter
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "acrosticme-acrosticme-trending.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

