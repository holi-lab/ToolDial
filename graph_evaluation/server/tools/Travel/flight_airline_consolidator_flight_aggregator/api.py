import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def flight_aggregator(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "FlightsLogic Global Airfare APIs connect you to the finest information in the travel industry. You can now build, innovate and launch your web and Mobile applications with our Flight API. Inspirational search features combine our Global Airfare APIs with your existing travel offering to complement your product range."
    
    """
    url = f"https://flight-airline-consolidator-flight-aggregator.p.rapidapi.comhttps://flightslogic.com"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "flight-airline-consolidator-flight-aggregator.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

