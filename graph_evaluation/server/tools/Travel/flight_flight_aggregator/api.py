import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def flight(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "FlightsLogic provides Flight API, Airline Consolidator, Flight Aggregator to the travel agents, tour operators and travel companies worldwide."
    
    """
    url = f"https://flight-flight-aggregator.p.rapidapi.comhttps://flightslogic.com/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "flight-flight-aggregator.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

