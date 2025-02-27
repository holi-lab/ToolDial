import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def flightslogic_flight_api(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "FlightsLogic provides Flight API integration within your new or existing online travel website. Flight API integration connects travel agents and customers across the globe, extensively broadening your market reach. We deliver Flight API Integration Solution- one of the most reliable and most trusted Global Distribution Systems (GDS), that helps the travel portals get the best service suppliers by their side to broadcast their effective services at the portal and that successively can bring vast website traffic as well."
    
    """
    url = f"https://flightslogic-flight.p.rapidapi.comhttps://www.flightslogic.com/flight-api.php"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "flightslogic-flight.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

