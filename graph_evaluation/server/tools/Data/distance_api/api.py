import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def getdistancebycoordinates(coordinates: str, apikey: str, unit: str='km', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns distance in either kilometres or miles."
    coordinates: Comma separated co-ordinates in the order of fromLatitude,fromLongitude,toLatitude,toLongitude.
        apikey: Your 7x API Key. Get it from https://app.7x.ax for free.
        unit: Unit of distance. 'km' for kilometres and 'mi' for miles.
        
    """
    url = f"https://distance-api.p.rapidapi.com/bycoordinates/{coordinates}"
    querystring = {'apikey': apikey, }
    if unit:
        querystring['unit'] = unit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "distance-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

