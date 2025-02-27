import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def tz_lookup_by_location(lat: int, lng: int, timestamp: int=1589932800, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The Stadia TZ Lookup API provides time zone information, as well as information about any special offset (such as DST) in effect based on the latest IANA TZDB. Note that this API may not be accurate for timestamps in the past and does not claim to report precise nautical times in the open ocean beyond territorial waters."
    lat: The latitude component of any point on land.
        lng: The longitude component of any point on land.
        timestamp: The UNIX timestamp at which the UTC and DST offsets will be calculated. This defaults to the present time. This endpoint is not necessarily guaranteed to be accurate for timestamps that occurred in the past. Time zone geographic boundaries change over time, so if the point you are querying for was previously in a different time zone, historical results will not be accurate. If, however, the point has been in the same geographic time zone for a very long time (ex: America/New_York), the historical data may be accurate for 100+ years in the past (depending on how far back the IANA TZDB rules have been specified).
        
    """
    url = f"https://stadia-maps-time-zone-api.p.rapidapi.com/tz/lookup/v1"
    querystring = {'lat': lat, 'lng': lng, }
    if timestamp:
        querystring['timestamp'] = timestamp
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "stadia-maps-time-zone-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

