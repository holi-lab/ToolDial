import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def multiple_zip_codes_to_location_information(zipcodes: str, units: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns location information for multiple zip codes (up to 100).  This information includes city, state, latitude, longitude, and time zone information.  It also contains a list of other acceptable city names for the locations.  **Each zip code provided will count as a separate request.** For example, if you send 5 zip codes, you will be charged for 5 requests."
    zipcodes: Zip Codes (Comma separated) - 100 Max
        units: Units: degrees or radians
        
    """
    url = f"https://redline-redline-zipcode.p.rapidapi.com/rest/multi-info.json/{zipcodes}/{units}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "redline-redline-zipcode.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def state_to_zip_codes(state: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all zip codes for a state.  **Each 10 zip codes returned are charged as separate request.** For example, if the state you select returns 200 zip codes, you will be charged for 20 requests."
    state: State Abbreviation (e.g. RI)
        
    """
    url = f"https://redline-redline-zipcode.p.rapidapi.com/rest/state-zips.json/{state}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "redline-redline-zipcode.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

