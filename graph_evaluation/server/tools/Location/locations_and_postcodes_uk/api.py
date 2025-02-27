import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def getdistancebetweenpostcodes(postcodesource: str, postcodedestination: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Calculate the distance between two postcodes in the UK. It will return the distance in meters and miles."
    
    """
    url = f"https://locations-and-postcodes-uk.p.rapidapi.com/getDistanceBetweenPostcodes"
    querystring = {'postcodesource': postcodesource, 'postcodedestination': postcodedestination, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "locations-and-postcodes-uk.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getlocationforpostcode(postcode: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Send a postcode in the UK and it will return the nearest geo coordinate"
    
    """
    url = f"https://locations-and-postcodes-uk.p.rapidapi.com/getLocationForPostcode"
    querystring = {'postcode': postcode, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "locations-and-postcodes-uk.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getpostcodeforlocation(longitude: int, latitude: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Send a latitude and a longitude in the UK and it will return the nearest postcode"
    
    """
    url = f"https://locations-and-postcodes-uk.p.rapidapi.com/getPostcodeForLocation"
    querystring = {'longitude': longitude, 'latitude': latitude, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "locations-and-postcodes-uk.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

