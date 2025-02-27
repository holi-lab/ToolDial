import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def getaddressusingget(lon: int, lat: int, apikey: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    lon: Longitude
        lat: Latitude
        apikey: Apikey
        
    """
    url = f"https://geocod.p.rapidapi.com/public/getAddress"
    querystring = {'lon': lon, 'lat': lat, 'apikey': apikey, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "geocod.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getcoordsusingget(postaladdress: str, apikey: str, zipcode: str=None, state: str=None, city: str=None, country: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    postaladdress: Postal address
        apikey: Apikey
        zipcode: Zip code
        state: State
        city: City
        country: Country
        
    """
    url = f"https://geocod.p.rapidapi.com/public/getCoords"
    querystring = {'postaladdress': postaladdress, 'apikey': apikey, }
    if zipcode:
        querystring['zipcode'] = zipcode
    if state:
        querystring['state'] = state
    if city:
        querystring['city'] = city
    if country:
        querystring['country'] = country
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "geocod.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

