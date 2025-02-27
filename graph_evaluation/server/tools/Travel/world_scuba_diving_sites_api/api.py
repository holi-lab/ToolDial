import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def query_divesites_by_a_country_or_a_region(country: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The endpoint returns a json list of dive sites which match the region or country entered as the query. The range of results depend but there is usually around 100-500 sites for each country. There are around 15'000 dive sites listed in the database."
    
    """
    url = f"https://world-scuba-diving-sites-api.p.rapidapi.com/api/divesite"
    querystring = {'country': country, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "world-scuba-diving-sites-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def query_divesites_by_gps_boundaries_for_use_with_maps(southwestlat: str='50.995577266225524', northeastlng: str='3.827225290533761', southwestlng: str='-12.542403615716239', northeastlat: str='58.59328356952258', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The API endpoint that queries dive sites by GPS boundaries allows developers to search for dive sites within a specified geographical region, defined by a set of latitude and longitude coordinates. This endpoint can be particularly useful for creating interactive maps or applications that require displaying dive sites within a specific area. To use it add the four map bounds as query params"
    
    """
    url = f"https://world-scuba-diving-sites-api.p.rapidapi.com/api/divesite/gps"
    querystring = {}
    if southwestlat:
        querystring['southWestLat'] = southwestlat
    if northeastlng:
        querystring['northEastLng'] = northeastlng
    if southwestlng:
        querystring['southWestLng'] = southwestlng
    if northeastlat:
        querystring['northEastLat'] = northeastlat
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "world-scuba-diving-sites-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

