import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_values(key: str, skip: int=100, sortbyorder: str='desc', limit: int=50, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a list of all **unique** values in the dataset for each of the properties. Results can be sorted and paginated."
    key: Values will be returned for this property. Possible values are: icao, iata, name, city, subd, country, elevation, lat, lon, tz and lid.
        skip: Skip n results
        sortbyorder: Specify the sorting order. Possible values are *asc* and *desc*. Results are sorted in ascending order by default. 
        limit: Limit to n results.
        
    """
    url = f"https://airportsdata.p.rapidapi.com/airports/values"
    querystring = {'key': key, }
    if skip:
        querystring['skip'] = skip
    if sortbyorder:
        querystring['sortByOrder'] = sortbyorder
    if limit:
        querystring['limit'] = limit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "airportsdata.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_airports(lid: str=None, sortby: str='iata', name: str='Ishigaki,Airport', limit: int=50, elevationover: int=50, country: str='JP', city: str='Ishigaki', skip: int=0, sortbyorder: str='desc', elevationunder: int=2000, tz: str='Asia/Tokyo', icao: str='ROIG', subd: str='Okinawa', iata: str='ISG', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Will return data from 28k+ airports in the database."
    lid: U.S. FAA Location Identifier (12,566 entries), or an empty string
        sortby: Sort results by a property value. Possible values are: IATA, IACO, name, city, subd, country, tz, lid and elevation. 
        name: Filter results by name. The value should be sent as comma-separated.  For example, for *Zabrat Airport*, send *Zabrat,Airport*
        limit: Limit to n results.
        elevationover: Filter airports with an MSL elevation, or highest point of the landing area in feet, over this value.
        country:  ISO 3166-1 alpha-2 country code (plus XK for Kosovo).
        city: Filter results by city. The value should be sent as comma-separated.  For example, for *Zakynthos Island*, send *Zakynthos,Island*
        skip: Skip first n results. 
        sortbyorder: Specify the sorting order. Possible values are *asc* and *desc*. Results are sorted in ascending order by default. 
        elevationunder: Filter airports with an MSL elevation, or highest point of the landing area in feet, under this value.
        tz: Timezone expressed as a tz database name (IANA-compliant) or an empty string for Antarctica.
        subd: Filter results by subdivision (e.g. state, province, region, etc.). The value should be sent as comma-separated.  For example, for *Newfoundland and Labrador*, send *Newfoundland,and,Labrador*
        
    """
    url = f"https://airportsdata.p.rapidapi.com/airports"
    querystring = {}
    if lid:
        querystring['lid'] = lid
    if sortby:
        querystring['sortBy'] = sortby
    if name:
        querystring['name'] = name
    if limit:
        querystring['limit'] = limit
    if elevationover:
        querystring['elevationOver'] = elevationover
    if country:
        querystring['country'] = country
    if city:
        querystring['city'] = city
    if skip:
        querystring['skip'] = skip
    if sortbyorder:
        querystring['sortByOrder'] = sortbyorder
    if elevationunder:
        querystring['elevationUnder'] = elevationunder
    if tz:
        querystring['tz'] = tz
    if icao:
        querystring['icao'] = icao
    if subd:
        querystring['subd'] = subd
    if iata:
        querystring['iata'] = iata
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "airportsdata.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

