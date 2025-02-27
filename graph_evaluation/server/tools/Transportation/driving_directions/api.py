import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_directions(origin: str, destination: str, avoid_routes: str='tolls,ferries', country: str='us', arrival_time: int=None, language: str='en', departure_time: int=None, distance_units: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get driving directions from an origin to a destination."
    origin: Directions origin / starting point - specified as free-text location query or latitude, longitude pair.

**`e.g.`** *`Church St & 29th St, San-Francisco, CA, USA`*
**`e.g.`** *`37.743292, -122.420437`*
        destination: Directions destination - specified as free-text location query or latitude, longitude pair.

**`e.g.`** *`145 5th Ave, New York, NY, USA`*
**`e.g.`** *`40.649238, -73.974229`*
        avoid_routes: Route types to avoid - specified as a comma delimited list of the following values: `tolls`, `highways`, `ferries`.

**`e.g.`** *`ferries,tolls`*
**`e.g.`** *`highways`*
        country: The country/region to use for the query, specified as a 2-letter country code - see [ISO 3166-1 alpha-2](https://en.wikipedia.org/wiki/ISO_3166-1_alpha-2).

**Default**: `us`.
        arrival_time: Arrival time - specified as a Unix-Timestamp in **destination local time** (e.g. *1665752460*).
        language: The language to use for the query, specified as a 2-letter language code - see [ISO 639-1 alpha-2](https://en.wikipedia.org/wiki/List_of_ISO_639-1_codes).

**Default**: `en`.
        departure_time: Departure time - specified as a Unix-Timestamp in **origin local time** (e.g. *1665785617*).
        distance_units: Set the distance units in the response to Kilometers or Miles - accepts one of the following values: `auto`, `km`, `mi`. The default `auto` value will use the common distance units used in origin/destination (e.g. Miles in the US, Kilometers in Germany).

**Default:** `auto`.
        
    """
    url = f"https://driving-directions1.p.rapidapi.com/get-directions"
    querystring = {'origin': origin, 'destination': destination, }
    if avoid_routes:
        querystring['avoid_routes'] = avoid_routes
    if country:
        querystring['country'] = country
    if arrival_time:
        querystring['arrival_time'] = arrival_time
    if language:
        querystring['language'] = language
    if departure_time:
        querystring['departure_time'] = departure_time
    if distance_units:
        querystring['distance_units'] = distance_units
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "driving-directions1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

