import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def search_sex_offenders(lastname: str=None, zipcode: str=None, state: str='California', city: str=None, firstname: str='David', lat: int=37, radius: int=1, lng: int=-122, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Support 2 types of searches, GIS search (lat, lng, radius) or attribute search (name, city, state, zipcode)
		
		**GIS Search:**
		Params:
		@lat: latitude
		@lng: longitude
		@radius: search radius in Miles (Mi), e.g. radius = 0.2 means the API will search for sex offenders within 0.2 miles of the given lat and lng
		
		You can supply additional query params to filter down your results, supported additional params:
		@firstName, @lastName, @city, @state, @zipcode
		
		**Attribute Search:**
		Params:
		@firstName, @lastName, @city, @state, @zipcode
		
		You can combine them however you like, at least one of the above params is required for the API to response, and we will return only results that matches all your provided query params.
		
		**Response:**
		@offenders: List of Offender Object; If no offenders are found, it will return empty list `offenders: []`.
		
		Extensive database of National Registered Sex Offenders API for the United States. This API covers 750k+ offender records, 1M+ crime records, 19k+ cities, and all 50 states. Supports lat/lng search with radius."
    
    """
    url = f"https://sex-offenders.p.rapidapi.com/sexoffender"
    querystring = {}
    if lastname:
        querystring['lastName'] = lastname
    if zipcode:
        querystring['zipcode'] = zipcode
    if state:
        querystring['state'] = state
    if city:
        querystring['city'] = city
    if firstname:
        querystring['firstName'] = firstname
    if lat:
        querystring['lat'] = lat
    if radius:
        querystring['radius'] = radius
    if lng:
        querystring['lng'] = lng
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sex-offenders.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

