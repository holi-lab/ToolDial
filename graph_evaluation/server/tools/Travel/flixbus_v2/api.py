import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def search_trips(date: str, to_id: str, adult: int, from_id: str, locale: str=None, bikes: int=0, children: int=0, currency: str='EUR', search_by: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search trips between two cities or stations"
    date: Travel date in format DD.MM.YYYY
        to_id: Flixbus id of the arrival city or station
        adult: Number of adult travellers (from 15 years)
        from_id: Flixbus id of the departure city or station
        locale: Locale of returned results
        bikes: If travelling with bikes
        children: Children travelling (0-14 years)
        currency: Currency of fares returned
        search_by: Search by city ids or stations ids (default: cities)
        
    """
    url = f"https://flixbus2.p.rapidapi.com/trips"
    querystring = {'date': date, 'to_id': to_id, 'adult': adult, 'from_id': from_id, }
    if locale:
        querystring['locale'] = locale
    if bikes:
        querystring['bikes'] = bikes
    if children:
        querystring['children'] = children
    if currency:
        querystring['currency'] = currency
    if search_by:
        querystring['search_by'] = search_by
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "flixbus2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_timetable(station_id: str, date: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns timetable for a given station id and date"
    station_id: Id of flixbus station
        date: Date in format DD.MM.YYYY
        
    """
    url = f"https://flixbus2.p.rapidapi.com/schedule"
    querystring = {'station_id': station_id, 'date': date, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "flixbus2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def autocomplete(query: str, locale: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search for Flixbus stations"
    query: query string
        locale: language of query and results
        
    """
    url = f"https://flixbus2.p.rapidapi.com/autocomplete"
    querystring = {'query': query, }
    if locale:
        querystring['locale'] = locale
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "flixbus2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

