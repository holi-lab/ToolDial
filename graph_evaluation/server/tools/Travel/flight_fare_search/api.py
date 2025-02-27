import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def flight_search_v2(to: str, date: str, is_from: str, adult: int, currency: str='USD', type: str='economy', child: str=None, infant: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "A faster, more agile Endpoint that's used to search flights."
    
    """
    url = f"https://flight-fare-search.p.rapidapi.com/v2/flights/"
    querystring = {'to': to, 'date': date, 'from': is_from, 'adult': adult, }
    if currency:
        querystring['currency'] = currency
    if type:
        querystring['type'] = type
    if child:
        querystring['child'] = child
    if infant:
        querystring['infant'] = infant
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "flight-fare-search.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def airport_arrivals(airportcode: str, carriercode: str=None, date: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "An Endpoint to fetch Arrivals on a given date"
    
    """
    url = f"https://flight-fare-search.p.rapidapi.com/v2/airport/arrivals"
    querystring = {'airportCode': airportcode, }
    if carriercode:
        querystring['carrierCode'] = carriercode
    if date:
        querystring['date'] = date
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "flight-fare-search.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def airport_departues(airportcode: str, carriercode: str=None, date: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "An endpoint to get Departues in an airport"
    
    """
    url = f"https://flight-fare-search.p.rapidapi.com/v2/airport/departures"
    querystring = {'airportCode': airportcode, }
    if carriercode:
        querystring['carrierCode'] = carriercode
    if date:
        querystring['date'] = date
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "flight-fare-search.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def airport_search(query: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "An endpoint to search airports"
    
    """
    url = f"https://flight-fare-search.p.rapidapi.com/v2/airport/search"
    querystring = {'query': query, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "flight-fare-search.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

