import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def healthcontroller_check(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://flight-fastest-booking.p.rapidapi.com/health"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "flight-fastest-booking.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def appcontroller_flightoffer(is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://flight-fastest-booking.p.rapidapi.com/search/{is_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "flight-fastest-booking.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def appcontroller_flightsearch(adults: int, to: str, departuredate: str, is_from: str, includedairlinecodes: str='[]', nonstop: bool=None, infants: int=None, children: int=None, currency: str='NGN', excludedairlinecodes: str='[]', provider: str='AMADEUS', travelclass: str='ECONOMY', limit: int=5, returndate: str='2021-12-15', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://flight-fastest-booking.p.rapidapi.com/search"
    querystring = {'adults': adults, 'to': to, 'departureDate': departuredate, 'from': is_from, }
    if includedairlinecodes:
        querystring['includedAirlineCodes'] = includedairlinecodes
    if nonstop:
        querystring['nonStop'] = nonstop
    if infants:
        querystring['infants'] = infants
    if children:
        querystring['children'] = children
    if currency:
        querystring['currency'] = currency
    if excludedairlinecodes:
        querystring['excludedAirlineCodes'] = excludedairlinecodes
    if provider:
        querystring['provider'] = provider
    if travelclass:
        querystring['travelClass'] = travelclass
    if limit:
        querystring['limit'] = limit
    if returndate:
        querystring['returnDate'] = returndate
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "flight-fastest-booking.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def appcontroller_flightinfo(flightnumber: str, departure: str, airline: str, provider: str='AMADEUS', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://flight-fastest-booking.p.rapidapi.com/flight-info"
    querystring = {'flightNumber': flightnumber, 'departure': departure, 'airline': airline, }
    if provider:
        querystring['provider'] = provider
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "flight-fastest-booking.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def appcontroller_flightcheckinlinks(airlinecode: str, language: str='en-GB', provider: str='AMADEUS', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://flight-fastest-booking.p.rapidapi.com/flight-check-in-links"
    querystring = {'airlineCode': airlinecode, }
    if language:
        querystring['language'] = language
    if provider:
        querystring['provider'] = provider
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "flight-fastest-booking.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def appcontroller_flightbooking(gdsorderid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://flight-fastest-booking.p.rapidapi.com/gds-flight-order/{gdsorderid}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "flight-fastest-booking.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

