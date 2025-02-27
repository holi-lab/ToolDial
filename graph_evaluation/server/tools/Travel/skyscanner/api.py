import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def hotel_reviews(hotelid: str, locale: str='en-US', currency: str='USD', market: str='US', offset: str='5', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint can be used to get all the review by the users about the hotels"
    
    """
    url = f"https://skyscanner65.p.rapidapi.com/api/v1/hotels/reviews"
    querystring = {'hotelId': hotelid, }
    if locale:
        querystring['locale'] = locale
    if currency:
        querystring['currency'] = currency
    if market:
        querystring['market'] = market
    if offset:
        querystring['offset'] = offset
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "skyscanner65.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def hotel_details(hotelid: str, entityid: str, locale: str='en-US', currency: str='USD', market: str='US', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint can be used to get the details about the hotels."
    
    """
    url = f"https://skyscanner65.p.rapidapi.com/api/v1/hotels/details"
    querystring = {'hotelId': hotelid, 'entityId': entityid, }
    if locale:
        querystring['locale'] = locale
    if currency:
        querystring['currency'] = currency
    if market:
        querystring['market'] = market
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "skyscanner65.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_cars(pickuptime: str, dropofftime: str, dropoffdate: str, pickupdate: str, pickupentityid: int, dropoffentityid: int, locale: str='en-US', currency: str='USD', driverage: int=30, market: str='US', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "-"
    
    """
    url = f"https://skyscanner65.p.rapidapi.com/api/v1/carhire/search"
    querystring = {'pickUpTime': pickuptime, 'dropOffTime': dropofftime, 'dropOffDate': dropoffdate, 'pickUpDate': pickupdate, 'pickUpEntityId': pickupentityid, 'dropOffEntityId': dropoffentityid, }
    if locale:
        querystring['locale'] = locale
    if currency:
        querystring['currency'] = currency
    if driverage:
        querystring['driverAge'] = driverage
    if market:
        querystring['market'] = market
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "skyscanner65.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def status(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "-"
    
    """
    url = f"https://skyscanner65.p.rapidapi.com/api/v1/checkStatus"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "skyscanner65.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def currencies(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "You can use the **/currencies** endpoint to retrieve the currencies that we support and information about how we format them in Skyscanner."
    
    """
    url = f"https://skyscanner65.p.rapidapi.com/api/v1/currencies"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "skyscanner65.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def markets(locale: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "You can use the **/markets** endpoint to retrieve the market countries that we support. Most suppliers (airlines, travel agents, and car hire dealers) set their fares based on the market (or country of purchase). It is therefore necessary to specify the market country in every query. The names of the markets returned are localised based on a locale passed as a parameter."
    
    """
    url = f"https://skyscanner65.p.rapidapi.com/api/v1/markets"
    querystring = {'locale': locale, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "skyscanner65.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def locales(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "You can use the **/locales** endpoint to retrieve the locales that we support to translate your content.
		The names of the locales returned are in the native language associated with the locale."
    
    """
    url = f"https://skyscanner65.p.rapidapi.com/api/v1/locales"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "skyscanner65.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def flights(locale: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The `/flights` endpoint takes a `locale` and returns a list of geographical locations in a language determined by the given **locale**."
    
    """
    url = f"https://skyscanner65.p.rapidapi.com/api/v1/flights"
    querystring = {'locale': locale, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "skyscanner65.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def carriers(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The Carriers API returns a full list of active carriers with name and IATA code indexed by their **carrierId**. The Carriers API returns a full list of active carriers with name and IATA code indexed by their **carrierId**."
    
    """
    url = f"https://skyscanner65.p.rapidapi.com/api/v1/carriers"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "skyscanner65.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

