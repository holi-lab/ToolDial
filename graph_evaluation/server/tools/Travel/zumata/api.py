import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def hotels(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    ""
    
    """
    url = f"https://community-zumata.p.rapidapi.com/hotels"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "community-zumata.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def healthcheck(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    ""
    
    """
    url = f"https://community-zumata.p.rapidapi.com/healthchekc"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "community-zumata.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def hotels_minimal(destinationid: str=None, checkindatetime: str=None, checkoutdatetime: str=None, lang: str=None, roomcount: str=None, adultcount: str=None, sessionid: str=None, city: str=None, countrycode: str=None, stateorprovincecode: str=None, gzip: str=None, production: str=None, hotelid: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    ""
    
    """
    url = f"https://community-zumata.p.rapidapi.com/hotels_minimal"
    querystring = {}
    if destinationid:
        querystring['destinationId'] = destinationid
    if checkindatetime:
        querystring['checkInDateTime'] = checkindatetime
    if checkoutdatetime:
        querystring['checkOutDateTime'] = checkoutdatetime
    if lang:
        querystring['lang'] = lang
    if roomcount:
        querystring['roomCount'] = roomcount
    if adultcount:
        querystring['adultCount'] = adultcount
    if sessionid:
        querystring['sessionId'] = sessionid
    if city:
        querystring['city'] = city
    if countrycode:
        querystring['countryCode'] = countrycode
    if stateorprovincecode:
        querystring['stateOrProvinceCode'] = stateorprovincecode
    if gzip:
        querystring['gzip'] = gzip
    if production:
        querystring['production'] = production
    if hotelid:
        querystring['hotelId'] = hotelid
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "community-zumata.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def hotels_dynamic(destination_id: str=None, checkindatetime: str=None, checkoutdatetime: str=None, lang: str=None, roomcount: str=None, adultcount: str=None, sessionid: str=None, city: str=None, countrycode: str=None, stateorprovincecode: str=None, gzip: str=None, production: str=None, hotelid: str=None, max_wait: str=None, requested_currency: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    ""
    
    """
    url = f"https://community-zumata.p.rapidapi.com/hotels_dynamic"
    querystring = {}
    if destination_id:
        querystring['destination_id'] = destination_id
    if checkindatetime:
        querystring['checkInDateTime'] = checkindatetime
    if checkoutdatetime:
        querystring['checkOutDateTime'] = checkoutdatetime
    if lang:
        querystring['lang'] = lang
    if roomcount:
        querystring['roomCount'] = roomcount
    if adultcount:
        querystring['adultCount'] = adultcount
    if sessionid:
        querystring['sessionId'] = sessionid
    if city:
        querystring['city'] = city
    if countrycode:
        querystring['countryCode'] = countrycode
    if stateorprovincecode:
        querystring['stateOrProvinceCode'] = stateorprovincecode
    if gzip:
        querystring['gzip'] = gzip
    if production:
        querystring['production'] = production
    if hotelid:
        querystring['hotelId[]'] = hotelid
    if max_wait:
        querystring['max_wait'] = max_wait
    if requested_currency:
        querystring['requested_currency'] = requested_currency
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "community-zumata.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

