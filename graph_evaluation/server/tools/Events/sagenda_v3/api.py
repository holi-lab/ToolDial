import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def list(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "A list of bookable items associated with supplied authentication token (account)"
    
    """
    url = f"https://sagenda-sagenda-v3-v1.p.rapidapi.com/api/v3/bookableItems"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sagenda-sagenda-v3-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def authentication_status(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "You can use this endpoint to check whether your authentication token is valid. Responds with HTTP 200, API version and list of available claims for a token (for reference)"
    
    """
    url = f"https://sagenda-sagenda-v3-v1.p.rapidapi.com/api/v3/status/oauth"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sagenda-sagenda-v3-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def api_status(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Receive an API status. Responds with HTTP 200 OK and API version in body if functional"
    
    """
    url = f"https://sagenda-sagenda-v3-v1.p.rapidapi.com/api/v3/status"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sagenda-sagenda-v3-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_specific_event_information(eventidentifier: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a single event object containing all the booking information."
    eventidentifier: Unique event identifier from booking or event search
        
    """
    url = f"https://sagenda-sagenda-v3-v1.p.rapidapi.com/api/v3/bookings/{eventidentifier}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sagenda-sagenda-v3-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def events_for_item_by_date_range(bookableitem: str, startdate: str, enddate: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns an array of events that start in the date range between startDate and endDate (inclusive) for given bookable item"
    bookableitem: Bookable item identifier
        startdate: Start of date range (inclusive). Format: YYYY-MM-DD (e.g 2018-03-17)
        enddate: End of date range (inclusive). Format: YYYY-MM-DD (e.g 2018-03-17). Format: YYYY-MM-DD (e.g 2018-03-17)
        
    """
    url = f"https://sagenda-sagenda-v3-v1.p.rapidapi.com/api/v3/events/{startdate}/{enddate}/{bookableitem}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sagenda-sagenda-v3-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_bookings_information(fromdate: str, todate: str, itemid: str=None, status: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint lets you request detailed booking information for a given period of time, optionally filtered by bookable item and status."
    fromdate: Inclusive request range start date 
        todate: Inclusive request range end date
        itemid: Filter bookings by bookable item ID
        status: Filter bookings by status
        
    """
    url = f"https://sagenda-sagenda-v3-v1.p.rapidapi.com/api/v3/bookings/{fromdate}/{todate}"
    querystring = {}
    if itemid:
        querystring['itemId'] = itemid
    if status:
        querystring['status'] = status
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sagenda-sagenda-v3-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

