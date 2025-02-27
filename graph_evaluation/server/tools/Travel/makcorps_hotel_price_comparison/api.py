import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def free(location: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This is a free resource for testing purposes. It will give you the list of 7 hotels with top vendors. visit makcorps.com for more info"
    
    """
    url = f"https://manthankool-makcorps-hotel-price-comparison-v1.p.rapidapi.com/free/{location}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "manthankool-makcorps-hotel-price-comparison-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def king(location: str, check_out_date: str, check_in_date: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint allows user to get hotel room prices from more than 200 websites like agoda.com,booking.com,hotels.com,etc. And compares them. Hotel prices are based on the check-in and check-out date. visit makcorps.com for API key"
    
    """
    url = f"https://manthankool-makcorps-hotel-price-comparison-v1.p.rapidapi.com/king/v2/{jakarta}/{2018-05-25}/{2018-05-28}"
    querystring = {'location': location, 'check-out-date': check_out_date, 'check-in-date': check_in_date, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "manthankool-makcorps-hotel-price-comparison-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def classic_api(check_out: str, hotel_name: str, location: str, currency: str, no_of_guest: str, check_in: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search any hotel by its name and location and get top 6 vendors/top OTAs within the JSON response. visit - makcorps.com for API key"
    
    """
    url = f"https://manthankool-makcorps-hotel-price-comparison-v1.p.rapidapi.com/flashdel/v2/{hotel_name}/{location}/{currency}/{no_of_guest}/{check_in}/{check_out}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "manthankool-makcorps-hotel-price-comparison-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

