import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_vrbo_listing_reviews(listingid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns up to the last 10 reviews on a listing from VRBO
		
		**Where to find your VRBO Listing ID**
		![](https://vitesa.com/rapidAPI_Images/vrbo_listingID.png)"
    
    """
    url = f"https://vrm-str-tools.p.rapidapi.com/getVrboReviews/"
    querystring = {'listingid': listingid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "vrm-str-tools.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_airbnb_listing_ratings(listingid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns the ratings from a listing on Airbnb
		
		**Where to find your Airbnb Listing ID**
		![](https://vitesa.com/rapidAPI_Images/airbnb_listingID.png)"
    
    """
    url = f"https://vrm-str-tools.p.rapidapi.com/getAirbnbRating/"
    querystring = {'listingid': listingid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "vrm-str-tools.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_airbnb_listing_reviews(listingid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns up to the last 10 reviews on a listing from Airbnb
		
		**Where to find your Airbnb Listing ID**
		![](https://vitesa.com/rapidAPI_Images/airbnb_listingID.png)"
    
    """
    url = f"https://vrm-str-tools.p.rapidapi.com/getAirbnbReviews/"
    querystring = {'listingid': listingid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "vrm-str-tools.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

