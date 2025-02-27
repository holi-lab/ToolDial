import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def birthdate_and_location_personality(date: str, country: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Discover your true personality with our birthdate and location webservice! By simply inputting your birthdate and location, our service will provide you with a detailed personality analysis based on Western astrology. Uncover your strengths, weaknesses, and unique traits to help you better understand yourself and those around you."
    
    """
    url = f"https://moonlight-secrets.p.rapidapi.com/personality/{date}/{country}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "moonlight-secrets.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def next_new_moon(location: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Stay in tune with the lunar cycle with our next new moon date webservice! Our service provides accurate and up-to-date information on the next new moon date, along with additional lunar information and features, such as upcoming full moon dates and lunar eclipses. Keep track of the lunar cycle to enhance your spiritual practice, gardening, or simply stay in tune with the natural rhythms of the universe."
    
    """
    url = f"https://moonlight-secrets.p.rapidapi.com/nextnewmoon/{location}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "moonlight-secrets.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def compatibility_score(country1: str, country2: str, date2: str, date1: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Are you curious about your compatibility with your partner, friend, or potential love interest? Our compatibility score webservice can help! By inputting your birthdates and locations, our service will provide you with a detailed analysis of your compatibility, including areas of strength and potential challenges. Find out if you're a match made in heaven or if you need to work a little harder to make it work."
    
    """
    url = f"https://moonlight-secrets.p.rapidapi.com/compatibility/{date1}/{country1}/{date2}/{country2}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "moonlight-secrets.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

