import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def coins(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://coinmarketcal.p.rapidapi.com/coins"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "coinmarketcal.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def categories(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://coinmarketcal.p.rapidapi.com/categories"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "coinmarketcal.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def events(sortby: str=None, showviews: bool=None, daterangeend: str=None, coins: str=None, showonly: str=None, translations: str='en', showvotes: bool=None, categories: str=None, max: int=16, daterangestart: str=None, page: int=1, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Multiple fields (Coins and Categories) can be provided with comma separated ID string. (eg Coins: bitcoin,ethereum | eg Categories: 1,2).
		
		**It’s not possible to retrieve events before 25/11/2017.
		Please, do not forget to use can_occur_before field in the response object, this field is very important as some events can occur before the date stated.** "
    sortby: Sort by
        showviews: Show views
        daterangeend: End date (default value is date of furthest event) (format like 2017-11-25)
        coins: Coins ID - Multiple field - See [GET] /api/coins for the entire list of available coins ID
        showonly: Show only
        translations: Translated title and description
        showvotes: Show votes
        categories: Categories ID - Multiple field - See [GET] /api/categories for the entire list of available categories ID
        max: Maximum amount of events to display per page
        daterangestart: Start date (default value is today) (format like 2017-11-25)
        page: Page number
        
    """
    url = f"https://coinmarketcal.p.rapidapi.com/events"
    querystring = {}
    if sortby:
        querystring['sortBy'] = sortby
    if showviews:
        querystring['showViews'] = showviews
    if daterangeend:
        querystring['dateRangeEnd'] = daterangeend
    if coins:
        querystring['coins'] = coins
    if showonly:
        querystring['showOnly'] = showonly
    if translations:
        querystring['translations'] = translations
    if showvotes:
        querystring['showVotes'] = showvotes
    if categories:
        querystring['categories'] = categories
    if max:
        querystring['max'] = max
    if daterangestart:
        querystring['dateRangeStart'] = daterangestart
    if page:
        querystring['page'] = page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "coinmarketcal.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

