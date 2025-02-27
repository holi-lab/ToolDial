import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def getallquotes(limit: int=10, page: int=1, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "You will get all the quotes here based on.
		- limit
		- page"
    limit: Item limitations per page. (default: 10)
        page: Page number to be displayed. (default: 1)
        
    """
    url = f"https://radio-world-75-000-worldwide-fm-radio-stations.p.rapidapi.com/get_quotes.php"
    querystring = {}
    if limit:
        querystring['limit'] = limit
    if page:
        querystring['page'] = page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "radio-world-75-000-worldwide-fm-radio-stations.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getstationbycountry(country_id: int, limit: int=10, order: str='ASC', page: int=1, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Getting all Stations by country based on.
		- country id
		- limit
		- page
		- order"
    country_id: Country ID* to fetch stations (default: 26 (USA))
        limit: Item limitation per page. (default: 10)
        order: Ascending/Disascending order. (use only ASC or DESC)
        page: Page Number. (default: 1)
        
    """
    url = f"https://radio-world-75-000-worldwide-fm-radio-stations.p.rapidapi.com/station_by_country.php"
    querystring = {'country_id': country_id, }
    if limit:
        querystring['limit'] = limit
    if order:
        querystring['order'] = order
    if page:
        querystring['page'] = page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "radio-world-75-000-worldwide-fm-radio-stations.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def gethomepage(limit: int=10, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "*We have made an option for you to display result in your landing/home page. So you don't have to work hard to achieve this.*
		Get limited radios to show in landing/home page, based on.
		- limit"
    limit: Item limitation on landing page. (default: 10)
        
    """
    url = f"https://radio-world-75-000-worldwide-fm-radio-stations.p.rapidapi.com/get_home.php"
    querystring = {}
    if limit:
        querystring['limit'] = limit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "radio-world-75-000-worldwide-fm-radio-stations.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getallstation(random: bool=None, order: str='ASC', page: int=1, limit: int=10, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "You will get all recent stations  here, based on.
		- limit (integer)
		- order (string)
		- page (integer)
		- random (boolean)"
    random: Calling random radios. (default : false)
        order: Ascending/Disascending order. (use only ASC or DESC)
        page: Page Number. (default: 1)
        limit: Limitation per page. (default: 10)
        
    """
    url = f"https://radio-world-75-000-worldwide-fm-radio-stations.p.rapidapi.com/all_radios.php"
    querystring = {}
    if random:
        querystring['random'] = random
    if order:
        querystring['order'] = order
    if page:
        querystring['page'] = page
    if limit:
        querystring['limit'] = limit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "radio-world-75-000-worldwide-fm-radio-stations.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def searchstation(keyword: str, order: str='ASC', page: int=1, limit: int=10, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search Radio station by keywords.
		- keyword = radio name, country name
		- limit = item limitation per page ***default 10***
		- page = page number to be displayed ***default 1***
		- order = order by ASCENDING/DISASCENDING  *(use only ASC or DESC)* ***default ASC***"
    keyword: Enter search keyword here.
        order: Display item by Ascending or Disascending order
        page: Page Number.
        limit: Item limitation per page.
        
    """
    url = f"https://radio-world-75-000-worldwide-fm-radio-stations.p.rapidapi.com/search_radio.php"
    querystring = {'keyword': keyword, }
    if order:
        querystring['order'] = order
    if page:
        querystring['page'] = page
    if limit:
        querystring['limit'] = limit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "radio-world-75-000-worldwide-fm-radio-stations.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getallcountry(limit: int=500, page: int=1, order: str='ASC', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all countries Name based on.
		- limit
		- page
		- order"
    limit: Item limitation per page. (default is length.value of countries array. here is 500 max)
        page: Page no to be displayed (default : 1)
        order: Ascending/Disascending order (use ASC or DESC)
        
    """
    url = f"https://radio-world-75-000-worldwide-fm-radio-stations.p.rapidapi.com/get_all_country.php"
    querystring = {}
    if limit:
        querystring['limit'] = limit
    if page:
        querystring['page'] = page
    if order:
        querystring['order'] = order
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "radio-world-75-000-worldwide-fm-radio-stations.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

