import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def show_city(mode: str, city: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "show city by its name"
    
    """
    url = f"https://basic-country-city-information.p.rapidapi.com/"
    querystring = {'mode': mode, 'city': city, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "basic-country-city-information.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def show_country(mode: str, code2: str=None, localname: str=None, code: str='DEU', name: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "search and show a country by 2 or 3 letter code or by name and localname."
    code2: 2 letter code like US, DE, RU...
        localname: like: Cesko, Viet Nam, Deutschland...
        code: 3 letter code like USA, DEU, RUS...
        name: like japan, china, germany
        
    """
    url = f"https://basic-country-city-information.p.rapidapi.com/"
    querystring = {'mode': mode, }
    if code2:
        querystring['code2'] = code2
    if localname:
        querystring['localname'] = localname
    if code:
        querystring['code'] = code
    if name:
        querystring['name'] = name
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "basic-country-city-information.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def show_world(order: str=None, sort: str=None, governmentform: str=None, continent: str=None, limit: str='10', language: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Show world countries"
    order: can be:
code, code2, name,  localname, country, region, continent, indepyear, surfacearea, governmentform.
        sort: ASC or DESC
        governmentform: can be:
Administrated by the UN, Autonomous Area, Co-administrated, Commonwealth of the US, Constitutional Monarchy, Constitutional Monarchy (Emirate), Constitutional Monarchy, Federation, Dependent Territory of Norway, Dependent Territory of the UK, Dependent Territory of the US, Emirate Federation, Federal Republic, Federation, Independent Church State, Islamic Emirate, Islamic Republic, Monarchy, Monarchy (Emirate), Monarchy (Sultanate), Nonmetropolitan Territory of France, Nonmetropolitan Territory of New Zealand, Nonmetropolitan Territory of The Netherlands, Occupied by Marocco, Overseas Department of France, Parlementary Monarchy, Parliamentary Coprincipality, Part of Denmark, People'sRepublic, Republic, Socialistic Republic, Socialistic State, Special Administrative Region of China, Territorial Collectivity of France, Territory of Australia, US Territory,
        continent: can be:
Africa, Antarctica, Asia, Europe, North America, Oceania, South America.
        language: can be:
english, dutch, german, chinese, japanese ....
        
    """
    url = f"https://basic-country-city-information.p.rapidapi.com/"
    querystring = {}
    if order:
        querystring['order'] = order
    if sort:
        querystring['sort'] = sort
    if governmentform:
        querystring['governmentform'] = governmentform
    if continent:
        querystring['continent'] = continent
    if limit:
        querystring['limit'] = limit
    if language:
        querystring['language'] = language
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "basic-country-city-information.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

