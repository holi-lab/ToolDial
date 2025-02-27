import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def getbreadcrumbnavitem(referer: str='https://fullreto.co/', origin: str='https://fullreto.co', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "GetBreadCrumbNavItem api data"
    
    """
    url = f"https://betsports.p.rapidapi.com/api/Sportsbook/GetBreadCrumbNavItem"
    querystring = {}
    if referer:
        querystring['Referer'] = referer
    if origin:
        querystring['Origin'] = origin
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "betsports.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_banners_new(referer: str='https://www.mozzartbet.com.co/es', host: str='www.mozzartbet.com.co', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "get-banners-new data"
    
    """
    url = f"https://betsports.p.rapidapi.com/get-banners-new"
    querystring = {}
    if referer:
        querystring['Referer'] = referer
    if host:
        querystring['Host'] = host
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "betsports.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def superoffer(cookie: str='i18next=es;', host: str='www.mozzartbet.com.co', referer: str='https://www.mozzartbet.com.co/es', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "superOffer data"
    
    """
    url = f"https://betsports.p.rapidapi.com/superOffer"
    querystring = {}
    if cookie:
        querystring['Cookie'] = cookie
    if host:
        querystring['Host'] = host
    if referer:
        querystring['Referer'] = referer
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "betsports.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def gettspmatches(cookie: str='i18next=es;', host: str='www.mozzartbet.com.co', referer: str='https://www.mozzartbet.com.co/es', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "getTspMatches data"
    
    """
    url = f"https://betsports.p.rapidapi.com/getTspMatches"
    querystring = {}
    if cookie:
        querystring['Cookie'] = cookie
    if host:
        querystring['Host'] = host
    if referer:
        querystring['Referer'] = referer
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "betsports.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def localized_routes(referer: str='https://www.mozzartbet.com.co/es', host: str='www.mozzartbet.com.co', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "localized-routes data"
    
    """
    url = f"https://betsports.p.rapidapi.com/localized-routes"
    querystring = {}
    if referer:
        querystring['Referer'] = referer
    if host:
        querystring['Host'] = host
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "betsports.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getspecialcompetitionview(referer: str='https://www.mozzartbet.com.co/es', host: str='www.mozzartbet.com.co', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "getSpecialCompetitionView data"
    
    """
    url = f"https://betsports.p.rapidapi.com/getSpecialCompetitionView"
    querystring = {}
    if referer:
        querystring['Referer'] = referer
    if host:
        querystring['Host'] = host
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "betsports.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def gettaxrulesv2(referer: str='https://www.mozzartbet.com.co/es', host: str='ww.mozzartbet.com.co', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "getTaxRulesV2 data"
    
    """
    url = f"https://betsports.p.rapidapi.com/getTaxRulesV2"
    querystring = {}
    if referer:
        querystring['Referer'] = referer
    if host:
        querystring['Host'] = host
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "betsports.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getallgames(cookie: str='i18next=es;', host: str='www.mozzartbet.com.co', referer: str='https://www.mozzartbet.com.co/es', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "getAllGames data"
    
    """
    url = f"https://betsports.p.rapidapi.com/getAllGames"
    querystring = {}
    if cookie:
        querystring['Cookie'] = cookie
    if host:
        querystring['Host'] = host
    if referer:
        querystring['Referer'] = referer
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "betsports.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def oldsort(host: str='www.mozzartbet.com.co', referer: str='https://www.mozzartbet.com.co/es', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "oldSort data"
    
    """
    url = f"https://betsports.p.rapidapi.com/oldSort"
    querystring = {}
    if host:
        querystring['Host'] = host
    if referer:
        querystring['Referer'] = referer
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "betsports.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

