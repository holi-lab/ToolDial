import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def getcouponmatchescount(skinname: str='betbiga', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "GetCouponMatchesCount data api"
    
    """
    url = f"https://sport-odds1.p.rapidapi.com/Sportsbook/GetCouponMatchesCount"
    querystring = {}
    if skinname:
        querystring['skinName'] = skinname
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sport-odds1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getevents(skinname: str='betbiga', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "GetEvents data api"
    
    """
    url = f"https://sport-odds1.p.rapidapi.com/Sportsbook/GetEvents"
    querystring = {}
    if skinname:
        querystring['skinName'] = skinname
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sport-odds1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getallsports(skinname: str='betbiga', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "GetAllSports data api"
    
    """
    url = f"https://sport-odds1.p.rapidapi.com/Sportsbook/GetAllSports"
    querystring = {}
    if skinname:
        querystring['skinName'] = skinname
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sport-odds1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getfavouriteschamps(skinname: str='betbiga', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "GetFavouritesChamps data api"
    
    """
    url = f"https://sport-odds1.p.rapidapi.com/Sportsbook/GetFavouritesChamps"
    querystring = {}
    if skinname:
        querystring['skinName'] = skinname
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sport-odds1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getupcoming(skinname: str='betbiga', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "GetUpcoming data api"
    
    """
    url = f"https://sport-odds1.p.rapidapi.com/Sportsbook/GetUpcoming"
    querystring = {}
    if skinname:
        querystring['skinName'] = skinname
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sport-odds1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getliveevents(skinname: str='betbiga', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "GetLiveEvents data api"
    
    """
    url = f"https://sport-odds1.p.rapidapi.com/Sportsbook/GetLiveEvents"
    querystring = {}
    if skinname:
        querystring['skinName'] = skinname
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sport-odds1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def geteventtypes(skinname: str='betbiga', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "GetEventTypes data api"
    
    """
    url = f"https://sport-odds1.p.rapidapi.com/SportsBook/GetEventTypes"
    querystring = {}
    if skinname:
        querystring['skinName'] = skinname
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sport-odds1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getlivenow(skinname: str='betbiga', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "GetLivenow data now"
    
    """
    url = f"https://sport-odds1.p.rapidapi.com/Sportsbook/GetLivenow"
    querystring = {}
    if skinname:
        querystring['skinName'] = skinname
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sport-odds1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getlivemenustreaming(skinname: str='betbiga', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "GetLiveMenuStreaming data api"
    
    """
    url = f"https://sport-odds1.p.rapidapi.com/Sportsbook/GetLiveMenuStreaming"
    querystring = {}
    if skinname:
        querystring['skinName'] = skinname
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sport-odds1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def geteventdetails(skinname: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "GetEventDetails api data"
    
    """
    url = f"https://sport-odds1.p.rapidapi.com/Sportsbook/GetEventDetails"
    querystring = {'skinName': skinname, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sport-odds1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def geteventexternalinfo(skinname: str='betbiga', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "GetEventExternalInfo data api"
    
    """
    url = f"https://sport-odds1.p.rapidapi.com/Sportsbook/GetEventExternalInfo"
    querystring = {}
    if skinname:
        querystring['skinName'] = skinname
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sport-odds1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getbreadcrumbnavitem(skinname: str='betbiga', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "GetBreadCrumbNavItem data api"
    
    """
    url = f"https://sport-odds1.p.rapidapi.com/Sportsbook/GetBreadCrumbNavItem"
    querystring = {}
    if skinname:
        querystring['skinName'] = skinname
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sport-odds1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def gettopsportmenu(skinname: str='betbiga', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "GetTopSportMenu data api"
    
    """
    url = f"https://sport-odds1.p.rapidapi.com/Sportsbook/GetTopSportMenu"
    querystring = {}
    if skinname:
        querystring['skinName'] = skinname
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sport-odds1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def gettopsports(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "GetTopSports data api"
    
    """
    url = f"https://sport-odds1.p.rapidapi.com/Sportsbook/GetTopSports"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sport-odds1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getmenubysport(skinname: str='betbiga', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "GetMenuBySport data api"
    
    """
    url = f"https://sport-odds1.p.rapidapi.com/SportsBook/GetMenuBySport"
    querystring = {}
    if skinname:
        querystring['skinName'] = skinname
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sport-odds1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getstatictranslations(skinname: str='betbiga', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "GetStaticTranslations data api"
    
    """
    url = f"https://sport-odds1.p.rapidapi.com/Translation/GetStaticTranslations"
    querystring = {}
    if skinname:
        querystring['skinName'] = skinname
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sport-odds1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def gethighlights(skinname: str='betbiga', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "GetHighlights Data Api"
    
    """
    url = f"https://sport-odds1.p.rapidapi.com/Sportsbook/GetHighlights"
    querystring = {}
    if skinname:
        querystring['skinName'] = skinname
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sport-odds1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

