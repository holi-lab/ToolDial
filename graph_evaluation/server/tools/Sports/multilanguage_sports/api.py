import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def geteventtypes(categoryids: int, champids: int, withlive: bool, sportids: str, enddate: str, startdate: str, timezoneoffset: int=300, langid: int=4, devicetype: str='Desktop', numformat: str='en', configid: int=1, culture: str='es-ES', skinname: str='bet365', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Event Types"
    
    """
    url = f"https://multilanguage-sports.p.rapidapi.com/GetEventTypes"
    querystring = {'categoryids': categoryids, 'champids': champids, 'withLive': withlive, 'sportids': sportids, 'endDate': enddate, 'startDate': startdate, }
    if timezoneoffset:
        querystring['timezoneOffset'] = timezoneoffset
    if langid:
        querystring['langId'] = langid
    if devicetype:
        querystring['deviceType'] = devicetype
    if numformat:
        querystring['numformat'] = numformat
    if configid:
        querystring['configId'] = configid
    if culture:
        querystring['culture'] = culture
    if skinname:
        querystring['skinName'] = skinname
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "multilanguage-sports.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getfavouriteschamps(enddate: str, startdate: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the Favourites Champs"
    
    """
    url = f"https://multilanguage-sports.p.rapidapi.com/GetFavouritesChamps"
    querystring = {'endDate': enddate, 'startDate': startdate, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "multilanguage-sports.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def gethighlights(sportid: int, numformat: str, showallevents: bool, count: int=10, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Highlights"
    
    """
    url = f"https://multilanguage-sports.p.rapidapi.com/GetHighlights"
    querystring = {'sportId': sportid, 'numformat': numformat, 'showAllEvents': showallevents, }
    if count:
        querystring['count'] = count
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "multilanguage-sports.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def gettopsports(langid: int, topsporttype: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Top Sports"
    
    """
    url = f"https://multilanguage-sports.p.rapidapi.com/GetTopSports"
    querystring = {'langId': langid, 'topSportType': topsporttype, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "multilanguage-sports.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getbreadcrumbnavitem(champids: int, culture: str='es-ES', langid: str=None, sportids: int=0, numformat: str='en', timezoneoffset: int=300, devicetype: str='Desktop', configid: int=1, eventids: int=0, categoryids: int=0, skinname: str='bet365', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Breadcrumb of League"
    
    """
    url = f"https://multilanguage-sports.p.rapidapi.com/GetBreadCrumbNavItem"
    querystring = {'champids': champids, }
    if culture:
        querystring['culture'] = culture
    if langid:
        querystring['langId'] = langid
    if sportids:
        querystring['sportids'] = sportids
    if numformat:
        querystring['numformat'] = numformat
    if timezoneoffset:
        querystring['timezoneOffset'] = timezoneoffset
    if devicetype:
        querystring['deviceType'] = devicetype
    if configid:
        querystring['configId'] = configid
    if eventids:
        querystring['eventids'] = eventids
    if categoryids:
        querystring['categoryids'] = categoryids
    if skinname:
        querystring['skinName'] = skinname
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "multilanguage-sports.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getliveevents(sportids: int, configid: int=1, culture: str='es-ES', langid: int=4, devicetype: str='Desktop', skinname: str='bet365', timezoneoffset: int=300, numformat: str='en', categoryids: int=0, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Live Events of League"
    
    """
    url = f"https://multilanguage-sports.p.rapidapi.com/GetLiveEvents"
    querystring = {'sportids': sportids, }
    if configid:
        querystring['configId'] = configid
    if culture:
        querystring['culture'] = culture
    if langid:
        querystring['langId'] = langid
    if devicetype:
        querystring['deviceType'] = devicetype
    if skinname:
        querystring['skinName'] = skinname
    if timezoneoffset:
        querystring['timezoneOffset'] = timezoneoffset
    if numformat:
        querystring['numformat'] = numformat
    if categoryids:
        querystring['categoryids'] = categoryids
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "multilanguage-sports.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getevents(group: str, markettypeids: int, startdate: str, period: str, enddate: str, timezoneoffset: int=300, langid: int=4, culture: str='es-ES', skinname: str='bet365', sportids: int=0, numformat: str='en', configid: int=1, categoryids: int=0, champids: int=1000000120, devicetype: str='Desktop', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Events of the League"
    
    """
    url = f"https://multilanguage-sports.p.rapidapi.com/GetEvents"
    querystring = {'group': group, 'marketTypeIds': markettypeids, 'startDate': startdate, 'period': period, 'endDate': enddate, }
    if timezoneoffset:
        querystring['timezoneOffset'] = timezoneoffset
    if langid:
        querystring['langId'] = langid
    if culture:
        querystring['culture'] = culture
    if skinname:
        querystring['skinName'] = skinname
    if sportids:
        querystring['sportids'] = sportids
    if numformat:
        querystring['numformat'] = numformat
    if configid:
        querystring['configId'] = configid
    if categoryids:
        querystring['categoryids'] = categoryids
    if champids:
        querystring['champids'] = champids
    if devicetype:
        querystring['deviceType'] = devicetype
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "multilanguage-sports.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def geteventdetails(eventid: int, sportid: int, timezoneoffset: int=300, numformat: str='en', configid: int=1, devicetype: str='Desktop', culture: str='es-ES', langid: int=4, skinname: str='bet365', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get details of Event"
    
    """
    url = f"https://multilanguage-sports.p.rapidapi.com/GetEventDetails"
    querystring = {'eventId': eventid, 'sportId': sportid, }
    if timezoneoffset:
        querystring['timezoneOffset'] = timezoneoffset
    if numformat:
        querystring['numformat'] = numformat
    if configid:
        querystring['configId'] = configid
    if devicetype:
        querystring['deviceType'] = devicetype
    if culture:
        querystring['culture'] = culture
    if langid:
        querystring['langId'] = langid
    if skinname:
        querystring['skinName'] = skinname
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "multilanguage-sports.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def gettopevents(skinname: str, eventcount: int, culture: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Top Events"
    
    """
    url = f"https://multilanguage-sports.p.rapidapi.com/GetTopEvents"
    querystring = {'skinName': skinname, 'eventCount': eventcount, 'culture': culture, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "multilanguage-sports.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getallsports(startdate: str, enddate: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get All Sports"
    
    """
    url = f"https://multilanguage-sports.p.rapidapi.com/GetAllSports"
    querystring = {'startDate': startdate, 'endDate': enddate, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "multilanguage-sports.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getlivenow(sportid: int, numformat: str, showallevents: bool=None, count: str='10', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the current events"
    
    """
    url = f"https://multilanguage-sports.p.rapidapi.com/GetLivenow"
    querystring = {'sportId': sportid, 'numformat': numformat, }
    if showallevents:
        querystring['showAllEvents'] = showallevents
    if count:
        querystring['count'] = count
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "multilanguage-sports.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getupcoming(numformat: str, showallevents: bool, sportid: str, count: int=10, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Upcoming"
    
    """
    url = f"https://multilanguage-sports.p.rapidapi.com/GetUpcoming"
    querystring = {'numformat': numformat, 'showAllEvents': showallevents, 'sportId': sportid, }
    if count:
        querystring['count'] = count
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "multilanguage-sports.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

