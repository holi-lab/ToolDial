import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def event_detail(is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    ""
    
    """
    url = f"https://lotadata-events-v1.p.rapidapi.com/events/{is_id}/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "lotadata-events-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def events_search(q: str='comedy', bottomright: str='37.124493,-121.589154', date_fromday: str='2015-04-01', date_fromtime: str='18:00', date_today: str='2015-04-01', date_totime: str='22:00', start: int=0, topleft: str='37.4695381,-122.0456679', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    ""
    q: returns events based on keywords, genres, categories
        bottomright: SouthWest bounding box corner (lat,lng)
        date_fromday: returns events scheduled from this day. (ISO Date)
        date_fromtime: Returns events scheduled between this time and toTime. Note: You can search during 6pm to 10pm window spanning multiple days (HH:MM)
        date_today: returns events scheduled to this day
        date_totime: returns events scheduled up to this time
        start: returns matching events starting with the specified offset (paging)
        topleft: NorthWest bounding corner (Latitude,Longitude)
        
    """
    url = f"https://lotadata-events-v1.p.rapidapi.com/search/"
    querystring = {}
    if q:
        querystring['q'] = q
    if bottomright:
        querystring['bottomRight'] = bottomright
    if date_fromday:
        querystring['date.fromDay'] = date_fromday
    if date_fromtime:
        querystring['date.fromTime'] = date_fromtime
    if date_today:
        querystring['date.toDay'] = date_today
    if date_totime:
        querystring['date.toTime'] = date_totime
    if start:
        querystring['start'] = start
    if topleft:
        querystring['topLeft'] = topleft
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "lotadata-events-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

