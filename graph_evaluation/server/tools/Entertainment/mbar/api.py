import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def event(name: str=None, club: str=None, club_name: str=None, start_time: str=None, end_time: str=None, time_modified: str=None, datefilter: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Can be ordered by start time, descending; and various fields can be filtered"
    name: the name of the event
        club: the id of the club series the event belongs to.
        club_name: the name of the club series the event belongs to.
        start_time: the event's start time, in YYYY-MM-DD HH:MM[:ss[.uuuuuu]] format.
        end_time: the event's end time, in YYYY-MM-DD HH:MM[:ss[.uuuuuu]] format.
        time_modified: the timestamp of the last modification, in YYYY-MM-DD HH:MM[:ss[.uuuuuu]] format.
        datefilter: ongoing — ongoing events, next — next events, including any ongoing ones., upcoming — upcoming events, excluding any ongoing ones.today — all events happening today (day changes at 4 in the morning).tomorrow — all events happening tomorrow (day changes at 4 in the morning).
        
    """
    url = f"https://mikaraunio-mbar.p.rapidapi.com/event/"
    querystring = {}
    if name:
        querystring['name'] = name
    if club:
        querystring['club'] = club
    if club_name:
        querystring['club__name'] = club_name
    if start_time:
        querystring['start_time'] = start_time
    if end_time:
        querystring['end_time'] = end_time
    if time_modified:
        querystring['time_modified'] = time_modified
    if datefilter:
        querystring['datefilter'] = datefilter
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "mikaraunio-mbar.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def club_series(resource_uri: str=None, resource_webpage_uri: str=None, name: str=None, lead: str=None, description: str=None, picture_uri: str=None, picture_caption: str=None, hosts: str=None, events: str=None, extrainfo_urls: str=None, time_modified: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    ""
    resource_uri: the URI of this club series resource.
        resource_webpage_uri: the URI of this club series webpage on mbar's site.
        name: the club series' name.
        lead: club series description lead.
        description: club series description body.
        picture_uri: club series picture URI.
        picture_caption: club series picture caption.
        hosts: a list of objects representing the artists hosts hosting this club series, each with the following fields: full_name — the artist's name, with affiliation. resource_uri — the URI of the artist resource.
        events: a list of objects representing the events in this club series, each with the following fields: name — the name of the event. start_time — event start time and date in ISO 8601 format, Helsinki local time. end_time — event end time and date in ISO 8601 format, Helsinki local time. resource_uri — the URI of the event resource.
        extrainfo_urls: a list of objects representing URLs pointing to extra info about the club series, each with the following fields: address — URL address. description — URL description. type — URL type (Facebook, homepage, ...)
        time_modified: the timestamp of the last modification.
        
    """
    url = f"https://mikaraunio-mbar.p.rapidapi.com/club/"
    querystring = {}
    if resource_uri:
        querystring['resource_uri'] = resource_uri
    if resource_webpage_uri:
        querystring['resource_webpage_uri'] = resource_webpage_uri
    if name:
        querystring['name'] = name
    if lead:
        querystring['lead'] = lead
    if description:
        querystring['description'] = description
    if picture_uri:
        querystring['picture_uri'] = picture_uri
    if picture_caption:
        querystring['picture_caption'] = picture_caption
    if hosts:
        querystring['hosts'] = hosts
    if events:
        querystring['events'] = events
    if extrainfo_urls:
        querystring['extrainfo_urls'] = extrainfo_urls
    if time_modified:
        querystring['time_modified'] = time_modified
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "mikaraunio-mbar.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def artist(name: str=None, time_modified: str=None, extra_info: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    ""
    name: the name of the artist.
        time_modified: the timestamp of the last modification, in YYYY-MM-DD HH:MM[:ss[.uuuuuu]] format.
        extra_info: extra info such as artist affiliation or country of origin.
        
    """
    url = f"https://mikaraunio-mbar.p.rapidapi.com/artist/"
    querystring = {}
    if name:
        querystring['name'] = name
    if time_modified:
        querystring['time_modified'] = time_modified
    if extra_info:
        querystring['extra_info'] = extra_info
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "mikaraunio-mbar.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

