import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def getattendee(conferenceid: str, attendeeid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns an Attendee object for the uuid of the conference"
    conferenceid: ID of conference with attendee to be updated
        attendeeid: ID of attendee
        
    """
    url = f"https://candcconferencecreator4.p.rapidapi.com/conference/{conferenceid}/attendee/{attendeeid}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "candcconferencecreator4.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getsession(conferenceid: str, sessionid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the session for a conference based on the session id provided"
    conferenceid: ID of conference with session
        sessionid: ID of session
        
    """
    url = f"https://candcconferencecreator4.p.rapidapi.com/conference/{conferenceid}/session/{sessionid}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "candcconferencecreator4.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getconferenceattendees(conferenceid: str, startindex: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns the list of attendees for a conference 1k at a time"
    conferenceid: ID of conference to attendee is to be added
        startindex: Start index for retrieveing next iteration of attendees
        
    """
    url = f"https://candcconferencecreator4.p.rapidapi.com/conference/{conferenceid}/attendee"
    querystring = {}
    if startindex:
        querystring['startIndex'] = startindex
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "candcconferencecreator4.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getconferencebyid(conferenceid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Gets as existing confernce object but uuid"
    conferenceid: ID of conference to return
        
    """
    url = f"https://candcconferencecreator4.p.rapidapi.com/conference/{conferenceid}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "candcconferencecreator4.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getsessions(conferenceid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Gets a list fo sessions for a conference"
    conferenceid: ID of conference with attendee to be updated
        
    """
    url = f"https://candcconferencecreator4.p.rapidapi.com/conference/{conferenceid}/session"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "candcconferencecreator4.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

