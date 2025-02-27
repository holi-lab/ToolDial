import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def list_devices(q: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    ""
    q: fuzzy search device id or plate/name
        
    """
    url = f"https://elyez.p.rapidapi.com/device"
    querystring = {}
    if q:
        querystring['q'] = q
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "elyez.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def list_geofences(q: str=None, type: str=None, bbox: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    ""
    q: Search by name
        type: Search by type
        bbox: Search by bbox (minLng, minLat, maxLng, maxLat)
        
    """
    url = f"https://elyez.p.rapidapi.com/geofence"
    querystring = {}
    if q:
        querystring['q'] = q
    if type:
        querystring['type'] = type
    if bbox:
        querystring['bbox'] = bbox
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "elyez.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def activate(code: str, token: str, redirect: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Registration activation, from email"
    redirect: Redirect URL for success activation, eg. login page or welcome page (default: none) -- ENCODE URI COMPONENT!
        
    """
    url = f"https://elyez.p.rapidapi.com/auth/activate"
    querystring = {'code': code, 'token': token, }
    if redirect:
        querystring['redirect'] = redirect
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "elyez.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def all_devices(moving: bool=None, input: str=None, output: str=None, since: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Track ALL user's devices"
    moving: Filter moving (speed > 0) or stopped (speed = 0), default: null
        input: Advanced filter with INPUT sensors conditions (— = any), ex: --1--0--
        output: Advanced filter OUTPUT conditions (— = any), ex: --1--0--
        since: Filter data with tracking time newer than this (format: ISO timestamp). ex: 2017-01-31T14:53:48Z
        
    """
    url = f"https://elyez.p.rapidapi.com/tracking"
    querystring = {}
    if moving:
        querystring['moving'] = moving
    if input:
        querystring['input'] = input
    if output:
        querystring['output'] = output
    if since:
        querystring['since'] = since
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "elyez.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def list_webhooks(type: str=None, endpoint: str=None, device: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    ""
    type: Filter by type
        endpoint: Filter by endpoint (email or url) <case-sensitive>
        device: Filter by device id
        
    """
    url = f"https://elyez.p.rapidapi.com/webhook"
    querystring = {}
    if type:
        querystring['type'] = type
    if endpoint:
        querystring['endpoint'] = endpoint
    if device:
        querystring['device'] = device
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "elyez.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def events(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List all tracking events (alerts), id and name"
    
    """
    url = f"https://elyez.p.rapidapi.com/tracking/events"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "elyez.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def device_models(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Supported device's models"
    
    """
    url = f"https://elyez.p.rapidapi.com/device/model"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "elyez.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def device(is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Track SPESIFIC device by id"
    
    """
    url = f"https://elyez.p.rapidapi.com/tracking/{is_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "elyez.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

