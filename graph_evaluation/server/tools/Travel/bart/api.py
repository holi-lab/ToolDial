import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def schedule_information(cmd: str, orig: str='24th', dest: str='rock', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    ""
    cmd: See more examples at http://api.bart.gov/docs/overview/examples.aspx
        
    """
    url = f"https://community-bart.p.rapidapi.com/sched.aspx"
    querystring = {'cmd': cmd, }
    if orig:
        querystring['orig'] = orig
    if dest:
        querystring['dest'] = dest
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "community-bart.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def advisory_information(cmd: str, orig: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    ""
    cmd: See more examples http://api.bart.gov/docs/overview/examples.aspx
        orig: Optional station filter. Uses 4 character BART station abbreviations (http://api.bart.gov/docs/overview/abbrev.aspx)
        
    """
    url = f"https://community-bart.p.rapidapi.com/bsa.aspx"
    querystring = {'cmd': cmd, }
    if orig:
        querystring['orig'] = orig
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "community-bart.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def route_information(cmd: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    ""
    cmd: See more examples http://api.bart.gov/docs/overview/examples.aspx
        
    """
    url = f"https://community-bart.p.rapidapi.com/route.aspx"
    querystring = {'cmd': cmd, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "community-bart.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def station_information(cmd: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    ""
    cmd: See more examples at http://api.bart.gov/docs/overview/examples.aspx
        
    """
    url = f"https://community-bart.p.rapidapi.com/stn.aspx"
    querystring = {'cmd': cmd, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "community-bart.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def real_time_information(cmd: str, orig: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    ""
    cmd: See more examples http://api.bart.gov/docs/overview/examples.aspx
        orig: Specifies the station. Stations are referenced by their abbreviations (http://api.bart.gov/docs/overview/abbrev.aspx). You can also use 'ALL' to get all of the current ETD's.
        
    """
    url = f"https://community-bart.p.rapidapi.com/etd.aspx"
    querystring = {'cmd': cmd, 'orig': orig, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "community-bart.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

