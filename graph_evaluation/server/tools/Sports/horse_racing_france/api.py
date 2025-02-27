import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def list_of_meetings_r_unions(date: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get meetings list by date"
    
    """
    url = f"https://horse-racing-france.p.rapidapi.com/meetings"
    querystring = {'date': date, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "horse-racing-france.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def meeting_r_union_details(date: str, meeting_num: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get meeting details by date and meeting number"
    
    """
    url = f"https://horse-racing-france.p.rapidapi.com/meeting"
    querystring = {'date': date, 'meeting_num': meeting_num, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "horse-racing-france.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def race_course_details(date: str, race_num: int=1, meeting_num: int=1, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get race details by date and race_num"
    
    """
    url = f"https://horse-racing-france.p.rapidapi.com/race"
    querystring = {'date': date, }
    if race_num:
        querystring['race_num'] = race_num
    if meeting_num:
        querystring['meeting_num'] = meeting_num
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "horse-racing-france.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

