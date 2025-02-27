import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def getmediabycode(mediacode: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "GetMediaByCode"
    
    """
    url = f"https://instagram64.p.rapidapi.com/Ins/Media_GetMediaByCode/go/"
    querystring = {'mediaCode': mediacode, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "instagram64.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getmedialistbyusername(authorname: str, userid: str, page: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "GetMediaListByTag"
    
    """
    url = f"https://instagram64.p.rapidapi.com/Ins/Media_GetMediaListByUserName/go/"
    querystring = {'authorName': authorname, 'userID': userid, 'page': page, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "instagram64.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getmedialistbytag(taginfo: str, lastendcursor: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "GetMediaListByTag"
    
    """
    url = f"https://instagram64.p.rapidapi.com/Ins/Media_GetMediaListByTag/go/"
    querystring = {'tagInfo': taginfo, }
    if lastendcursor:
        querystring['lastEndCursor'] = lastendcursor
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "instagram64.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

