import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def sanctions_and_watch_lists_screening_name_check(name: str, matchtype: str=None, format: str=None, webhook: str=None, searchtype: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This API endpoint searches a queried person or entity in SafetyNet’s High Risk Database and returns if it is found or not. If the person or the entity is found it also adds short details to the response."
    
    """
    url = f"https://highrisk-check3.p.rapidapi.com/information/existence"
    querystring = {'name': name, }
    if matchtype:
        querystring['matchtype'] = matchtype
    if format:
        querystring['format'] = format
    if webhook:
        querystring['webhook'] = webhook
    if searchtype:
        querystring['searchtype'] = searchtype
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "highrisk-check3.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def sanctions_and_watch_lists_screening(name: str, nationality: str=None, format: str=None, gender: str=None, pob: str=None, matchtype: str=None, age: int=None, limit: int=None, searchtype: str=None, offset: int=None, webhook: str=None, alias: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This API endpoint can be used to search an individual or an organisation in SafetyNet’s High Risk Database. If the individual or entity is found in the High Risk Database, it returns detailed information about the person of interest or entity. SafetyNet has a database of over 7.5 million individuals and entities obtained from over 1800 global sources such as Sanctions Lists, PEP (Politically Exposed Person) Lists, Most Wanted Lists etc."
    
    """
    url = f"https://highrisk-check3.p.rapidapi.com/information/data"
    querystring = {'name': name, }
    if nationality:
        querystring['nationality'] = nationality
    if format:
        querystring['format'] = format
    if gender:
        querystring['gender'] = gender
    if pob:
        querystring['pob'] = pob
    if matchtype:
        querystring['matchtype'] = matchtype
    if age:
        querystring['age'] = age
    if limit:
        querystring['limit'] = limit
    if searchtype:
        querystring['searchtype'] = searchtype
    if offset:
        querystring['offset'] = offset
    if webhook:
        querystring['webhook'] = webhook
    if alias:
        querystring['alias'] = alias
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "highrisk-check3.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

