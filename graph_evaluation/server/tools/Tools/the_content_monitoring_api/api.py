import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def list_detectors(limit: int, name: str=None, orderby: str='name', endat: str=None, userid: str=None, startat: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieve a list of all active and paused detectors. "
    limit: A required field specifying maxim number of items to retrieve.
        name: Filter by a name of a detector.
        orderby: Order by a specific field.
        endat: Cursor for pagination. Specifies the item id to end a window of list of items to retrieve. startAt and endAt can not be provided simultaneously. If neither of two is provided a list is started from the first item in the order.
        userid: Filter by a user id.
        startat: Cursor for pagination. Specifies the item id to start a window of list of items to retrieve. startAt and endAt can not be provided simultaneously. If neither of two is provided a list is started from the first item in the order.
        
    """
    url = f"https://the-content-monitoring-api.p.rapidapi.com/detector"
    querystring = {'limit': limit, }
    if name:
        querystring['name'] = name
    if orderby:
        querystring['orderby'] = orderby
    if endat:
        querystring['endAt'] = endat
    if userid:
        querystring['userId'] = userid
    if startat:
        querystring['startAt'] = startat
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "the-content-monitoring-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def list_changes(limit: int, detectorid: str=None, endat: str=None, startat: str=None, orderby: str='userId', userid: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieve a list of all changes as per the provided filters. To get the details including AI comment and the calculated difference between the corresponding content snapshots, it is required to request the data for each item individually.
		
		Not every detector activaton leads to a change being detected. The change won't be stored if the snapshots are identical, it is an initial snapshot (nothing to compare to), previous url request failed."
    limit: A required field specifying maxim number of items to retrieve.
        detectorid: Filter by a detector id.
        endat: Cursor for pagination. Specifies the item id to end a window of list of items to retrieve. startAt and endAt can not be provided simultaneously. If neither of two is provided a list is started from the first item in the order.
        startat: Cursor for pagination. Specifies the item id to start a window of list of items to retrieve. startAt and endAt can not be provided simultaneously. If neither of two is provided a list is started from the first item in the order.
        orderby: Order by a specific field.
        userid: Filter by a user id.
        
    """
    url = f"https://the-content-monitoring-api.p.rapidapi.com/change"
    querystring = {'limit': limit, }
    if detectorid:
        querystring['detectorId'] = detectorid
    if endat:
        querystring['endAt'] = endat
    if startat:
        querystring['startAt'] = startat
    if orderby:
        querystring['orderby'] = orderby
    if userid:
        querystring['userId'] = userid
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "the-content-monitoring-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_change(changeid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get detailed information about a specified change including the content snapshots difference and an AI comment."
    
    """
    url = f"https://the-content-monitoring-api.p.rapidapi.com/change/{changeid}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "the-content-monitoring-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

