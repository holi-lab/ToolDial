import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def retrieving_past_content(accept: str='application/json', hub_topic: str=None, hub_callback: str=None, count: int=10, before: str=None, after: str=None, format: str=None, callback: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This call allows you to retrieve past entries from one or more feeds. Note that you need to be subscribed to the feed(s) in order to do this."
    accept: if you want to retrieve entries in json format (for feeds only!), similar to using the query string header format=json
        hub_topic: The URL of the HTTP resource for which you want the past entries.
        hub_callback: The value can either be a callback with which you are subscribed to one or more feeds or a search query that should match one or more callback urls used to subscribed to several feeds. Please, use the query syntax used to search for subscriptions. In both cases, make sure there are less than 200 matching feeds.
        count: Optional number of items you want to retrieve. Current max is 50 and default is 10.
        before: The id of an entry in the feed. The response will only include entries published before this one.
        after: The id of an entry in the feed. The response will only include entries published after this one.
        format: json if you want to retrieve entries in json format (for feeds only!).
        callback: This will render the entries as a JSONP.
        
    """
    url = f"https://superfeedr-superfeedr-v1.p.rapidapi.com/"
    querystring = {}
    if accept:
        querystring['Accept'] = accept
    if hub_topic:
        querystring['hub.topic'] = hub_topic
    if hub_callback:
        querystring['hub.callback'] = hub_callback
    if count:
        querystring['count'] = count
    if before:
        querystring['before'] = before
    if after:
        querystring['after'] = after
    if format:
        querystring['format'] = format
    if callback:
        querystring['callback'] = callback
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "superfeedr-superfeedr-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

