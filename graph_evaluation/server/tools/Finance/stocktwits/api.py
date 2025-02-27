import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def messages_like(is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Like a message on StockTwits as the authenticating user."
    id: ID of the message you want to like for the authenticating user
        
    """
    url = f"https://stocktwits.p.rapidapi.com/messages/like.json"
    querystring = {'id': is_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "stocktwits.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def account_verify(callback: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This verifies the credentials of a user. Useful for checking if authentication method is correct."
    callback: Define your own callback function name, add this parameter as the value.
        
    """
    url = f"https://stocktwits.p.rapidapi.com/account/verify.json"
    querystring = {}
    if callback:
        querystring['callback'] = callback
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "stocktwits.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def messages_show(is_id: str, conversation: bool=None, callback: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This shows the specified message details. This is used in a stand alone display. View the display guidelines"
    conversation: Set to true to retrieve all meesages of the associated conversation.
        callback: Define your own callback function name, add this parameter as the value.
        
    """
    url = f"https://stocktwits.p.rapidapi.com/messages/show/{is_id}.json"
    querystring = {}
    if conversation:
        querystring['conversation'] = conversation
    if callback:
        querystring['callback'] = callback
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "stocktwits.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def streams_mentions(since: int=None, max: int=None, limit: int=20, callback: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns the most recent 30 messages containing mentions of the authenticating user's handle. These are considered public replies"
    since: Returns results with an ID greater than (more recent than) the specified ID.
        max: Returns results with an ID less than (older than) or equal to the specified ID.
        limit: Default and max limit is 30. This limit must be a number under 30.
        callback: Define your own callback function name, add this parameter as the value.
        
    """
    url = f"https://stocktwits.p.rapidapi.com/streams/mentions.json"
    querystring = {}
    if since:
        querystring['since'] = since
    if max:
        querystring['max'] = max
    if limit:
        querystring['limit'] = limit
    if callback:
        querystring['callback'] = callback
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "stocktwits.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def messages_unlike(is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Unlike a message on StockTwits as the authenticating user."
    id: ID of the message you want to unlike for the authenticating user
        
    """
    url = f"https://stocktwits.p.rapidapi.com/messages/unlike.json"
    querystring = {'id': is_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "stocktwits.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def streams_user(is_id: str, since: int=None, max: int=None, limit: int=None, callback: str=None, filter: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns the most recent 30 messages for the specified user. Includes user object in response."
    id: User ID or Username of the stream's user you want to show
        since: Returns results with an ID greater than (more recent than) the specified ID.
        max: Returns results with an ID less than (older than) or equal to the specified ID.
        limit: Default and max limit is 30. This limit must be a number under 30.
        callback: Define your own callback function name, add this parameter as the value.
        filter: Filter messages by links, charts, or videos.
        
    """
    url = f"https://stocktwits.p.rapidapi.com/streams/user/{is_id}.json"
    querystring = {}
    if since:
        querystring['since'] = since
    if max:
        querystring['max'] = max
    if limit:
        querystring['limit'] = limit
    if callback:
        querystring['callback'] = callback
    if filter:
        querystring['filter'] = filter
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "stocktwits.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def watchlists_index(callback: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a list of private watch lists for the authenticating user."
    callback: Define your own callback function name, add this parameter as the value.
        
    """
    url = f"https://stocktwits.p.rapidapi.com/watchlists.json"
    querystring = {}
    if callback:
        querystring['callback'] = callback
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "stocktwits.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def account_update(name: str=None, email: str=None, username: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This updates the properties of the authenticating user's account."
    name: The full name of the account holder
        email: The email address for the account holder
        username: The username for the account holder
        
    """
    url = f"https://stocktwits.p.rapidapi.com/account/update.json"
    querystring = {}
    if name:
        querystring['name'] = name
    if email:
        querystring['email'] = email
    if username:
        querystring['username'] = username
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "stocktwits.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def streams_direct(since: int=None, max: int=None, limit: int=20, callback: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns the most recent 30 direct messages sent to the authenticating user. These area all private messages sent and recieved."
    since: Returns results with an ID greater than (more recent than) the specified ID.
        max: Returns results with an ID less than (older than) or equal to the specified ID.
        limit: Default and max limit is 30. This limit must be a number under 30.
        callback: Define your own callback function name, add this parameter as the value.
        
    """
    url = f"https://stocktwits.p.rapidapi.com/streams/direct.json"
    querystring = {}
    if since:
        querystring['since'] = since
    if max:
        querystring['max'] = max
    if limit:
        querystring['limit'] = limit
    if callback:
        querystring['callback'] = callback
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "stocktwits.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def streams_home(since: int=None, max: int=None, limit: int=20, callback: str=None, filter: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns the most recent 30 messages posted to the authenticating user's home stream, which is made up of the users and stocks they follow."
    since: Returns results with an ID greater than (more recent than) the specified ID
        max: Returns results with an ID less than (older than) or equal to the specified ID.
        limit: Default and max limit is 30. This limit must be a number under 30.
        callback: Define your own callback function name, add this parameter as the value.
        filter: Filter messages by links, charts, videos, or top.
        
    """
    url = f"https://stocktwits.p.rapidapi.com/streams/home.json"
    querystring = {}
    if since:
        querystring['since'] = since
    if max:
        querystring['max'] = max
    if limit:
        querystring['limit'] = limit
    if callback:
        querystring['callback'] = callback
    if filter:
        querystring['filter'] = filter
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "stocktwits.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def streams_symbol(is_id: str, since: int=None, max: int=None, limit: int=20, callback: str=None, filter: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns the most recent 30 messages for the specified symbol."
    id: Ticker symbol, Stock ID, or RIC code of the symbol
        since: Returns results with an ID greater than (more recent than) the specified ID
        max: Returns results with an ID less than (older than) or equal to the specified ID
        limit: Default and max limit is 30. This limit must be a number under 30
        callback: Define your own callback function name, add this parameter as the value.
        filter: Filter messages by links, charts, videos, or top.
        
    """
    url = f"https://stocktwits.p.rapidapi.com/streams/symbol/{is_id}.json"
    querystring = {}
    if since:
        querystring['since'] = since
    if max:
        querystring['max'] = max
    if limit:
        querystring['limit'] = limit
    if callback:
        querystring['callback'] = callback
    if filter:
        querystring['filter'] = filter
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "stocktwits.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def streams_investor_relations(since: int=None, max: int=None, limit: str='20', callback: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns the most recent 30 messages posted by verified Investor Relations customers."
    since: Returns results with an ID greater than (more recent than) the specified ID.
        max: Returns results with an ID less than (older than) or equal to the specified ID.
        limit: Default and max limit is 30. This limit must be a number under 30.
        callback: Define your own callback function name, add this parameter as the value.
        
    """
    url = f"https://stocktwits.p.rapidapi.com/streams/investor_relations.json"
    querystring = {}
    if since:
        querystring['since'] = since
    if max:
        querystring['max'] = max
    if limit:
        querystring['limit'] = limit
    if callback:
        querystring['callback'] = callback
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "stocktwits.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def streams_friends(since: int=None, max: int=None, limit: int=20, callback: str=None, filter: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns the most recent 30 messages posted to the authenticating user's people stream of the users they follow."
    since: Returns results with an ID greater than (more recent than) the specified ID.
        max: Returns results with an ID less than (older than) or equal to the specified ID.
        limit: Default and max limit is 30. This limit must be a number under 30.
        callback: Define your own callback function name, add this parameter as the value.
        filter: Filter messages by links, charts, videos, or top.
        
    """
    url = f"https://stocktwits.p.rapidapi.com/streams/friends.json"
    querystring = {}
    if since:
        querystring['since'] = since
    if max:
        querystring['max'] = max
    if limit:
        querystring['limit'] = limit
    if callback:
        querystring['callback'] = callback
    if filter:
        querystring['filter'] = filter
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "stocktwits.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def streams_watchlist(is_id: str, since: int=None, max: int=None, limit: int=20, callback: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns the most recent 30 messages for the specified watch list for the authenticating user. The watch list is a private list of all the symbols a user is watching."
    id: ID of the watch list you want to show from the authenticating user
        since: Returns results with an ID greater than (more recent than) the specified ID.
        max: Returns results with an ID less than (older than) or equal to the specified ID.
        limit: Default and max limit is 30. This limit must be a number under 30.
        callback: Define your own callback function name, add this parameter as the value.
        
    """
    url = f"https://stocktwits.p.rapidapi.com/streams/watchlist/{is_id}.json"
    querystring = {}
    if since:
        querystring['since'] = since
    if max:
        querystring['max'] = max
    if limit:
        querystring['limit'] = limit
    if callback:
        querystring['callback'] = callback
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "stocktwits.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def watchlists_show(is_id: str, callback: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns the the list of ticker symbols in a specified watch list for the authenticating user. Required parameter is the ID of the watch list, not the name of the watch list."
    id: The ID of the watch list to be shown
        callback: Define your own callback function name, add this parameter as the value.
        
    """
    url = f"https://stocktwits.p.rapidapi.com/watchlists/show/{is_id}.json"
    querystring = {}
    if callback:
        querystring['callback'] = callback
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "stocktwits.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

