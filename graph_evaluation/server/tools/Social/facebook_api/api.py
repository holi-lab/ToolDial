import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_likes_and_reactions(url: str, cookie: str, nextcursor: str=None, token: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get users who liked or reacted to a facebook post"
    
    """
    url = f"https://facebook-api10.p.rapidapi.com/getReactions"
    querystring = {'url': url, 'cookie': cookie, }
    if nextcursor:
        querystring['nextCursor'] = nextcursor
    if token:
        querystring['token'] = token
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "facebook-api10.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_comments(url: str, cookie: str, nextcursor: str=None, token: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all comments from from given post including comment text, reactions count, mentions, comment url, user information, etc"
    
    """
    url = f"https://facebook-api10.p.rapidapi.com/getComments"
    querystring = {'url': url, 'cookie': cookie, }
    if nextcursor:
        querystring['nextCursor'] = nextcursor
    if token:
        querystring['token'] = token
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "facebook-api10.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_group_members(cookie: str, groupid: str, token: str=None, nextcursor: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Gets list of group members by group id.  10 results per page. Use `nextCursor` parameter from the response  to get to members from next page"
    
    """
    url = f"https://facebook-api10.p.rapidapi.com/getGroupMembers"
    querystring = {'cookie': cookie, 'groupId': groupid, }
    if token:
        querystring['token'] = token
    if nextcursor:
        querystring['nextCursor'] = nextcursor
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "facebook-api10.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_my_groups(cookie: str, nextcursor: str=None, token: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get list of groups where you are a member"
    
    """
    url = f"https://facebook-api10.p.rapidapi.com/getMyGroups"
    querystring = {'cookie': cookie, }
    if nextcursor:
        querystring['nextCursor'] = nextcursor
    if token:
        querystring['token'] = token
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "facebook-api10.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def authenticate(cookie: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Generates authorization token to use for future api calls"
    
    """
    url = f"https://facebook-api10.p.rapidapi.com/authenticate"
    querystring = {'cookie': cookie, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "facebook-api10.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

