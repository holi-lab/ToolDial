import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def linkedin_followed_companies(secret: str, token: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Companies followed in LinkedIn API - API help Endpoint: https://hub.loginradius.com/GetCompany/help"
    secret: LoginRadius API Secret
        token: User profile session Token
        
    """
    url = f"https://loginradius.p.rapidapi.com/GetCompany/{SECRET}/{TOKEN}"
    querystring = {'secret': secret, 'token': token, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "loginradius.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def twitter_mentions_api(secret: str, token: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Twitter mentions API - API help Endpoint	https://hub.loginradius.com/status/help"
    secret: LoginRadius API Secret
        token: User profile session Token
        
    """
    url = f"https://loginradius.p.rapidapi.com/status/mentions/{SECRET}/{TOKEN}"
    querystring = {'secret': secret, 'token': token, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "loginradius.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def facebook_groups_api(secret: str, token: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Facebook Groups API - API help Endpoint	https://hub.loginradius.com/GetGroups/help"
    secret: LoginRadius API Secret
        token: User profile session Token
        
    """
    url = f"https://loginradius.p.rapidapi.com/GetGroups/{SECRET}/{TOKEN}"
    querystring = {'secret': secret, 'token': token, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "loginradius.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def contact_network_api(secret: str, token: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This API is use to get contacts/network data for user’s social account. This API is only supported by oAuth ID providers such as Facebook, LinkedIn, Twitter, Google, Yahoo, MSN. - API help Endpoint	https://hub.loginradius.com/contacts/help"
    secret: LoginRadius API Secret
        token: User profile session Token
        
    """
    url = f"https://loginradius.p.rapidapi.com/contacts/{SECRET}/{TOKEN}"
    querystring = {'secret': secret, 'token': token, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "loginradius.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def user_profile_api(secret: str, token: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This API is use to get the social profile data from user’s social account after authentication. The social profile will be retrieved via oAuth and OpenID protocols. The data would be normalized into LoginRadius’s JSON format. - API help Endpoint	https://hub.loginradius.com/userprofile/help"
    secret: LoginRadius API Secret
        token: LoginRadius login session Token
        
    """
    url = f"https://loginradius.p.rapidapi.com/UserProfile/{SECRET}/{TOKEN}"
    querystring = {'secret': secret, 'token': token, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "loginradius.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def facebook_posts_api(secret: str, token: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Facebook Posts API - API help Endpoint	https://hub.loginradius.com/GetPosts/help"
    secret: LoginRadius API Secret
        token: User profile session Token
        
    """
    url = f"https://loginradius.p.rapidapi.com/GetPosts/{SECRET}/{TOKEN}"
    querystring = {'secret': secret, 'token': token, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "loginradius.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def facebook_events_api(secret: str, token: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Facebook Events API - API help Endpoint	https://hub.loginradius.com/GetEvents/help"
    secret: LoginRadius API Secret
        token: User profile session Token
        
    """
    url = f"https://loginradius.p.rapidapi.com/GetEvents/{SECRET}/{TOKEN}"
    querystring = {'secret': secret, 'token': token, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "loginradius.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

