import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def user_event_timeline(event_type: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    ""
    event_type: The type of event you wish to see the user's activity for.
        
    """
    url = f"https://juliuss-dailycred.p.rapidapi.com/admin/api/user/activity.json"
    querystring = {'event_type': event_type, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "juliuss-dailycred.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def create_a_user_with_only_an_email(email: str, client_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    ""
    email: The user's email.
        client_id: Your client id, found on your settings page.
        
    """
    url = f"https://juliuss-dailycred.p.rapidapi.com/admin/api/email"
    querystring = {'email': email, 'client_id': client_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "juliuss-dailycred.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def account_event_timeline(event_type: str, client_id: str, client_secret: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    ""
    event_type: The specific event type you want to query for.
        client_id: Your client id, found on your settings page.
        client_secret: Your client secret key, found on your settings page.
        
    """
    url = f"https://juliuss-dailycred.p.rapidapi.com/admin/api/eventgraph.json"
    querystring = {'event_type': event_type, 'client_id': client_id, 'client_secret': client_secret, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "juliuss-dailycred.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def find_a_user(client_id: str, client_secret: str, user_id: str=None, email: str=None, access_token: str=None, username: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    ""
    client_id: Your client id, found on your settings page.
        client_secret: Your client secret key, found on your settings page.
        user_id: The user's dailycred user id.
        email: The user's email.
        access_token: An access token for your user.
        username: The user's username.
        
    """
    url = f"https://juliuss-dailycred.p.rapidapi.com/admin/api/user.json"
    querystring = {'client_id': client_id, 'client_secret': client_secret, }
    if user_id:
        querystring['user_id'] = user_id
    if email:
        querystring['email'] = email
    if access_token:
        querystring['access_token'] = access_token
    if username:
        querystring['username'] = username
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "juliuss-dailycred.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def connect_an_existing_user_with_a_new_identity_provider(identity_provider: str, client_id: str, redirect_uri: str=None, state: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    ""
    identity_provider: The desired social network your user to authenticate with. If no identity_provider parameter is provided, it will default to email and password authentication. You can also set identity_provider to gateway to use our UI to let the user choose who to sign in with.
        client_id: Your DailyCred client ID
        redirect_uri: After authentication, the user will be redirected to this url. The url must be within one of your approved domains in your account settings. If omitted we will use your account default callback.
        state: You can include a state parameter to help prevent cross-site request forgery. When your user is redirected back to your site after authenticating, the state parameter will be included.
        
    """
    url = f"https://juliuss-dailycred.p.rapidapi.com/connect"
    querystring = {'identity_provider': identity_provider, 'client_id': client_id, }
    if redirect_uri:
        querystring['redirect_uri'] = redirect_uri
    if state:
        querystring['state'] = state
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "juliuss-dailycred.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

