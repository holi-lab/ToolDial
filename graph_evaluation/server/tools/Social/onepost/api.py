import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_a_provider(is_id: str, secret_key: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get data for a single Provider record"
    
    """
    url = f"https://onepost1.p.rapidapi.com/api/v1/providers/{is_id}"
    querystring = {'secret_key': secret_key, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "onepost1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_all_providers(secret_key: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns all Provider records owned by the API user account."
    
    """
    url = f"https://onepost1.p.rapidapi.com/api/v1/providers"
    querystring = {'secret_key': secret_key, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "onepost1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_an_authorization(secret_key: str, is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns data for a single Authorization"
    
    """
    url = f"https://onepost1.p.rapidapi.com/api/v1/authorizations/{is_id}"
    querystring = {'secret_key': secret_key, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "onepost1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_all_authorizations(secret_key: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Return all authorizations belonging to all Providers."
    
    """
    url = f"https://onepost1.p.rapidapi.com/api/v1/authorizations"
    querystring = {'secret_key': secret_key, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "onepost1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_all_social_posts(secret_key: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all SocialPosts owned by the API user"
    
    """
    url = f"https://onepost1.p.rapidapi.com/api/v1/social_posts"
    querystring = {'secret_key': secret_key, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "onepost1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_a_social_post(is_id: str, secret_key: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get data for a single SocialPost owned by the API user"
    
    """
    url = f"https://onepost1.p.rapidapi.com/api/v1/social_posts/{is_id}"
    querystring = {'secret_key': secret_key, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "onepost1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_an_authorized_page(is_id: str, secret_key: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Gets data for a single AuthorizedPage record"
    
    """
    url = f"https://onepost1.p.rapidapi.com/api/v1/authorized_pages/{is_id}"
    querystring = {'secret_key': secret_key, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "onepost1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_all_authorized_pages(secret_key: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns all AuthorizedPage records owned by the API user"
    
    """
    url = f"https://onepost1.p.rapidapi.com/api/v1/authorized_pages"
    querystring = {'secret_key': secret_key, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "onepost1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_all_posts(secret_key: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all posts that have been created by the API user."
    
    """
    url = f"https://onepost1.p.rapidapi.com/api/v1/posts"
    querystring = {'secret_key': secret_key, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "onepost1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_a_post(secret_key: str, is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Fetch data for a single Post record owned by the API user"
    
    """
    url = f"https://onepost1.p.rapidapi.com/api/v1/posts/{is_id}"
    querystring = {'secret_key': secret_key, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "onepost1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_a_webhook(secret_key: str, is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Return data for a single Webhook owned by the API user"
    
    """
    url = f"https://onepost1.p.rapidapi.com/api/v1/webhooks/{is_id}"
    querystring = {'secret_key': secret_key, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "onepost1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_all_webhooks(secret_key: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Return all data for webhooks owned by the API user"
    
    """
    url = f"https://onepost1.p.rapidapi.com/api/v1/webhooks"
    querystring = {'secret_key': secret_key, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "onepost1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_all_events(secret_key: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Return all event data created on behalf of the API user"
    
    """
    url = f"https://onepost1.p.rapidapi.com/api/v1/events"
    querystring = {'secret_key': secret_key, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "onepost1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_an_event(is_id: str, secret_key: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Return data for a single Event record owned by the API user"
    
    """
    url = f"https://onepost1.p.rapidapi.com/api/v1/events/{is_id}"
    querystring = {'secret_key': secret_key, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "onepost1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_all_images(secret_key: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List all images that have been uploaded."
    
    """
    url = f"https://onepost1.p.rapidapi.com/api/v1/images"
    querystring = {'secret_key': secret_key, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "onepost1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_an_image(is_id: str, secret_key: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get data for a single Image record"
    
    """
    url = f"https://onepost1.p.rapidapi.com/api/v1/images/{is_id}"
    querystring = {'secret_key': secret_key, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "onepost1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_a_post_intent(secret_key: str, is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The [OnepostUI Javascript library](https://github.com/akdarrah/onepost-ui) will create Post Intent records rather than a Post directly. This is due to the fact that the UI library uses the `public_key` rather than the `secret_key` to authenticate. Use this endpoint to inspect the Post Intent record before using the [Create a Post endpoint](https://rapidapi.com/onepost/api/onepost1?endpoint=apiendpoint_7cae6f56-d9c9-4d9c-8c6f-51d0feccb598) to create an actual Post record. *Note:* Specifially, it is important to look at the `authorized_page_ids` of the Post Intent to make sure the user actually has access to the IDs requested."
    
    """
    url = f"https://onepost1.p.rapidapi.com/api/v1/post_intents/{is_id}"
    querystring = {'secret_key': secret_key, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "onepost1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

