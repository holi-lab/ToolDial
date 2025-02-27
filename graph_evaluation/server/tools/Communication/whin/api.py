import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def personal_listener(gid: str='1234567890123', origin: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The response is an url that you have to use as a webhook destination at the origin service. The content sent to it will be routed as a message to a whatsapp group owned by the user, or to the user.
		Learn use-cases on these videos: [grafana alerts](https://youtu.be/RvVgg0qwNDk),[Shelly notifications](https://youtu.be/B6MLlBUkyvo), [synology notifications](https://youtu.be/zeAJNuXYqH4)."
    gid: Each whatsapp group will have its own webhook address; set the group-id on this field to get a unique valid url and deliver webhooks as messages on each whatsapp group. Learn how to get a valid value [here](https://rapidapi.com/inutil-inutil-default/api/whin2/tutorials/how-to-use-the-groups-category-to-send-and-receive-whatsapps-to%2Ffrom-groups%3F-1).
        origin: This is the service that is sending Webhooks to your whin personal receiver. If the origin service is nos listed you can select `Do not include in request`, or leave the value blank, and whin will do its best to parse the payload received.
        
    """
    url = f"https://whin2.p.rapidapi.com/webhk"
    querystring = {}
    if gid:
        querystring['gid'] = gid
    if origin:
        querystring['origin'] = origin
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "whin2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def create_a_group(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This API will create a group, it will add you, and will promote you to Admin of the group. Check this [video](https://youtu.be/wD0DWoua0L4) to learn how to use it."
    
    """
    url = f"https://whin2.p.rapidapi.com/creategroup"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "whin2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def group_invite_link(gid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "the endpoint returns an invite link url for a group you own."
    gid: the group id of which you want to get an invite link url
        
    """
    url = f"https://whin2.p.rapidapi.com/getlink"
    querystring = {'gid': gid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "whin2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def list_groups(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint returns a list of the groups you created through whin."
    
    """
    url = f"https://whin2.p.rapidapi.com/mygroups"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "whin2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def hk_receiver(user: str, gid: str='120363044823414490', service: str='origin-service', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "this endpoint shall be used by the origin services, it's **NOT** to be used on the playground."
    
    """
    url = f"https://whin2.p.rapidapi.com/hk/{user}/{service}"
    querystring = {}
    if gid:
        querystring['gid'] = gid
    if service:
        querystring['service'] = service
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "whin2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def hook_receiver(user: str, gid: str='120363044823414490', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint shall be triggered by the webhook origin and is not intended to be used through this frontend."
    
    """
    url = f"https://whin2.p.rapidapi.com/hook/{user}"
    querystring = {}
    if gid:
        querystring['gid'] = gid
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "whin2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def show_url(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint is used to request the webhook routes created on your account. 
		Learn how to use it on this [video](https://youtu.be/8WyG_becZXM)"
    
    """
    url = f"https://whin2.p.rapidapi.com/showurl"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "whin2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def websocket_checker(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Whin-receive (thick-client) websockets checker. 
		This endpoint is **ONLY** used to authorize websocket connections to the backend by thick clients. The Response is a token needed to complete handshaking."
    
    """
    url = f"https://whin2.p.rapidapi.com/wskchk"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "whin2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def signup(code: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint is **ONLY** used to enter the code you received to signup. Follow this [video](https://youtu.be/uOZ-oH4kP58), or read the [tutorial](https://rapidapi.com/inutil-inutil-default/api/whin2/tutorials/what-to-do-to-start-using-whin%3F-1), to learn how to start using whin."
    
    """
    url = f"https://whin2.p.rapidapi.com/signup"
    querystring = {'code': code, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "whin2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def delete_url(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint will erase the url you set to receive the whatsapps sent to whin. Learn how to use it on this [video](https://youtu.be/8WyG_becZXM)."
    
    """
    url = f"https://whin2.p.rapidapi.com/delurl"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "whin2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

