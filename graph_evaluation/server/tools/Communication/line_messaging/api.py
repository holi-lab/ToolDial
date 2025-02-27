import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_content(messageid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Gets image, video, and audio data sent by users."
    messageid: Message ID
        
    """
    url = f"https://line-messaging1.p.rapidapi.com/bot/message/{messageid}/content"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "line-messaging1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_rich_menu_id_of_user(userid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Gets the ID of the rich menu linked to a user."
    userid: User ID. Found in the source object of webhook event objects. Do not use the LINE ID used in LINE.
        
    """
    url = f"https://line-messaging1.p.rapidapi.com/bot/user/{userid}/richmenu"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "line-messaging1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def download_rich_menu_image(richmenuid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Downloads an image associated with a rich menu."
    richmenuid: ID of the rich menu with the image to be downloaded
        
    """
    url = f"https://line-messaging1.p.rapidapi.com/bot/richmenu/{richmenuid}/content"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "line-messaging1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_rich_menu_list(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Gets a list of all uploaded rich menus."
    
    """
    url = f"https://line-messaging1.p.rapidapi.com/bot/richmenu/list"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "line-messaging1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_rich_menu(richmenuid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Gets a rich menu via a rich menu ID."
    richmenuid: ID of an uploaded rich menu
        
    """
    url = f"https://line-messaging1.p.rapidapi.com/bot/richmenu/{richmenuid}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "line-messaging1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_profile(userid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Gets user profile information."
    userid: User ID that is returned in a webhook event object. Do not use the LINE ID found on LINE.
        
    """
    url = f"https://line-messaging1.p.rapidapi.com/bot/profile/{userid}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "line-messaging1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_default_rich_menu_id(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Gets the ID of the default rich menu set with the Messaging API."
    
    """
    url = f"https://line-messaging1.p.rapidapi.com/bot/user/all/richmenu"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "line-messaging1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_number_of_sent_reply_messages(date: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Gets the number of messages sent with the /bot/message/reply endpoint."
    date: Date the messages were sent. Format: yyyyMMdd (Example: 20191231) Timezone: UTC+9
        
    """
    url = f"https://line-messaging1.p.rapidapi.com/bot/message/delivery/reply"
    querystring = {'date	': date, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "line-messaging1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

