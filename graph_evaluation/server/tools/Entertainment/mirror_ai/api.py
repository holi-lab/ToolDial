import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_all_available_face_parts(x_token: str, face_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get IDs of parts currently applied to the face and available options. Parts may vary from face to face based on recognised attributes."
    x_token: Token you got from /token endpoint
        face_id: Face ID you got from /generate endpoint
        
    """
    url = f"https://mirror-ai.p.rapidapi.com/get_all_parts"
    querystring = {'X-Token': x_token, 'face_id': face_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "mirror-ai.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_face_attributes(x_token: str, face_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get estimated face attributes such as age and gender."
    x_token: Your session token
        face_id: Face ID obtained from Generate endpoint
        
    """
    url = f"https://mirror-ai.p.rapidapi.com/attrs"
    querystring = {'X-Token': x_token, 'face_id': face_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "mirror-ai.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_sticker(x_token: str, sticker: str, face_id: str, loc: str='en', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get URL for a sticker for given face. Supported stickers can be explored at https://mirrorai.github.io
		
		Note that sticker style must match face style used while generating the face.
		
		Currently there are two styles availible:
		* kenga (aka "line" in MirrorAI app)
		* anime"
    
    """
    url = f"https://mirror-ai.p.rapidapi.com/sticker"
    querystring = {'X-Token': x_token, 'sticker': sticker, 'face_id': face_id, }
    if loc:
        querystring['loc'] = loc
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "mirror-ai.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_token(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a token for making further requests. Faces can be managed only with tokens they were generated with."
    
    """
    url = f"https://mirror-ai.p.rapidapi.com/token"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "mirror-ai.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

