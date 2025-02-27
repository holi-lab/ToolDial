import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def recognize_result(recognize_uuid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "gets a faces recognition result."
    recognize_uuid: the requested recognize identifier.
        
    """
    url = f"https://betaface-face-recognition-v1.p.rapidapi.com/recognize"
    querystring = {'recognize_uuid': recognize_uuid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "betaface-face-recognition-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def transform_result(transform_uuid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "gets a faces transform result."
    transform_uuid: the requested transform identifier.
        
    """
    url = f"https://betaface-face-recognition-v1.p.rapidapi.com/transform"
    querystring = {'transform_uuid': transform_uuid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "betaface-face-recognition-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def face_info_cropped(face_uuid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "gets a single cropped face information including cropped face image."
    face_uuid: the requested media identifier.
        
    """
    url = f"https://betaface-face-recognition-v1.p.rapidapi.com/face/cropped"
    querystring = {'face_uuid': face_uuid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "betaface-face-recognition-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def face_info(face_uuid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "gets a single face information."
    face_uuid: the requested media identifier.
        
    """
    url = f"https://betaface-face-recognition-v1.p.rapidapi.com/face"
    querystring = {'face_uuid': face_uuid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "betaface-face-recognition-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_media_info(media_uuid: str='78d6de60-27e2-4b8d-a48e-e402c4f98218', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "gets a media information."
    media_uuid: the requested media identifier.
        
    """
    url = f"https://betaface-face-recognition-v1.p.rapidapi.com/media"
    querystring = {}
    if media_uuid:
        querystring['media_uuid'] = media_uuid
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "betaface-face-recognition-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_media_info_by_hash(checksum: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "gets a media information using SHA256 hash of media file."
    checksum: SHA256 media file hash.
        
    """
    url = f"https://betaface-face-recognition-v1.p.rapidapi.com/media/hash"
    querystring = {'checksum': checksum, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "betaface-face-recognition-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

