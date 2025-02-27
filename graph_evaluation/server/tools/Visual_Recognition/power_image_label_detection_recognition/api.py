import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def alive(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Check the AI is alive"
    
    """
    url = f"https://power-image-label-detection-recognition.p.rapidapi.com/api/imagedetection/status"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "power-image-label-detection-recognition.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def visual_image_objects_from_url(url: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Visual image objects, this will outline objects in the image and return the image highlighted as seen below
		![](https://convertrix.net/OutlineObject.jpeg)"
    
    """
    url = f"https://power-image-label-detection-recognition.p.rapidapi.com/api/imagedetection/getitemdetectvisualurl"
    querystring = {'url': url, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "power-image-label-detection-recognition.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def text_detection_from_url(url: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Optical character recognition (OCR) for an image; text recognition and conversion to machine-coded text. Identifies and extracts UTF-8 text in an image.
		Images are optimized for sparse areas of text within a larger image."
    
    """
    url = f"https://power-image-label-detection-recognition.p.rapidapi.com/api/imagedetection/gettextdetectionurl"
    querystring = {'url': url, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "power-image-label-detection-recognition.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def face_annotation_and_points_from_url(url: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Provides likelihood ratings for the following explicit content categories: adult, spoof, medical, violence, and racy and gives location on the face."
    
    """
    url = f"https://power-image-label-detection-recognition.p.rapidapi.com/api/imagedetection/getfaceannotationurl"
    querystring = {'url': url, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "power-image-label-detection-recognition.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def image_properties_dominant_color_from_url(url: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Image properties will return the dominant colours in the image provided, you can upload the image via post and this will return an array of dominant colours and there score and pixel cover percent."
    
    """
    url = f"https://power-image-label-detection-recognition.p.rapidapi.com/api/imagedetection/getimagepropertiesurl"
    querystring = {'url': url, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "power-image-label-detection-recognition.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def logo_detection_from_url(url: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Provides labels for an image
		Returns each logo and conference rating."
    
    """
    url = f"https://power-image-label-detection-recognition.p.rapidapi.com/api/imagedetection/getlogodetectionurl"
    querystring = {'url': url, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "power-image-label-detection-recognition.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def landmark_detection_from_url(url: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Provides the name of the landmark and a confidence score.
		Gives coordinates for the detected entity."
    
    """
    url = f"https://power-image-label-detection-recognition.p.rapidapi.com/api/imagedetection/getlandmarkdetectionurl"
    querystring = {'url': url, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "power-image-label-detection-recognition.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def image_labels_contains_from_url(url: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint will return objects for an image provided from a URL, for each object, a text label and percentage score will be returned."
    
    """
    url = f"https://power-image-label-detection-recognition.p.rapidapi.com/api/imagedetection/getimagelabelsurl"
    querystring = {'url': url, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "power-image-label-detection-recognition.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

