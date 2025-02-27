import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def keywords(content: str=None, url: str='http://image.everypixel.com/2014.12/67439828186edc79b9be81a4dedea8b03c09a12825b_b.jpg', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "By sending an image to this method you can get a list of suggested keywords. You may specify a number of returned words or a threshold of its minimum score. Just provide num_keywords or threshold parameter to this method."
    content: You can also send an actual image files for auto-tagging.
        url: Image URL to perform auto-tagging on.
        
    """
    url = f"https://everypixel-api.p.rapidapi.com/keywords"
    querystring = {}
    if content:
        querystring['content'] = content
    if url:
        querystring['url'] = url
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "everypixel-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def quality_ugc(content: str=None, url: str='http://image.everypixel.com/2014.12/67439828186edc79b9be81a4dedea8b03c09a12825b_b.jpg', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The main difference between Stock photo scoring and this model is in the training dataset. User-Generated Photo Scoring is a model trained on a 347 000 of user photos from Instagram. Estimation parameters for this model were prepared by a group of 10 professional photographers. Scoring methods are based on five classes: very bad (0-20), bad (20-40), normal (40-60), good (60-80) and excellent (80-100).  This model is designed to evaluate user photos taken both by a professional camera and by a camera of a smartphone. It doesn't estimate the plot and do not measure how cool or beautiful a person or an object on a photo may look. It cares only about technical parts like brightness, contrast, noise and so on. The service is not dedicated for scoring historical photos, illustrations or 3D visualizations."
    content: You can also send an actual image files for scoring.
        url: Image URL to perform scoring on.
        
    """
    url = f"https://everypixel-api.p.rapidapi.com/quality_ugc"
    querystring = {}
    if content:
        querystring['content'] = content
    if url:
        querystring['url'] = url
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "everypixel-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def quality(content: str=None, url: str='http://image.everypixel.com/2014.12/67439828186edc79b9be81a4dedea8b03c09a12825b_b.jpg', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This method allows you to get the quality score for your photo. This service doesn't measure how cool or beautiful a person or an object on a photo may look. It cares only about technical parts like brightness, contrast, noise and so on. The service is not dedicated for scoring historical photos, illustrations or 3D visualizations."
    content: You can also send an actual image files for scoring.
        url: Image URL to perform scoring on.
        
    """
    url = f"https://everypixel-api.p.rapidapi.com/quality"
    querystring = {}
    if content:
        querystring['content'] = content
    if url:
        querystring['url'] = url
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "everypixel-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

