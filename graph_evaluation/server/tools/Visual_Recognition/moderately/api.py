import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def image(url: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Send in an image URL and get a list of inappropriate, unwanted, or offensive labels detected for the uploaded image.
		If no inappropriate content is detected in the image, the response is an empty list.
		Works for images up of size up to 5MB
		
		Properties scanned for:
		- Explicit Nudity
		- Suggestive (i.e. partial nudity)
		- Violence
		- Visually Disturbing
		- Rude Gestures
		- Drugs
		- Tobacco
		- Alcohol
		- Gambling
		- Hate Symbols"
    url: URL of the image to get moderation lables for. Supported formats: JPEG and PNG
        
    """
    url = f"https://moderately.p.rapidapi.com/image"
    querystring = {'url': url, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "moderately.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

