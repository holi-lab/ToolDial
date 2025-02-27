import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def think(hint: str, image: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "With this endpoint, you can detect different types of objects,  
		Use the *hint* query to hint us about with the type of algorithm we need to run.
		
		Hint values:
		
		- objects
		Detect object in an image for example with this algorithm we can detect ambulances, cars, and many more things.
		- faces
		The value speaks for itself! 
		
		*Let us do the thinking*!**
		
		And 1 more thing. There are more *hint* values coming soon."
    hint: The are different types of image detection capabilities.

**object**
Detect objects in an image for example you can detect ambulances, cars, etc.

**faces**
With this value, you can detect faces in images.

        
    """
    url = f"https://deepmind.p.rapidapi.com/think"
    querystring = {'hint': hint, 'image': image, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "deepmind.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

