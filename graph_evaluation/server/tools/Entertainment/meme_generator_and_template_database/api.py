import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def search(searchstring: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns meme templates that contain a specific string. **searchString is case sensitive**"
    
    """
    url = f"https://meme-generator-and-template-database.p.rapidapi.com/search"
    querystring = {'searchString': searchstring, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "meme-generator-and-template-database.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def fonts(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Any of the fonts can be used  in any text box or caption.
		Some fonts are language specific:
		- Chinese -> zcool-wenyi
		- Japanese -> takaopmincho
		-  Korean -> gugi
		- Hindi -> poppins"
    
    """
    url = f"https://meme-generator-and-template-database.p.rapidapi.com/fonts"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "meme-generator-and-template-database.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def fonts(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Any of the fonts can be used  in any text box or caption.
		Some fonts are language specific:
		- Chinese -> zcool-wenyi
		- Japanese -> takaopmincho
		-  Korean -> gugi
		- Hindi -> poppins"
    
    """
    url = f"https://meme-generator-and-template-database.p.rapidapi.com/fonts"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "meme-generator-and-template-database.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def templates(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns all of the currently supported templates and the number of text boxes that a particular template can support. For how to use this response go to **/template/:image**."
    
    """
    url = f"https://meme-generator-and-template-database.p.rapidapi.com/templates"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "meme-generator-and-template-database.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def templates(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns all of the currently supported templates and the number of text boxes that a particular template can support. For how to use this response go to **/template/:image**."
    
    """
    url = f"https://meme-generator-and-template-database.p.rapidapi.com/templates"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "meme-generator-and-template-database.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

