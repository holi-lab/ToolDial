import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def edit_a_photo(import: str, export: str, export_title: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    ""
    import: the url of the image you want edited
        export: the url where the user is sent after saving
        export_title: the caption used on the save button when editing
        
    """
    url = f"https://ribbet-ribbet.p.rapidapi.com/"
    querystring = {'_import': import, '_export': export, '_export_title': export_title, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ribbet-ribbet.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

