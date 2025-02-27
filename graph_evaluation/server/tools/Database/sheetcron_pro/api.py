import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def api(is_id: str, sheet: str, callback: str=None, q: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "API de Sheetcron con la insuperable velocidad de un servidor en NODEJS.
		**ID: id del documento de Google Sheet
		Sheet: nombre de la tab/hoja**"
    id: Id del documento
        sheet: Nombre del tab
        
    """
    url = f"https://sheetcron-pro.p.rapidapi.com/api"
    querystring = {'id': is_id, 'sheet': sheet, }
    if callback:
        querystring['callback'] = callback
    if q:
        querystring['q'] = q
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sheetcron-pro.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

