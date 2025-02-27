import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def img2pdf(url: str='https://cdn.discordapp.com/attachments/1063839855911653379/1063841657151291392/asvar_Convert_image_to_Pdf_App_a2a8644d-9b22-4471-8cca-3ec91710115d.png', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Converter image file to pdf file!"
    
    """
    url = f"https://img2pdf2.p.rapidapi.com/convert"
    querystring = {}
    if url:
        querystring['url'] = url
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "img2pdf2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

