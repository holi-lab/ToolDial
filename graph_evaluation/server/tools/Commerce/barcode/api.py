import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def barcode_encode(number: str, totalheight: int=None, barcodeformat: str=None, widthfactor: int=None, outputformat: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a Bar Code image for the given barcode number"
    number: Barcode number
        totalheight: Total height of the image
        barcodeformat: Barcode format default C39. Valid values are the keys to those returned from /barcode/encode/types.
        widthfactor: Width factor of the image
        outputformat: Output image format. Must be one of png/html/jpg/svg
        
    """
    url = f"https://barcode4.p.rapidapi.com/barcode/encode"
    querystring = {'number': number, }
    if totalheight:
        querystring['totalheight'] = totalheight
    if barcodeformat:
        querystring['barcodeformat'] = barcodeformat
    if widthfactor:
        querystring['widthfactor'] = widthfactor
    if outputformat:
        querystring['outputformat'] = outputformat
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "barcode4.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def barcode_decode_types(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the supported barcode types for the decoding process."
    
    """
    url = f"https://barcode4.p.rapidapi.com/barcode/decode/types"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "barcode4.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def barcode_encode_types(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the supported barcode types for encoding / image generation."
    
    """
    url = f"https://barcode4.p.rapidapi.com/barcode/encode/types"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "barcode4.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

