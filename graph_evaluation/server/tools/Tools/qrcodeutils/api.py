import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def qrcodefree(text: str, validate: bool=None, size: int=150, type: str='svg', level: str='M', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Free QR Code Barcode Generator"
    text: Barcode text
        validate: true / false . Default true
        size: Size of the output image in pixels. Max: 250 Default: 150
        type: Warning: the parameter must be named lowercase png,svg or eps
        level: Correction Level L (Low) 7% of codewords can be restored. Level M (Medium) 15% of codewords can be restored. Level Q (Quartile)[39] 25% of codewords can be restored. Level H (High) 30% of codewords can be restored. H,L,Q,M . Default M
        
    """
    url = f"https://qrcodeutils.p.rapidapi.com/qrcodefree"
    querystring = {'text': text, }
    if validate:
        querystring['validate'] = validate
    if size:
        querystring['size'] = size
    if type:
        querystring['type'] = type
    if level:
        querystring['level'] = level
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "qrcodeutils.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def qrcodepro(text: str, validate: bool=None, setlabel: bool=None, forecolor: str='000000', type: str='svg', labeltext: str=None, size: int=150, labelalign: str='center', backcolor: str='FFFFFF', level: str='M', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Pro QR Code Barcode Generator"
    text: Barcode text
        validate: true / false . Default true
        setlabel: true / false . Default false
        forecolor: Foreground color in Hexadecimal value. Note: Please do not enter the # prefix
        type: Warning: the parameter must be named lowercase png,svg or eps
        size: Size of the output image in pixels. Max: 4000 Default: 150
        labelalign: left , right and center. Default center
        backcolor: Background color in Hexadecimal value. Note: Please do not enter the # prefix
        level: Correction Level L (Low) 7% of codewords can be restored. Level M (Medium) 15% of codewords can be restored. Level Q (Quartile)[39] 25% of codewords can be restored. Level H (High) 30% of codewords can be restored. H,L,Q,M . Default M. H,L,Q,M . Default M
        
    """
    url = f"https://qrcodeutils.p.rapidapi.com/qrcodepro"
    querystring = {'text': text, }
    if validate:
        querystring['validate'] = validate
    if setlabel:
        querystring['setlabel'] = setlabel
    if forecolor:
        querystring['forecolor'] = forecolor
    if type:
        querystring['type'] = type
    if labeltext:
        querystring['labeltext'] = labeltext
    if size:
        querystring['size'] = size
    if labelalign:
        querystring['labelalign'] = labelalign
    if backcolor:
        querystring['backcolor'] = backcolor
    if level:
        querystring['level'] = level
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "qrcodeutils.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

