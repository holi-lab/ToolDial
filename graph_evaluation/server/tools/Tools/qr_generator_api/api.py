import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def generate(text: str, pixelspermodule: int=10, backcolor: str='#ffffff', forecolor: str='#000000', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Generate HTML image-tag with base64-image-string as QR code of input text (Query Parameter)"
    text: QR Code Text Content
        pixelspermodule: (Optional) The pixel size each b/w module is drawn (Default: 10)
        backcolor: (Optional) Background color in hexadecimal value (Default: White = #ffffff). Note: Should start with # prefix, and each basic-color (red, green, blue) should has two hex-digits.
        forecolor: (Optional) Foreground color in hexadecimal value (Default: Black = #000000). Note: Should start with # prefix, and each basic-color (red, green, blue) should has two hex-digits.
        
    """
    url = f"https://qr-generator-api.p.rapidapi.com/api/qrcode/generate"
    querystring = {'text': text, }
    if pixelspermodule:
        querystring['pixelsPerModule'] = pixelspermodule
    if backcolor:
        querystring['backColor'] = backcolor
    if forecolor:
        querystring['foreColor'] = forecolor
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "qr-generator-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def generate_invoice_vat_qr(date: str, tax: int, vatno: str, seller: str, total: int, pixelspermodule: int=5, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Generate Invoice VAT QR image file stream (KSA VAT Format)."
    date: Invoice Date (format: yyyy-mm-dd)
        tax: Invoice VAT Tax
        vatno: Seller VAT Number
        seller: Seller Name
        total: Invoice Total
        pixelspermodule: (Optional) The pixel size each b/w module is drawn (Default: 5)
        
    """
    url = f"https://qr-generator-api.p.rapidapi.com/api/qrcode/generateinvoicevatqr"
    querystring = {'date': date, 'tax': tax, 'vatNo': vatno, 'seller': seller, 'total': total, }
    if pixelspermodule:
        querystring['pixelsPerModule'] = pixelspermodule
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "qr-generator-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def generate_file(text: str, backcolor: str='#ffffff', pixelspermodule: int=10, forecolor: str='#000000', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Generate image file stream as QR code of input text (Query Parameter)"
    text: QR Code Text Content.
        backcolor: (Optional) Background color in hexadecimal value (Default: White = #ffffff). Note: Should start with # prefix, and each basic-color (red, green, blue) should has two hex-digits.
        pixelspermodule: (Optional) The pixel size each b/w module is drawn (Default: 10)
        forecolor: (Optional) Foreground color in hexadecimal value (Default: Black = #000000). Note: Should start with # prefix, and each basic-color (red, green, blue) should has two hex-digits.
        
    """
    url = f"https://qr-generator-api.p.rapidapi.com/api/qrcode/generatefile"
    querystring = {'text': text, }
    if backcolor:
        querystring['backColor'] = backcolor
    if pixelspermodule:
        querystring['pixelsPerModule'] = pixelspermodule
    if forecolor:
        querystring['foreColor'] = forecolor
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "qr-generator-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

