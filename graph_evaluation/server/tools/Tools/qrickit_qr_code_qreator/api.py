import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def generate_a_qr_code_image(d: str, logotext: str=None, t: str=None, fgdcolor: str=None, qrsize: int=None, lang: str=None, e: str=None, addtext: str=None, txtcolor: str=None, bgdcolor: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Format of GET request to generate QR Code image. See documentation at https://qrickit.com/qrickit_apps/qrickit_api.php"
    d: Data for QR Code (e.g. URL, text, vCard data, iCal data, etc.) See documentation at https://qrickit.com/qrickit_apps/qrickit_api.php for instructions.
        logotext: Header Text: logotext=Any text about 15 to 35 characters maximum depending on the size of your QR Code. Color of header text is the same as QR Code color (fgdcolor). In cases where data is very large, text will not fit and should not be used. Text can be urlencoded to avoid problems with special characters. 
        t: Type of generated image. p = png (default). j = jpg. g = gif.
        fgdcolor: QR Code Color: fgdcolor=The color of your QR Code expressed in HTML Hex# (i.e. FFFFFF for white, 000000 for black, etc.). If nothing is specified, the default is black (000000).
        qrsize: Size of image in pixels. Default = 150 (i.e. 150 pixel square) . Min = 80. Max = 1480.
        lang: Set to "jp" if your optional footer text (addtext) uses Japanese characters. Otherwise leave blank.
        e: Error Correction: e=The level of error correction capability. The choices are L (Low) 7%, M (Medium) 15%, Q (Quartile) 25%, and H (High) 30% represented as l, m, q, and h respectively. If nothing is specified, the default error correction level is M (Medium). The higher the error correction means more of the QR Code can still be read if it is damaged (or covered/hidden). However, the higher the error correction, the less data the QR Code can hold.  Usually you don't have to specify anything here unless you have a lot of data to fit in the QR Code. Then you can choose e=L (Low).
        addtext: Footer Text: addtext=Any text about 15 to 60 characters maximum depending on the size of your QR Code. In cases where data is very large, text will not fit an d should not be used. Text can be urlencoded to avoid problems with special characters. If using Japanese characters please set "lang" parameter to "jp" (i.e. lang=jp). Otherwise leave "lang" blank.
        txtcolor: Footer Text Color: txtcolor=The color of your optional footer text expressed in HTML Hex# (i.e. FFFFFF for white, 000000 for black, etc.). If nothing is specified, the default is black (000000). 
        bgdcolor: Background Color: bgdcolor=The color of the background expressed in HTML Hex# (i.e. FFFFFF for white, 000000 for black, etc.). If nothing is specified, the default is white (FFFFFF).
        
    """
    url = f"https://qrickit-qr-code-qreator.p.rapidapi.com/api/qrickit.php"
    querystring = {'d': d, }
    if logotext:
        querystring['logotext'] = logotext
    if t:
        querystring['t'] = t
    if fgdcolor:
        querystring['fgdcolor'] = fgdcolor
    if qrsize:
        querystring['qrsize'] = qrsize
    if lang:
        querystring['lang'] = lang
    if e:
        querystring['e'] = e
    if addtext:
        querystring['addtext'] = addtext
    if txtcolor:
        querystring['txtcolor'] = txtcolor
    if bgdcolor:
        querystring['bgdcolor'] = bgdcolor
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "qrickit-qr-code-qreator.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

