import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def create_qr_code(url: str, imagetype: str, bgcolor: str='#cbcbcb', fgcolor: str='#ff0000', rimtexttop: str='Scan Me', style: str='default', moduleshape: str='default', rimtextbottom: str='Scan Me', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This API takes an endpoint and return and image for a QR Code in either svg, png, jpg, or pdf format"
    url: This is the URL the QR code will scan to
        imagetype: Response image type:
png, svg, pdf, or jpg
        bgcolor: Hex background color for flowcode.
Format: #rrggbb or transparent for transparent background.
        fgcolor: Hex foreground color for flowcode
Format: #rrggbb
        rimtexttop: For style=rim, the call to action text to put at top of the rim
        style: Style of flowcode:
default, inset, or rim
        moduleshape: Shape to use for code pattern:
default, circle, square, or diamond
        rimtextbottom: For style=rim, the call to action text to put at bottom of the rim
        
    """
    url = f"https://flowcode1.p.rapidapi.com/flowcode"
    querystring = {'url': url, 'imageType': imagetype, }
    if bgcolor:
        querystring['bgColor'] = bgcolor
    if fgcolor:
        querystring['fgColor'] = fgcolor
    if rimtexttop:
        querystring['rimTextTop'] = rimtexttop
    if style:
        querystring['style'] = style
    if moduleshape:
        querystring['moduleShape'] = moduleshape
    if rimtextbottom:
        querystring['rimTextBottom'] = rimtextbottom
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "flowcode1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

