import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def dynamic_qr_codes(data: str, eyebottomleft: str='77F62A', eyetopleft: str='#E32AF6', eyestyle: str='frame5', bodystyle: str='atched-rounded', bgcolor: str='#FFFFFF', size: str='500', eyeballstyle: str='ball5', file: str='SVG', bodycolor: str='#5B2C6F', eyetopright: str='#F6D12A', logo: str='#whatsapp', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Generate dynamic QR Codes with custom colors for Qr Code Background, Qr Code Body, eyes and eyeballs.  Choose designs for QR Body, QR eyes and eyeballs. Encode URL, TEXT, EMAIL, PHONE, SMS, VCARD, LOCATION, FACEBOOK, TWITTER, YOUTUBE, WIFI, EVENT, BITCOIN  in your QR Code."
    eyebottomleft: Hexa Color Code for Bottom Left eye.
        eyetopleft: Hexa Color Code for Top Left eye.
        eyestyle: eyeStyle : frame1, frame2, frame3, frame4, frame4, frame5, frame6, frame7, frame8, frame9, frame10, frame11
        bodystyle: Define Qr Code body style from  the styles : diamond, star, 'circle,  dot,  square,  h-line,  v-line, 
 skewed-rect,  rounded-circle,  rounded-rect,  hatched,  atched-rounded,  pointed,  pointed-rounded,  needle,  pointed-diamond, pointed-in,  pointed-in-diamond,  rounded-in,  rounded-in-rounded
        bgcolor: Hexa Color Code. Please use light color for background.
        size: size : 50-2000
        eyeballstyle: eyeBallStyle : ball1, ball2, ball3, ball4, ball5, ball6, ball7, ball8, ball9, ball10, ball11, ball12, ball13, ball14, ball15
        file: filename : PNG, JPG, SVG, PDF
        bodycolor: Hexa Color Code. Please use dark colors for Qr Code Body.
        eyetopright: Hexa Color Code for Top Right eye.
        logo: for using facebook, twitter or whatsapp logo, put #facebook, #twitter, #whatsapp 
otherwise upload logo and use filename, you will get in upload logo API response. 
        
    """
    url = f"https://custom-qr-code1.p.rapidapi.com/custom"
    querystring = {'data': data, }
    if eyebottomleft:
        querystring['eyeBottomLeft'] = eyebottomleft
    if eyetopleft:
        querystring['eyeTopLeft'] = eyetopleft
    if eyestyle:
        querystring['eyeStyle'] = eyestyle
    if bodystyle:
        querystring['bodyStyle'] = bodystyle
    if bgcolor:
        querystring['bgColor'] = bgcolor
    if size:
        querystring['size'] = size
    if eyeballstyle:
        querystring['eyeballStyle'] = eyeballstyle
    if file:
        querystring['file'] = file
    if bodycolor:
        querystring['bodyColor'] = bodycolor
    if eyetopright:
        querystring['eyeTopRight'] = eyetopright
    if logo:
        querystring['logo'] = logo
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "custom-qr-code1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

