import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def harmonies_cielab(a: int, l: int, b: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Displays the Harmonies for one CIELab color."
    l: 0 - 100
        
    """
    url = f"https://convert-colors.p.rapidapi.com/harmonies/cielab/{l}/{a}/{b}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "convert-colors.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def harmonies_android(color: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Displays the Harmonies for one Android color."
    color: The decimal color value
        
    """
    url = f"https://convert-colors.p.rapidapi.com/harmonies/android/{color}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "convert-colors.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def harmonies_cielch(h: int, l: int, c: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Displays the Harmonies for one CIELch color."
    l: 0 - 100
        
    """
    url = f"https://convert-colors.p.rapidapi.com/harmonies/cielch/{l}/{c}/{h}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "convert-colors.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def harmonies_cmy(cyan: int, yellow: int, magenta: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Displays the Harmonies for one CMY color."
    cyan: 0 - 1
        yellow: 0 - 1
        magenta: 0 - 1
        
    """
    url = f"https://convert-colors.p.rapidapi.com/harmonies/cmy/{cyan}/{magenta}/{yellow}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "convert-colors.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def harmonies_cmyk(cyan: int, yellow: int, key: int, magenta: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Displays the Harmonies for one CMYK color."
    cyan: 0 - 1
        yellow: 0 - 1
        key: 0 - 1
        magenta: 0 - 1
        
    """
    url = f"https://convert-colors.p.rapidapi.com/harmonies/cmyk/{cyan}/{magenta}/{yellow}/{key}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "convert-colors.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def harmonies_decimal(color: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Displays the Harmonies for one Decimal color."
    color: The decimal color value
        
    """
    url = f"https://convert-colors.p.rapidapi.com/harmonies/decimal/{color}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "convert-colors.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def harmonies_hex(color: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Displays the Harmonies for one Hex color."
    color: 6 digits color string
        
    """
    url = f"https://convert-colors.p.rapidapi.com/harmonies/hex/{color}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "convert-colors.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def harmonies_hsl(saturation: int, lightness: int, hue: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Displays the Harmonies for one HSL color."
    saturation: 0 - 100
        lightness: 0 - 100
        hue: 0 - 360
        
    """
    url = f"https://convert-colors.p.rapidapi.com/harmonies/hsl/{hue}/{saturation}/{lightness}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "convert-colors.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def harmonies_hsv(saturation: int, hue: int, value: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Displays the Harmonies for one HSV color."
    saturation: 0 - 100
        hue: 0 - 360
        value: 0 - 100
        
    """
    url = f"https://convert-colors.p.rapidapi.com/harmonies/hsv/{hue}/{saturation}/{value}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "convert-colors.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def harmonies_rgb(green: int, red: int, blue: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Displays the Harmonies for one RGB color."
    green: 0 - 255
        red: 0 - 255
        blue: 0 - 255
        
    """
    url = f"https://convert-colors.p.rapidapi.com/harmonies/rgb/{red}/{green}/{blue}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "convert-colors.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def harmonies_rgb_percent(green: int, blue: int, red: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Displays the Harmonies for one RGB Percent color."
    green: 0 - 100
        blue: 0 - 100
        red: 0 - 100
        
    """
    url = f"https://convert-colors.p.rapidapi.com/harmonies/rgbpercent/{red}/{green}/{blue}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "convert-colors.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def harmonies_ryb(red: int, blue: int, yellow: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Displays the Harmonies for one RYB color."
    red: 0 - 255
        blue: 0 - 255
        yellow: 0 - 255
        
    """
    url = f"https://convert-colors.p.rapidapi.com/harmonies/ryb/{red}/{yellow}/{blue}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "convert-colors.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def harmonies_xyz(z: int, y: int, x: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Displays the Harmonies for one XYZ color."
    z: 0 - 100
        y: 0 - 100
        x: 0 - 100
        
    """
    url = f"https://convert-colors.p.rapidapi.com/harmonies/xyz/{x}/{y}/{z}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "convert-colors.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def harmonies_yuv(u: int, v: int, y: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Displays the Harmonies for one YUV color."
    
    """
    url = f"https://convert-colors.p.rapidapi.com/harmonies/yuv/{y}/{u}/{v}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "convert-colors.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def harmonies_yxy(ytwo: int, yone: int, x: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Displays the Harmonies for one Yxy color."
    
    """
    url = f"https://convert-colors.p.rapidapi.com/harmonies/yxy/{yone}/{x}/{ytwo}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "convert-colors.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def convert_android(target: str, color: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Convert one Android color to one of the 15 formats."
    target: 'hex', 'rgb', 'ryb', 'hsl', 'hsv', 'cmy', 'cmyk', 'cielab', 'cielch', 'xyz', 'yxy', 'android', 'decimal', 'yuv', 'rgbpercent'
        color: The decimal color value
        
    """
    url = f"https://convert-colors.p.rapidapi.com/convert/android/{target}/{color}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "convert-colors.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def convert_cielab(a: int, b: int, target: str, l: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Convert one CIELab color to one of the 15 formats."
    target: 'hex', 'rgb', 'ryb', 'hsl', 'hsv', 'cmy', 'cmyk', 'cielab', 'cielch', 'xyz', 'yxy', 'android', 'decimal', 'yuv', 'rgbpercent'
        l: 0 - 100
        
    """
    url = f"https://convert-colors.p.rapidapi.com/convert/cielab/{target}/{l}/{a}/{b}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "convert-colors.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def convert_cielch(h: int, c: int, l: int, target: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Convert one CIELCh color to one of the 15 formats."
    l: 0 - 100
        target: 'hex', 'rgb', 'ryb', 'hsl', 'hsv', 'cmy', 'cmyk', 'cielab', 'cielch', 'xyz', 'yxy', 'android', 'decimal', 'yuv', 'rgbpercent'
        
    """
    url = f"https://convert-colors.p.rapidapi.com/convert/cielch/{target}/{l}/{c}/{h}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "convert-colors.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def convert_cmy(magenta: int, yellow: int, cyan: int, target: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Convert one CMY color to one of the 15 formats."
    magenta: 0 - 1
        yellow: 0 - 1
        cyan: 0 - 1
        target: 'hex', 'rgb', 'ryb', 'hsl', 'hsv', 'cmy', 'cmyk', 'cielab', 'cielch', 'xyz', 'yxy', 'android', 'decimal', 'yuv', 'rgbpercent'
        
    """
    url = f"https://convert-colors.p.rapidapi.com/convert/cmy/{target}/{cyan}/{magenta}/{yellow}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "convert-colors.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def convert_cmyk(cyan: int, key: int, target: str, yellow: int, magenta: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Convert one CMYK color to one of the 15 formats."
    cyan: 0 - 1
        key: 0 - 1
        target: 'hex', 'rgb', 'ryb', 'hsl', 'hsv', 'cmy', 'cmyk', 'cielab', 'cielch', 'xyz', 'yxy', 'android', 'decimal', 'yuv', 'rgbpercent'
        yellow: 0 - 1
        magenta: 0 - 1
        
    """
    url = f"https://convert-colors.p.rapidapi.com/convert/cmyk/{target}/{cyan}/{magenta}/{yellow}/{key}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "convert-colors.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def convert_decimal(target: str, color: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Convert one Decimal color to one of the 15 formats."
    target: 'hex', 'rgb', 'ryb', 'hsl', 'hsv', 'cmy', 'cmyk', 'cielab', 'cielch', 'xyz', 'yxy', 'android', 'decimal', 'yuv', 'rgbpercent'
        color: The decimal color value
        
    """
    url = f"https://convert-colors.p.rapidapi.com/convert/decimal/{target}/{color}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "convert-colors.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def convert_hex(color: str, target: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Convert one HEX color to one of the 15 formats."
    color: 6 digits color string
        target: 'hex', 'rgb', 'ryb', 'hsl', 'hsv', 'cmy', 'cmyk', 'cielab', 'cielch', 'xyz', 'yxy', 'android', 'decimal', 'yuv', 'rgbpercent'
        
    """
    url = f"https://convert-colors.p.rapidapi.com/convert/hex/{target}/{color}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "convert-colors.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def convert_hsl(hue: int, target: str, saturation: int, lightness: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Convert one HSL color to one of the 15 formats."
    hue: 0 - 360
        target: 'hex', 'rgb', 'ryb', 'hsl', 'hsv', 'cmy', 'cmyk', 'cielab', 'cielch', 'xyz', 'yxy', 'android', 'decimal', 'yuv', 'rgbpercent'
        saturation: 0 - 100
        lightness: 0 - 100
        
    """
    url = f"https://convert-colors.p.rapidapi.com/convert/hsl/{target}/{hue}/{saturation}/{lightness}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "convert-colors.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def convert_hsv(hue: int, value: int, saturation: int, target: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Convert one HSV color to one of the 15 formats."
    hue: 0 - 360
        value: 0 - 100
        saturation: 0 - 100
        target: 'hex', 'rgb', 'ryb', 'hsl', 'hsv', 'cmy', 'cmyk', 'cielab', 'cielch', 'xyz', 'yxy', 'android', 'decimal', 'yuv', 'rgbpercent'
        
    """
    url = f"https://convert-colors.p.rapidapi.com/convert/hsv/{target}/{hue}/{saturation}/{value}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "convert-colors.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def convert_rgb(target: str, red: int, green: int, blue: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Convert one RGB color to one of the 15 formats."
    target: 'hex', 'rgb', 'ryb', 'hsl', 'hsv', 'cmy', 'cmyk', 'cielab', 'cielch', 'xyz', 'yxy', 'android', 'decimal', 'yuv', 'rgbpercent'
        red: 0 - 255
        green: 0 - 255
        blue: 0 - 255
        
    """
    url = f"https://convert-colors.p.rapidapi.com/convert/rgb/{target}/{red}/{green}/{blue}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "convert-colors.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def convert_rgb_percent(red: int, green: int, blue: int, target: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Convert one RGB Percent color to one of the 15 formats."
    red: 0 - 100
        green: 0 - 100
        blue: 0 - 100
        target: 'hex', 'rgb', 'ryb', 'hsl', 'hsv', 'cmy', 'cmyk', 'cielab', 'cielch', 'xyz', 'yxy', 'android', 'decimal', 'yuv', 'rgbpercent'
        
    """
    url = f"https://convert-colors.p.rapidapi.com/convert/rgbpercent/{target}/{red}/{green}/{blue}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "convert-colors.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def convert_ryb(blue: int, red: int, target: str, yellow: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Convert one RYB color to one of the 15 formats."
    blue: 0 - 255
        red: 0 - 255
        target: 'hex', 'rgb', 'ryb', 'hsl', 'hsv', 'cmy', 'cmyk', 'cielab', 'cielch', 'xyz', 'yxy', 'android', 'decimal', 'yuv', 'rgbpercent'
        yellow: 0 - 255
        
    """
    url = f"https://convert-colors.p.rapidapi.com/convert/ryb/{target}/{red}/{yellow}/{blue}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "convert-colors.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def convert_xyz(target: str, y: int, z: int, x: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Convert one XYZ color to one of the 15 formats."
    target: 'hex', 'rgb', 'ryb', 'hsl', 'hsv', 'cmy', 'cmyk', 'cielab', 'cielch', 'xyz', 'yxy', 'android', 'decimal', 'yuv', 'rgbpercent'
        y: 0 - 100
        z: 0 - 100
        x: 0 - 100
        
    """
    url = f"https://convert-colors.p.rapidapi.com/convert/xyz/{target}/{x}/{y}/{z}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "convert-colors.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def convert_yuv(y: int, u: int, target: str, v: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Convert one YUV color to one of the 15 formats."
    target: 'hex', 'rgb', 'ryb', 'hsl', 'hsv', 'cmy', 'cmyk', 'cielab', 'cielch', 'xyz', 'yxy', 'android', 'decimal', 'yuv', 'rgbpercent'
        
    """
    url = f"https://convert-colors.p.rapidapi.com/convert/yuv/{target}/{y}/{u}/{v}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "convert-colors.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def convert_yxy(x: int, yone: int, target: str, ytwo: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Convert one Yxy color to one of the 15 formats."
    target: 'hex', 'rgb', 'ryb', 'hsl', 'hsv', 'cmy', 'cmyk', 'cielab', 'cielch', 'xyz', 'yxy', 'android', 'decimal', 'yuv', 'rgbpercent'
        
    """
    url = f"https://convert-colors.p.rapidapi.com/convert/yxy/{target}/{yone}/{x}/{ytwo}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "convert-colors.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def color_format(value: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a color format guess based on the string passed."
    value: Any color string. Some color formats use the same notation, and the API will guess the composition of the input string. As an example, if you pass the string "135°, 96%, 39%" it could be an HSV or HSL color. The API will assume HSL in this case.
        
    """
    url = f"https://convert-colors.p.rapidapi.com/format/{value}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "convert-colors.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def wcag_color_contrast(hexone: str, hextwo: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the WCAG Color Contrast for two Hex Colors."
    
    """
    url = f"https://convert-colors.p.rapidapi.com/wcag/contrast/{hexone}/{hextwo}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "convert-colors.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

