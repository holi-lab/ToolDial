import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_svg_pattern(name: str='waves-1', scale: int=2, moveleft: int=0, angle: int=100, strokejoin: str='round', movetop: int=0, colors: str='E11D48|rgb(234,179,8)', spacing: str='0|0', stroke: int=3, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get SVG Pattern"
    name: Name of the pattern specified at https://pattern.monster/api
(Default: Random name if not specified)
        scale: 0.5 to Max Scale of the pattern specified at https://pattern.monster/api
(Default: Random scale if not specified)
        moveleft: 0 - 100 ( Default: 0 )
        angle: 0 - 360 (Default: Random angle if not specified)
        strokejoin: round or square 
(Default: round)
Valid only for patterns with Stroke Join as specified at https://pattern.monster/api
        movetop: 0 - 100 ( Default: 0 )
        colors: 2 to Max Colors of the pattern specified at https://pattern.monster/api
Format: Color1|Color2|Color3|Color4|Color5
At least, two colors are required if specified
(Default: Random colors if not specified)
Accepted color values: 
CSS colors ( eg. aqua, red )
RGB or RGBA colors ( eg. rgb(59,130,246), rgba(234,179,8,0.7) )
HSL or HSLA colors ( eg. hsl(300, 76%, 72%), hsla(147,50%,47%,0.5) )
HEX colors without the hash symbol ( eg. E11D48, 2563eb )
        spacing: Horizontal Spacing | Vertical Spacing
(Default: 0|0)
Max Spacing of the pattern specified at https://pattern.monster/api
Valid only for patterns with Max Spacing other than 0|0

        stroke: 0.5 to Max Stroke of the pattern specified at https://pattern.monster/api
(Default: Random stroke if not specified)
Valid only for patterns with Mode as Stroke
        
    """
    url = f"https://pattern-monster.p.rapidapi.com/api/v1/vector"
    querystring = {}
    if name:
        querystring['name'] = name
    if scale:
        querystring['scale'] = scale
    if moveleft:
        querystring['moveLeft'] = moveleft
    if angle:
        querystring['angle'] = angle
    if strokejoin:
        querystring['strokeJoin'] = strokejoin
    if movetop:
        querystring['moveTop'] = movetop
    if colors:
        querystring['colors'] = colors
    if spacing:
        querystring['spacing'] = spacing
    if stroke:
        querystring['stroke'] = stroke
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "pattern-monster.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

