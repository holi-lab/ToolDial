import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def danienguyen(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Tôi là Danie Nguyễn chuyên gia trong lĩnh vực cá độ đá gà trực tiếp và nhiều kinh nghiệm, kiến thức nuôi gà đá đẳng cấp.
		#danienguyen #tructiepdaga #tructiepdagatv 
		Gmail: danienguyendaga@gmail.com
		Website : https://tructiepdaga.tv/
		https://www.mixcloud.com/danienguyen/
		https://readthedocs.org/projects/danie-nguyen/
		https://danienguyen.simplesite.com/
		https://profile.ameba.jp/ameba/danienguyen
		https://ko-fi.com/danienguyen
		https://fliphtml5.com/homepage/xyqkn
		https://www.themehorse.com/support-forum/users/danienguyendaga
		https://www.discogs.com/fr/user/danienguyen
		https://danienguyendaga.gitbook.io/danie-nguyen/
		https://www.mojomarketplace.com/user/danienguyen-IFvCnE9TZB
		https://padlet.com/danienguyendaga/danienguyen
		https://onlyfans.com/danienguyen
		https://www.thingiverse.com/danienguyen/designs
		https://flipboard.com/profile
		https://jsfiddle.net/danienguyen/tarum8od/
		https://my.archdaily.com/us/@danienguyen
		https://pastebin.com/u/danienguyen
		https://camp-fire.jp/profile/danienguyen
		https://www.credly.com/users/danie-nguyen/badges
		https://telegra.ph/danienguyen-06-28
		https://comicvine.gamespot.com/profile/danienguyen/
		https://band.us/band/88101317
		https://qiita.com/danienguyen
		https://www.scoop.it/u/danienguyen"
    
    """
    url = f"https://danienguyen.p.rapidapi.com/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "danienguyen.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

