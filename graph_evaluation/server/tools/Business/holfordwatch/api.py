import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def holfordwatch(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Holfordwatch.info là website chia sẻ tất cả mọi thứ (cách chơi, review, mẹo) liên quan đến các tựa game hàng đầu: LMHT, PUBG, game bắn súng, liên quân, game đánh nhau, game bài,… và những kinh nghiệm và mẹo cược esports
		#holfordwatch #cuocesports #gaming #thuthuatgame #huongdanlmht  #cá_cược_esport
		Gmail: holfordwatchinfo@gmail.com
		Địa chỉ : 109 Thùy Vân, Phường Thắng Tam, Thành phố Vũng Tàu, Bà Rịa – Vũng Tàu, Vietnam
		Webiste https://holfordwatch.info/  
		https://holfordwatch.info/top-game/ 
		https://holfordwatch.info/thu-thuat-game/ 
		https://holfordwatch.info/huong-dan-lmht/ 
		https://buddypress.org/members/holfordwatch/profile/
		https://gitlab.pasteur.fr/holfordwatchinfo 
		https://gitlab.manjaro.org/holfordwatch 
		https://gallery.autodesk.com/users/7RA23FTGPSGLEST3 
		https://www.giantbomb.com/profile/holfordwatch/ 
		https://dailygram.com/index.php/profile-427286
		https://www.mtbproject.com/user/201365944/holford-watch
		https://www.gta5-mods.com/users/holfordwatch
		https://mydramalist.com/profile/holfordwatch
		https://worldcosplay.net/member/1045815
		https://onlyfans.com/watchholford
		https://www.tetongravity.com/community/profile/m496c4/
		https://www.mobypicture.com/user/holfordwatch
		https://kyteapp.mn.co/members/11190754
		https://www.gaiaonline.com/profiles/holfordwatch/45855630/
		https://www.bhimchat.com/1652067068674333_15155
		https://pawoo.net/@holfordwatch
		http://pixelhub.me/holfordwatch
		https://repo.getmonero.org/holfordwatch 
		https://www.hackathon.io/users/272297
		https://www.bakespace.com/members/profile/holfordwatch/1474942/
		https://www.allmyfaves.com/holfordwatch/
		https://my.mamul.am/en/profile/8310036/info
		https://kuwestions.248am.com/user/holfordwatch
		https://www.mapleprimes.com/users/holfordwatch
		https://www.bahamaslocal.com/userprofile/1/128539/holfordwatch.html
		https://www.misterpoll.com/users/3623478"
    
    """
    url = f"https://holfordwatch.p.rapidapi.com/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "holfordwatch.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

