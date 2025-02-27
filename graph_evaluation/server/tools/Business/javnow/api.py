import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def javnow_watch_free_jav_streaming_online_japanese_tube_list(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Javnow is a website that aggregates the latest Japanese high school movie genres in the world.
		Hotline 0859.42.8686
		Address Tân An, Tân Giang, Cao Bằng, Việt Nam
		GG MAP https://www.google.com/maps?cid=13826102472295922551 
		Gmail: javnowlive@gmail.com
		Website https://javnow.live/ 
		https://roundme.com/@javnow/about
		https://ioby.org/users/javnowlive1653381
		https://www.joindota.com/users/2049486-javnow
		http://prsync.com/javnowlive/
		https://community.aodyo.com/user/javnow
		https://mmo4me.com/members/javnow.198145/
		https://myanimelist.net/profile/javnow
		https://communities.bentley.com/members/62473873_2d00_e07b_2d00_41ae_2d00_828b_2d00_b2e279cb115d
		https://pxhere.com/en/photographer/3862456
		https://www.cplusplus.com/user/javnow
		https://www.picfair.com/users/javnow
		https://replit.com/@javnow
		https://leetcode.com/javnow/
		https://www.intensedebate.com/profiles/javnowlive
		http://cloudsdeal.xobor.de/u8062_javnow.html
		https://lnsw-rec.portail.gsitlab.com/user/javnow/home
		https://platform.xr4all.eu/javnowlive1
		https://justpaste.it/u/javnow
		https://pantip.com/profile/7138622#topics
		https://www.emoneyspace.com/javnow
		https://www.veoh.com/users/javnow
		https://app.glosbe.com/profile/6956598608019852528"
    
    """
    url = f"https://javnow.p.rapidapi.com/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "javnow.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

