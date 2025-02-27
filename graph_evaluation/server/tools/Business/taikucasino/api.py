import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def taikucasino(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "taikucasino.app trang tải app kubet chính thức. Tải Ku casino, KU casinp app, KU app, App ku casino phiên bản mới nhất 2022 cho iPhone và Android tại taikucasino.app - Website chính thức của KU Casino. #taikubet #taiappkubet #kubet  /m/033_l8  /m/033_l8 #Casino 
		Địa chỉ Ng. 37 Đại Đồng, Thanh Trì, Hoàng Mai, Hà Nội, Việt Nam
		Hotline 0868269875
		Webiste https://taikucasino.app/ 
		https://sites.google.com/view/taikucasinoapp/link-vao-app-kucasino https://taikucasinoapp.blogspot.com/ 
		https://www.facebook.com/taikucasinoapp/ 
		https://en.wikipedia.org/wiki/Online_casino   
		https://www.youtube.com/channel/UCU-sWEy4MNBOBDO-xL_cIcw/about
		https://www.pinterest.com/taikucasinoapp
		https://repo.getmonero.org/taikucasinoapp 
		https://taikucasinoapp.bandcamp.com/album/taikucasino
		https://www.twitch.tv/taikucasinoapp/about
		https://ok.ru/taikucasinoapp/statuses/154774551158129
		https://www.liveinternet.ru/users/taikucasinoapp/blog#post491959511
		https://profile.hatena.ne.jp/taikucasinoapp/profile
		https://issuu.com/taikucasinoapp
		https://disqus.com/by/taikucasinoapp/ 
		https://bitbucket.org/taikucasinoapp-admin/workspace/snippets/qX5Aoz
		https://www.reverbnation.com/artist/taikucasinoapp
		https://note.com/taikucasinoapp/
		https://www.provenexpert.com/taikucasinoapp/
		https://gallery.autodesk.com/users/8LXJGVXTJ2RVU4UW 
		https://ko-fi.com/taikucasinoapp 
		https://tawk.to/taikucasinoapp
		https://about.me/taikucasinoapp  
		https://connect.garmin.com/modern/profile/78190a96-9e26-4587-8ef6-595bee9da173 
		https://gitlab-test.eclipse.org/taikucasinoapp1 
		https://www.veoh.com/users/taikucasinoapp 
		https://www.giantbomb.com/profile/taikucasinoapp/"
    
    """
    url = f"https://taikucasino.p.rapidapi.com/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "taikucasino.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

