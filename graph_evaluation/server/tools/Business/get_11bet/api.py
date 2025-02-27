import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_11bet_link(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "11BET ⭐️trang  Web cá cược thể thao, bóng đá đỉnh cao uy tín nhất Việt Nam ⭐️ Có những ưu đãi lớn cho người chơi, Link truy cập vào nhà cái 11bet mới nhất.	
		#đăng ký 11bet #nạp tiền 11bet #tải ứng dụng 11bet #rút tiền_11bet  #/m/033_l8  /m/033_l8 #Casino
		Địa chỉ : 43 Lý Thánh Tông, An Đào, Đa Tốn, Gia Lâm, Hà Nội, Việt Nam
		Hotline 0376202879
		Gmail: 11betlink@gmail.com	
		Website https://11bet.link/  	
		https://11bet.link/dang-ky-11bet/  	
		https://11bet.link/nap-tien-11bet/    	
		https://11bet.link/rut-tien-11bet/
		https://11bet.link/tai-ung-dung-11bet/  	
		https://sites.google.com/view/11betlink/11bet-di-dong-link-vao-nha-cai-11bet-tren-mobile 
		https://sites.google.com/view/11betlink 
		https://11betlink.blogspot.com/ 	
		https://en.wikipedia.org/wiki/Online_casino
		https://issuu.com/11betlink?
		https://profile.hatena.ne.jp/betlink11/
		https://www.liveinternet.ru/users/nhacaibetlink/profile
		https://www.openstreetmap.org/user/11betlink
		https://ok.ru/profile/589311685548
		https://linktr.ee/11betlink
		https://www.bark.com/en/gb/company/11betlink/gyvgQ/
		http://community.getvideostream.com/user/11betlink
		https://www.110designs.com/profile/28077-11betlink
		https://myspace.com/11betlink
		https://hub.docker.com/u/11betlink
		https://about.me/link11bet
		https://start.me/p/ADxp7n/11bet
		https://tawk.to/11betlink
		https://www.doyoubuzz.com/11bet-link
		https://www.reverbnation.com/11betlink
		https://note.com/11betlink/
		https://www.discogs.com/fr/user/11betlink
		https://fliphtml5.com/homepage/sfodp
		https://www.producthunt.com/@11bet_link
		https://gallery.autodesk.com/users/AMUJJ9W5QP6HQP8T?relProf=1
		https://support.wedesignthemes.com/users/11betlink/
		https://descubre.beqbe.com/p/-11betlink
		https://ko-fi.com/11betlink
		http://wpc.hotlog.ru/profile.php?user_id=423307"
    
    """
    url = f"https://11bet.p.rapidapi.com/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "11bet.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

