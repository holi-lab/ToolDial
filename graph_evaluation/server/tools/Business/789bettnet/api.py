import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_789betlinkvao2022(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "789bet là nhà cái cá cược trực tuyến uy tín top đầu khu vực châu Á, cung cấp các dịch vụ cá cược như casino, cá cược thể thao, nổ hũ, bắn cá, tài xỉu.
		#789bettnet #789bet #nhacai789bet #linkvao #dangky789bet #dangnhap789bet
		Hotline 0985 433 336
		Địa chỉ : 89 Ngõ 15 Giải Phóng, Hoàng Liệt, Hai Bà Trưng, Hà Nội, Việt Nam 
		Google map https://www.google.com/maps?cid=5871956287660487504 
		GmaiL: 789bett.net@gmail.com
		Website https://789bett.net/ 
		https://789bett.net/dang-ky-789bet/ 
		https://789bett.net/khuyen-mai-789bet/ 
		https://sites.google.com/view/789bettnet/home 
		https://dailygram.com/index.php/profile-432167
		https://www.mtbproject.com/user/201407644/789bett-net
		https://mydramalist.com/profile/789bettnet
		https://my.desktopnexus.com/789bettnet/
		https://worldcosplay.net/member/1058048
		https://www.tetongravity.com/community/profile/789bettnet/
		https://community.amplifi.com/user/789bett-net
		https://www.gaiaonline.com/profiles/789bettnet/45915060/
		https://www.hackathon.io/789bettnet
		https://pawoo.net/@789bettnet
		http://pixelhub.me/789bettnet
		https://guides.co/p/789bett-net
		https://www.allmyfaves.com/789bettnet
		https://my.mamul.am/en/profile/1314087/info
		http://www.effecthub.com/people/789bettnet
		https://kuwestions.248am.com/user/789bettnet
		https://www.mapleprimes.com/users/789bettnet
		https://www.bahamaslocal.com/userprofile/137779/789bettnet.html
		https://www.misterpoll.com/users/4137466
		https://yarabook.com/789bettnet
		https://anchor.fm/789bettnet
		https://www.bitsdujour.com/profiles/Vj49if"
    
    """
    url = f"https://789bettnet.p.rapidapi.com/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "789bettnet.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

