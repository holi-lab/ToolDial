import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_24hkhoedep(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "24hkhoedep.com là trang bán lẻ trực tuyến trực thuộc Công Ty TNHH Chăm Sóc Khoẻ Và Đẹp Việt Nam – một công ty chuyên nhập khẩu và phân phối độc quyền cho các hãng dược mỹ phẩm và trang thiết bị y tế từ Châu  u.
		#khoedep #khoedepvietnam #mypham #myphamchauau #duocphamchauau #duocmypham
		#mụn thịt #rối loạn tiêu hóa #ngực chảy xệ
		Địa chỉ : 40/19 Bàu Cát 2, Phường 14, Quận Tân Bình, TP. HCM, Việt Nam
		Hotline 028 - 6296 2262 | 0931320062
		Email: hbcvn@healthbeautycare.com.vn
		Google map https://www.google.com/maps?cid=5320003827413062825 
		Website https://24hkhoedep.com/ 
		https://24hkhoedep.com/mun-thit-la-gi-nguyen-nhan-va-cach-chua-mun-thit-an-toan-hieu-qua/ 
		https://24hkhoedep.com/roi-loan-tieu-hoa-la-gi/ 
		https://24hkhoedep.com/nguc-chay-xe-phai-lam-gi-cach-nang-nguc-chay-xe-tai-nha/ 
		https://www.facebook.com/24hkhoedepHBC 
		https://twitter.com/24hkhoedep 
		https://community.opengroup.org/24hkhoedep 
		https://www.pinterest.com/24hkhoedep/ 
		https://www.instagram.com/comem_homelab 
		https://www.beatstars.com/24hkhoedep/about 
		https://gifyu.com/24hkhoedep
		https://folkd.com/user/24hkhoedep
		https://www.blogger.com/profile/06161746755325811075 
		https://www.youtube.com/channel/UCk-noRhYKMPmPdKN2s1mIlQ 
		https://soundcloud.com/24hkhoedep
		https://www.behance.net/24hkhoedep
		https://www.openstreetmap.org/user/24hkhoedep
		https://issuu.com/24hkhoedep
		https://profile.hatena.ne.jp/hkhoedep/profile
		https://ko-fi.com/24hkhoedep
		https://gitlab-test.eclipse.org/24hkhoedep 
		https://www.twitch.tv/24hkhoedep/about
		https://dribbble.com/24hkhoedep/about
		https://onlyfans.com/khoedep24h 
		https://myspace.com/24hkhoedep
		https://www.goodreads.com/user/show/151102363-24hkhoedep
		https://linktr.ee/24hkhoedep
		https://www.deviantart.com/24hkhoedep
		https://about.me/dep.24hkhoe 
		https://www.yumpu.com/xx/document/view/66823841/24hkhoedep
		https://tawk.to/24hkhoedep 
		https://gallery.autodesk.com/users/5EB2PJGRPF8CEZU6 
		https://sketchfab.com/24hkhoedep 
		https://li.sten.to/24hkhoedep
		https://connect.garmin.com/modern/profile/7b87f84b-717b-463c-9760-9a2f431f11c7"
    
    """
    url = f"https://24hkhoedep.p.rapidapi.com/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "24hkhoedep.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

