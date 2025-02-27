import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def sv388_link_nha_cai_da_ga_uy_tin(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "SV388 -🎖️ Link nhà cái đá gà uy tín 
		
		SV388 – Nhà Cái chuyên về đá gà online nổi tiếng hàng đầu châu Á với hàng nghìn trận đấu mỗi ngày tử nhiều loại gà Đá Gà Cựa Sắt, Đá Gà Cựa dao, đá gà thomo,...
		Ngoài ra SV388 còn có rất nhiều trò chơi casino trực tuyến, slot game, bóng đá, lô đề,...Tại Việt Nam, SV388 có nhiều trường gà để nuôi và huấn luyện như trường gà  Thomo Cambodia.
		Về điểm vượt trội của SV388 mạnh về truyền hình trực tiếp và cho phép người chơi đặt cược lớn. Dù só lượng chơi đông  nhưng chất lượng hình ảnh tốt, đường truyền ổn định là điểm cộng rất lớn cho SV388.
		#sv388 #sv388sv288 #sv288 #sv388sv288donet #/m/033_l8  /m/033_l8 #Casino #linkvao #dangnhap  
		Địa chỉ: 12 Đường Thanh Niên, Trúc Bạch, Ba Đình, Hà Nội
		Gmail:sv388sv288net@gmail.com
		Hotline 0877.55.66.33
		Website https://sv388sv288.net/ 
		https://sv388sv288.net/dang-ky-sv388/ 
		https://sv388sv288.net/blog-sv388/ 
		https://sv388sv288.net/blog-da-ga/ 
		https://sites.google.com/view/sv388sv288net 
		https://sketchfab.com/sv388sv288net 
		https://community.amplifi.com/user/sv388-sv288net
		https://www.gaiaonline.com/profiles/sv388sv288net/45895257/
		https://www.bhimchat.com/sv388sv288net
		https://pawoo.net/@sv388sv288net
		http://pixelhub.me/sv388sv288net
		https://www.hackathon.io/sv388sv288net
		https://www.bakespace.com/members/profile/sv388sv288net/1485412/
		https://guides.co/p/sv388-sv288net
		https://www.allmyfaves.com/sv388sv288net
		https://my.mamul.am/en/profile/7247504
		http://www.effecthub.com/people/sv388sv288net
		https://kuwestions.248am.com/user/sv388sv288net
		https://www.mapleprimes.com/users/sv388sv288net
		https://www.bahamaslocal.com/userprofile/134638/sv388sv288net.html
		https://www.misterpoll.com/users/4006009
		https://yarabook.com/sv388sv288net
		https://anchor.fm/sv388sv288net
		https://www.bitsdujour.com/profiles/d04nJd
		https://www.techrum.vn/members/sv388sv288net.203660/#about
		https://www.facer.io/u/sv388sv288net
		https://sv388sv288net.netboard.me/
		https://socialcompare.com/en/member/sv388sv288net-6hvtmbxo
		https://git.qoto.org/sv388sv288net"
    
    """
    url = f"https://sv388-link-nha-cai-da-ga-uy-tin.p.rapidapi.com/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sv388-link-nha-cai-da-ga-uy-tin.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

