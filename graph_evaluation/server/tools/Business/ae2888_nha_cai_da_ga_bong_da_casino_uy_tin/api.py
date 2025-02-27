import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def ae2888(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "AE2888 - ae288vn.com ✔️ trò đánh bạc trực tuyến hấp dẫn nhất, được giới thiệu bởi sòng bạc Việt Nam, nhiều trò chơi sội động, AE288 ✔️ chắc chắn là sự lựa chọn tốt nhất AE3888
		#đăng_ký_ae2888 #link_vào_ae2888 #tải_app_ae888 #đá_gà_mạng_trực_tuyến #dangnhapae2888moi2022  /m/033_l8  /m/033_l8 #Casino 
		Gmail: ae288vncom@gmail.com
		Địa chỉ 14 P. Minh Khai, Mai Động, Hoàng Mai, Hà Nội, Việt Nam
		Hotline 0832.633.777
		Website https://ae288vn.com/ 
		https://ae288vn.com/ae2888-com-dang-ky-ae2888-ae888-tai-ae288vn-com/ 
		https://ae288vn.com/link-vao-ae2888-ae288-nhanh-nhat/ 
		https://ae288vn.com/huong-dan-tai-ae888-chi-tiet/ 
		https://ae288vn.com/da-ga-mang-truc-tuyen-la-gi/ 
		https://en.wikipedia.org/wiki/Online_casino 
		https://www.bitrated.com/ae288vn
		http://yourlisten.com/ae288vn
		https://www.plasterersforum.com/members/ae288vn.78165/
		https://independent.academia.edu/ae288vn
		https://scm.cms.hu-berlin.de/ae288vn
		https://www.podomatic.com/podcasts/ae288vncom
		https://yoo.rs/useroverview/185858/ae288vncom?Ysid=185858
		https://www.pedalroom.com/members/ae288vn
		https://cyber.harvard.edu/cyberlaw_winter10/User_talk:192.168.10.74
		https://www.magcloud.com/user/ae288vn
		https://forum.cs-cart.com/user/218437-ae288vn/
		https://www.plurk.com/ae288vn
		https://www.symbaloo.com/profile/ae288vn
		https://www.myminifactory.com/users/ae288vn
		https://www.surfaceforums.net/members/ae288vn.42171/
		https://forum.acronis.com/user/404368
		https://www.instapaper.com/p/10774549
		https://pubhtml5.com/homepage/eozi
		https://gitlab.tails.boum.org/ae288vn
		https://www.dualmonitorbackgrounds.com/ae288vn
		https://sketchfab.com/ae288vn
		https://muckrack.com/ae288-vn/bio
		https://roundme.com/@ae288vn/about
		https://www.40billion.com/profile/486326068"
    
    """
    url = f"https://ae2888-nha-cai-da-ga-bong-da-casino-uy-tin.p.rapidapi.com/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ae2888-nha-cai-da-ga-bong-da-casino-uy-tin.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

