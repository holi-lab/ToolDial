import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def jun88(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Jun88 là nhà cái cá cược trực tuyến uy tín ở khu vực Việt Nam và Đông Nam Á, các sản phẩm cá cược được cung cấp độc quyền bởi tập đoàn Taipei.
		#Jun88live #Dang_ky_Jun88 #Dangnhap_Jun88 #link_vào_jun88 #khuyenmaiJun88 #Casino #bắn_cá #Game_slot #TaiappJun88
		Hotline 0983 833 686
		Gmail: jun888.live@gmail.com
		Địa chỉ : Ng. 8 Đ. Ngọc Hồi, Hoàng Liệt, Hoàng Mai, Hà Nội, Việt Nam 
		Google map https://www.google.com/maps?cid=12453187255415819568 
		Webiste https://jun888.live/ 
		https://jun888.live/dang-ky-jun88/ 
		https://jun888.live/khuyen-mai-jun88/ 
		https://sites.google.com/view/jun888live/tin-tuc-jun888 
		https://fr.ulule.com/jun888live/#/
		https://my.archdaily.com/us/@jun888live
		http://codepad.org/users/jun888live
		https://www.twitch.tv/jun888live/about
		https://www.feedsfloor.com/profile/jun888live
		https://glose.com/u/jun888live
		https://community.tubebuddy.com/index.php?members/106224/
		https://deepai.org/profile/jun888live
		https://discover.events.com/profile/jun888live/3638073/savethedate/
		https://artmight.com/user/profile/596737
		https://wakelet.com/@jun888live
		https://www.slideserve.com/jun888live
		https://www.blackhatworld.com/members/jun888live.1621749/
		https://twitter.com/Jun888L
		https://www.youtube.com/channel/UChIAX4J6vf_ydfzmR8Zwtfg
		https://social.technet.microsoft.com/Profile/jun888live
		https://www.bigbasstabs.com/profile/38480.html
		https://social.msdn.microsoft.com/profile/jun888live/
		https://lightroom.adobe.com/u/jun888live
		http://www.nfomedia.com/profile?uid=rKhRhkI&result=x99fyq5z
		https://www.provenexpert.com/jun888live/"
    
    """
    url = f"https://jun88-thuong-hieu-nha-cai-uy-tin-dang-choi-nhat-nam-2022.p.rapidapi.com/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "jun88-thuong-hieu-nha-cai-uy-tin-dang-choi-nhat-nam-2022.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

