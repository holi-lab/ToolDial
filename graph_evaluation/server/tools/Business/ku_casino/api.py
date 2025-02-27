import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def ku_casino(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "KU casino còn gọi là Kucasino nhà cái cá cược online hàng đầu tại Châu Á. Trang chủ hỗ trợ đăng ký, tải app, đăng nhập Ku casino chính thức https://kucasino1.com tại Việt Nam. Chúng tôi cung cấp các sản phẩm cá cược và dịch vụ giải trí hàng đầu như bóng đá, xổ số, lô đề, casino trực tuyến hấp dẫn nhất.
		#KU casino  #linkvaoKUcasino #kucasino1com #KU casino_nhà_Cái #KU casino_đăng_nhập #KU casino _đăng_ký /m/033_l8  /m/033_l8 #Casino 
		Địa chỉ : 170 Ngõ 141 Giáp Nhị, Thịnh Liệt, Hoàng Mai, Hà Nội, Việt Nam
		Google map https://www.google.com/maps?cid=7650342403876856674 
		Hotline 84569912969
		Gmail: kucasino1.com@gmail.com              
		Website https://kucasino1.com 
		https://kucasino1.com/tai-ku-casino/ 
		https://kucasino1.com/link-ku-casino/ 
		https://kucasino1.com/tin-tuc/ 
		https://en.wikipedia.org/wiki/Online_casino 
		https://sites.google.com/view/kucasino1com
		https://twitter.com/kucasino1com
		https://www.youtube.com/channel/UCMJdKz3UmOxXm4wO0j5YYYg
		https://www.twitch.tv/kucasino1com/about
		https://pbase.com/kucasino1com/profile
		https://kucasino1com.simplesite.com/
		https://scm.cms.hu-berlin.de/kucasino1com
		https://yoo.rs/useroverview/186255/kucasino1.com
		https://www.pedalroom.com/members/kucasino1com
		https://cyber.harvard.edu/cyberlaw_winter10/User_talk:Chuyenvanphongkienvang
		https://www.magcloud.com/user/kucasino1com
		https://forum.cs-cart.com/user/220493-kucasino1com/
		https://www.plurk.com/kucasino1com
		https://www.myminifactory.com/users/kucasino1com
		https://www.surfaceforums.net/members/kucasino1com.42320/
		https://forum.acronis.com/user/405845
		https://www.instapaper.com/p/10815620
		https://pubhtml5.com/homepage/giazu
		https://gitlab.tails.boum.org/kucasino1com
		https://www.dualmonitorbackgrounds.com/kucasino1com
		https://sketchfab.com/kucasino1com
		https://muckrack.com/kucasino1-com/bio
		https://roundme.com/@kucasino1com/about
		https://www.joindota.com/users/2038706-kucasino1com
		https://kienthucykhoa.edu.vn/members/kucasino1com.24265/
		https://community.aodyo.com/user/kucasino1com"
    
    """
    url = f"https://ku-casino.p.rapidapi.com/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ku-casino.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

