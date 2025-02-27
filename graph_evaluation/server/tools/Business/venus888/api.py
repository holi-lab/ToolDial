import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def venus888(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Venus888 - AE8888 - ae8888.net ⭐ Trang chủ đăng ký và đăng nhập của nhà cái ae388 ⭐ Link vào ae888 chính thức để chơi bóng đá, đá gà, tài xỉu, lô đề tại Venus casino.
		#Hướng_dẫn_ae8888 #Game_bài_venus888 #thể_thao_venus888 #Khuyến_mãi_venus888 #ae8888net #Link_vào_Venus888  #nhà_cái_Venus888  #/m/033_l8  /m/033_l8 #Casino 
		Địa chỉ : 199 P. Minh Khai, Minh Khai, Hai Bà Trưng, Hà Nội, Việt Nam
		Hotline 0857090996
		Gmail: ae8888net1@gmail.com 
		Website https://ae8888.net/ 
		https://ae8888.net/chuyen-muc/huong-dan-ae8888/ 
		https://ae8888.net/chuyen-muc/game-bai/ 
		https://ae8888.net/chuyen-muc/the-thao/ 
		https://ae8888.net/chuyen-muc/khuyen-mai/ 
		https://en.wikipedia.org/wiki/Online_casino   
		http://www.effecthub.com/user/2315194
		https://kuwestions.248am.com/user/ae8888net
		https://www.mapleprimes.com/users/ae8888net
		https://www.bahamaslocal.com/userprofile/1/138241/ae8888net.html
		https://www.misterpoll.com/forums/1/topics/331882/pg/1
		https://yarabook.com/ae8888net
		https://anchor.fm/ae8888net
		https://www.bitsdujour.com/profiles/FrEdYT
		https://www.techrum.vn/members/ae8888net.206783/
		https://www.facer.io/u/ae8888net
		https://ae8888net.netboard.me/
		https://socialcompare.com/en/member/ae8888net-6it0et02
		https://git.qoto.org/ae8888net
		https://www.gapo.vn/1636544593
		https://www.bbuzzart.com/profile/438270
		https://faithlife.com/ae8888net
		http://kiredu.ru/UserProfile/tabid/182/userId/89482/Default.aspx
		https://www.thebranfordgroup.com/dnn3/UserProfile/tabid/214/UserId/62922/Default.aspx
		https://pinshape.com/users/2362055-ae8888net#designs-tab-open
		https://beermapping.com/account/ae8888net
		https://sumally.com/ae8888net
		https://www.furaffinity.net/user/ae8888net/
		https://www.bitrated.com/ae8888net
		http://yourlisten.com/ae8888net
		https://www.plasterersforum.com/members/ae8888net.79114/#about
		https://www.jigsawplanet.com/ae8888net?viewas=08e8be7ee06b
		https://independent.academia.edu/ae8888net
		https://pbase.com/ae8888net/profile
		https://scm.cms.hu-berlin.de/ae8888net
		https://yoo.rs/useroverview/186770/ae8888net1
		https://www.pedalroom.com/members/ae8888net
		https://confengine.com/user/ae8888-net
		https://www.magcloud.com/user/ae8888net
		https://forum.cs-cart.com/user/222486-ae8888net/
		https://www.plurk.com/ae8888net
		https://www.myminifactory.com/users/ae8888net
		https://www.surfaceforums.net/members/ae8888net.42508/"
    
    """
    url = f"https://venus888.p.rapidapi.com/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "venus888.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

