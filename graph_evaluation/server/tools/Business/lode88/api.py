import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def lode88(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "lode88 - lode888.com ⭐️ Là cty lô đề trên mạng, đánh số đề tại LODE88 đảm bảo tỷ lệ ăn cao, rút & nạp tiền nhanh, an toàn. Có bí kíp soi cầu lô đề, số đề...	
		#đăng_ký_lode88 #nạp_tiền_lode88 #tải_ứng_dụng_lode88 # lode88 #rút_tiền_lode88_2022 /m/033_l8  /m/033_l8 #Casino 
		Gmail: lode888com@gmail.com
		Dia chi : 33 ngõ 147 Phố Tân Mai, Tân Mai, Hoàng Mai, Hà Nội, Việt Nam
		Hotline 0987743174
		Webiste https://lode888.com/ 	
		https://lode888.com/dang-ky-lode88/ 
		https://lode888.com/nap-tien-lode88/ 
		https://lode888.com/rut-tien-lode88/ 
		https://lode888.com/tai-ung-dung-lode88/ 	
		https://en.wikipedia.org/wiki/Online_casino  
		https://sites.google.com/view/lode888com 
		https://sites.google.com/view/lode888com/lode88 
		http://uid.me/lode888_com 
		https://gitlab.pasteur.fr/lode888com 
		https://www.emoneyspace.com/lode888com 
		https://lode888com.blogspot.com/2022/04/link-nha-cai-lode-88.html 
		https://lode888com.blogspot.com/ 
		https://www.bitsdujour.com/profiles/RvF1Ky
		https://www.giantbomb.com/profile/lode888com/
		https://www.techrum.vn/members/lode888com.196261/#about
		https://www.facer.io/u/lode888com
		https://lottiefiles.com/lode888com
		https://lode888com.netboard.me/
		https://pantip.com/profile/7002512#topics 
		https://repo.getmonero.org/lode888com 
		https://platform.xr4all.eu/lode888com 
		https://socialcompare.com/en/member/lode888com-6fdx8812
		https://git.qoto.org/-/snippets/2497
		https://www.gapo.vn/lode888com
		https://www.bbuzzart.com/profile/432385
		https://englishbaby.com/findfriends/gallery/detail/2396395
		http://kiredu.ru/UserProfile/tabid/182/userId/82937/Default.aspx
		https://www.thebranfordgroup.com/dnn3/UserProfile/tabid/214/UserId/61166/Default.aspx
		https://beermapping.com/account/lode888com
		http://gendou.com/user/lode888com
		https://www.furaffinity.net/user/lode888com/
		https://www.bitrated.com/lode888com
		https://www.plasterersforum.com/members/lode888com.75657/#about
		https://independent.academia.edu/lode888com
		https://articledirectoryzone.com/members/lode888com/
		https://thetravelbrief.com/tips/hanoi-lode88
		https://www.starters.co/lode888com
		https://lode888com.simplesite.com/
		https://scm.cms.hu-berlin.de/lode888com"
    
    """
    url = f"https://lode88.p.rapidapi.com/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "lode88.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

