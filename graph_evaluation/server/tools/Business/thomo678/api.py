import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def thomo67(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Thomo67 thomo678.com ✔️ nhà cái cá cược Đá Gà thomo trực tiếp uy tín số 1️⃣ Việt Nam ✔️ đến với thomo678 bạn sẽ được tặng 6888k thành viên mới  ✔️
		#đá_gà_cựa_sắt #đăng_ký_sv388 #đá_gà_68 #tải_app_sv388 #/m/033_l8  /m/033_l8 
		
		Hotline 0876.684.789
		Địa chỉ : 604 Đ. Bạch Đằng, Bạch Đằng, Hai Bà Trưng, Hà Nội, Việt Nam
		Gmail: thomo678com@gmail.com
		Website https://thomo678.com/   
		https://thomo678.com/da-ga-cua-sat-campuchia-truc-tiep-tu-dau-truong/ 
		https://thomo678.com/dang-ky-tai-khoan-da-ga-sv388-ca-cuoc-online/ 
		https://thomo678.com/tong-quan-ve-daga68/ 
		https://thomo678.com/huong-dan-tai-app-sv388-tren-may-iphone/ 
		https://twinoid.com/user/10054198
		http://recipes.mentaframework.org/user/editDone/217022.page
		https://d.cosx.org/u/thomo678
		https://roomstyler.com/users/thomo678
		http://qooh.me/thomo678
		https://www.exchangle.com/thomo678
		http://onlineboxing.net/jforum/user/editDone/185025.page
		https://svetovalnica.zrc-sazu.si/user/thomo678
		https://support.themecatcher.net/forums/users/thomo678
		https://www.babelcube.com/user/tho-mo678
		https://deepai.org/profile/thomo678
		https://www.yourquote.in/thomo678-com-dkb6a/quotes
		https://ficwad.com/a/thomo678
		https://www.woddal.com/thomo678
		https://englishbaby.com/findfriends/gallery/detail/2404282
		https://storium.com/user/thomo678
		https://worldbeyblade.org/User-thomo678
		https://www.drupalgovcon.org/user/201421
		https://www.mpug.com/members/thomo678/profile/
		https://gettogether.community/profile/36370/
		https://chillspot1.com/user/thomo678
		https://www.pokecommunity.com/member.php?u=1073426
		https://www.ohay.tv/profile/thomo678
		https://www.bigpictureclasses.com/users/thomo678
		http://www.talkstats.com/members/thomo678.143400/#about
		https://lazi.vn/user/mo.tho1
		https://bsaber.com/members/thomo678/
		https://coda.io/@tho-mo678
		https://bikepgh.org/message-board/users/thomo678"
    
    """
    url = f"https://thomo678.p.rapidapi.com/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "thomo678.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

