import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def tylekeoprocom(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Tỷ lệ kèo nhà cái trực tuyến, tỷ lệ kèo bóng đá hôm nay: kèo cược chấp châu á, châu âu, kèo tỷ số, tài xỉu… các giải đấu uy tín được cập nhật nhanh chóng và đầy đủ nhất.
		#tylekeopro #Keonhacai #Keobongda  #Tỷ_Lệ_Kèo_Pro /m/033_l8 /m/033_l8 #Casino
		
		Gmail: tylekeopro1@gmail.com
		Hotline 0928893847
		Địa chỉ : 30 Đường D2, Sơn Kỳ, Tân Phú, Thành phố Hồ Chí Minh
		700000
		GG Map https://www.google.com/maps?cid=7377252734704520508 
		Website https://tylekeopro.com/ 
		https://sites.google.com/view/tylekeoprocom
		https://www.pinterest.com/tylekeoprocom/  
		https://tylekeopro1.gitbook.io/ty-le-keo-nha-cai-truc-tuyen-or-ty-le-ca-cuoc-bong/
		https://www.gaiaonline.com/profiles/tylekeoprocom/45941097/
		https://www.bhimchat.com/tylekeoprocom
		https://pawoo.net/@tylekeoprocom
		https://www.hackathon.io/tylekeoprocom
		https://guides.co/p/tylekeopro-com
		https://www.allmyfaves.com/tylekeoprocom
		https://my.mamul.am/en/profile/7848327/info
		http://www.effecthub.com/user/2344491
		https://kuwestions.248am.com/user/tylekeoprocom
		https://www.mapleprimes.com/users/tylekeoprocom
		https://www.bahamaslocal.com/userprofile/1/141369/tylekeoprocom.html
		https://www.misterpoll.com/forums/1/topics/332749
		https://yarabook.com/tylekeoprocom
		https://anchor.fm/tylekeoprocom
		https://www.bitsdujour.com/profiles/qQ4F2N
		https://www.techrum.vn/members/tylekeoprocom.209641/
		https://tylekeoprocom.netboard.me/
		https://socialcompare.com/en/member/tylekeoprocom-6jqc50uu
		https://git.qoto.org/tylekeoprocom
		https://www.gapo.vn/1360351567
		https://www.bbuzzart.com/profile/440122
		https://faithlife.com/tylekeoprocom
		http://kiredu.ru/UserProfile/tabid/182/userId/91375/Default.aspx
		https://www.thebranfordgroup.com/dnn3/UserProfile/tabid/214/UserId/63463/Default.aspx
		https://pinshape.com/users/2371218-tylekeopro1#designs-tab-open
		https://beermapping.com/account/tylekeoprocom
		https://www.beautylish.com/profile/spapicn
		https://sumally.com/tylekeoprocom
		https://play.eslgaming.com/player/18375229/
		https://www.bitrated.com/tylekeoprocom"
    
    """
    url = f"https://tylekeoprocom.p.rapidapi.com/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tylekeoprocom.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

