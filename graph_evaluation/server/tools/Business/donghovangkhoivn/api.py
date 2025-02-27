import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def donghovangkhoivn(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Donghovangkhoi.vn Là đơn vị đầu tiên và lâu năm trong lĩnh vực chế tác đồng hồ và trang sức .Tất cả sản phẩm đồng hồ như Đồng hồ rolex fake , đồng hồ hublot fake , đồng hồ patek philippe fake , đồng hồ richard mille fake , đồng hồ omega fake , đồng hồ audemars piguet fake  chế tác đều được làm từ vàng thật 18k , và kim cương thiên nhiên .Đồng Hồ Vàng Khối cam kết tất cả sản phẩm đều được làm từ vàng 18k và kim cương thiên nhiên có đầy đủ kiểm định của nhưng trung tâm kiểm định lớn và uy tín như Doji và GIV 
		#Đồng_Hồ_Vàng_Khối #Donghovangkhoi.vn #đồng_hồ_patek_philippe_fake #donghodeotaynam #donghorichardmillefake
		Hotline/zalo 0854 277 777 
		Gmail: donghovangkhoi1@gmail.com
		WEBSITE  https://www.donghovangkhoi.vn/ 
		https://www.donghovangkhoi.vn/tin-tuc-dong-ho.html 
		https://sites.google.com/view/donghovangkhoi-vn/ 
		https://vi.wikipedia.org/wiki/Đồng_hồ_đeo_tay 
		https://sumally.com/donghovangkhoivn
		https://www.furaffinity.net/user/donghovangkhoivn1/
		https://www.plasterersforum.com/members/donghovangkhoivn.77778/
		https://independent.academia.edu/dkhoivn
		https://donghovangkhoivn.simplesite.com/
		https://scm.cms.hu-berlin.de/donghovangkhoivn
		https://www.podomatic.com/podcasts/donghovangkhoi1
		https://yoo.rs/useroverview/185470/donghovangkhoi1?Ysid=185470
		https://cyber.harvard.edu/cyberlaw_winter10/User_talk:192.168.10.74
		https://forum.cs-cart.com/user/216847-donghovangkhoi-vn/
		https://www.plurk.com/donghovangkhoivn
		https://www.symbaloo.com/profile/donghovangkhoivn
		https://www.theodysseyonline.com/donghovangkhoivn?draft=1
		https://forum.acronis.com/user/403152
		https://www.instapaper.com/p/10739173
		https://pubhtml5.com/homepage/gyzkd
		https://gitlab.tails.boum.org/donghovangkhoivn
		https://paper.li/~/publisher/536ae4ba-8df2-4839-a737-6182b362990a
		https://sketchfab.com/donghovangkhoi_vn
		https://muckrack.com/donghovangkhoi-vn
		https://roundme.com/@donghovangkhoivn/about
		https://www.40billion.com/profile/453258564
		https://www.joindota.com/users/2032846-donghovangkhoivn"
    
    """
    url = f"https://donghovangkhoivn.p.rapidapi.com/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "donghovangkhoivn.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

