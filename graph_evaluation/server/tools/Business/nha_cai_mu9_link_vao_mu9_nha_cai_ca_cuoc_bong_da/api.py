import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def nha_cai_mu9_link_vao_mu9_nha_cai_ca_cuoc_bong_da(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "MU9 là nhà cái uy tín số 1 về cá cược bóng đá, game bài, nổ hũ, bắn cá, xổ số lô đề, live casino,...đứng top 1 tại thị trường Anh Quốc. Đăng ký Mu9 để nhận khuyến mãi siêu hot, Link đăng nhập Mu9 chính thức 2022!  
		#Mu9_nhà_cái #Mu9 #Link_đăng_nhập_Mu9 #/m/033_l8  /m/033_l8 #Casino #naptienMU9 #dangkyMu9 #linkvaonhacaiMu92022
		Địa chỉ: 15 Đ. Tôn Đức Thắng, Phường Phạm Ngũ Lão, Quận 1, Thành phố Hồ Chí Minh
		Hotline 0961707270
		Gmail: mu9city@gmail.com
		Website https://mu9.city/  
		https://mu9.city/casino-online/ 
		https://mu9.city/top-nha-cai/ 
		https://en.wikipedia.org/wiki/Online_casino
		https://www.facebook.com/mu9city 
		https://www.youtube.com/channel/UC2lvGuYin0ZbYQ6GIs_iyEA/about 
		https://twitter.com/mu9city 
		https://www.pinterest.com/mu9city/ 
		https://gallery.autodesk.com/users/XTF26UVBPHYHLMT8 
		https://www.youtube.com/channel/UC2lvGuYin0ZbYQ6GIs_iyEA/about
		https://twitter.com/mu9city
		https://www.pinterest.com/mu9city/
		https://www.cplusplus.com/user/mu9city
		https://mootools.net/forge/profile/mu9city
		https://community.dynamics.com/members/mu9city
		https://themehunt.com/profile/mu9city
		https://www.picfair.com/users/mu9city
		https://www.elephantjournal.com/profile/mu9city/
		https://play.eslgaming.com/player/18338907/
		https://www.furaffinity.net/user/mu9city/
		https://folkd.com/user/mu9city
		https://micro.blog/mu9city
		https://xipnios.tribe.so/user/mu9city
		https://www.intensedebate.com/profiles/mu9city
		https://www.lifeofpix.com/photographers/mu9city/
		https://community.tubebuddy.com/index.php?members/107482/
		https://www.emoneyspace.com/mu9city
		https://tapas.io/mu9city
		https://www.360cities.net/profile/mu9city
		https://rapidapi.com/user/mu9city
		https://www.ultimate-guitar.com/u/mu9city
		https://pantip.com/profile/7118598#topics
		https://vocal.media/authors/mu9city
		https://coub.com/mu9city24088
		https://8tracks.com/mu9city
		https://www.myminifactory.com/users/mu9city
		https://www.giantbomb.com/profile/mu9city/
		https://allmylinks.com/mu9city
		https://gifyu.com/mu9city
		https://justpaste.it/u/mu9city
		https://linkhay.com/link/5475663/mu9city
		https://www.imagekind.com/MemberProfile.aspx?MID=14f530cd-1848-4175-8b43-99a843e2738a
		https://www.metal-archives.com/users/mu9city
		https://www.jigsawplanet.com/mu9city?viewas=152db2197306
		https://developers.oxwall.com/user/mu9city
		https://www.mobygames.com/user/sheet/userSheetId,893719/
		https://www.veoh.com/users/mu9city
		https://app.roll20.net/users/10725232/mu9city
		https://app.lookbook.nu/mu9city"
    
    """
    url = f"https://nha-cai-mu9-link-vao-mu9-nha-cai-ca-cuoc-bong-da.p.rapidapi.com/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "nha-cai-mu9-link-vao-mu9-nha-cai-ca-cuoc-bong-da.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

