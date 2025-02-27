import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def da_ga_truc_tiep_xem_da_ga_cua_dao_cua_sat_thomo(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Đá gà trực tiếp ⭐️ Xem đá gà cựa sắt thomo & cựa dao là hình thức chơi cá cược phổ biến hiện nay ⭐️ được phát trực tiếp 12h mỗi ngày trên daga988.com
		#Đá_Gà_Campuchia #Đá_Gà_Cựa Dao #Đá_Gà_SV388 #daga988com #daga988 #Đá Gà_Thomo
		Địa chỉ : 281 Ngô Xuân Quảng, Trâu Quỳ, Gia Lâm, Hà Nội, Việt Nam
		Hotline 02438769605
		Gmail: daga988com@gmail.com
		Website https://daga988.com/ 
		https://daga988.com/da-ga-campuchia/ 
		https://daga988.com/da-ga-cua-dao/ 
		https://daga988.com/da-ga-sv388/ 
		https://daga988.com/da-ga-thomo/ 
		https://sites.google.com/view/daga988/home 
		https://comicvine.gamespot.com/profile/daga988/
		https://gifyu.com/daga988
		https://phab.mercurial-scm.org/p/daga988/
		http://daga988.wikidot.com/system:welcome
		https://bikepgh.org/message-board/users/daga988/
		https://gotartwork.com/Profile/da-ga988/147713/
		https://stocktwits.com/daga988
		https://connect.garmin.com/modern/profile/d4a94e67-d5cf-421e-8d0f-088eaccdbad8
		https://ko-fi.com/daga988#paypalModal
		https://linktr.ee/daga988
		http://uid.me/daga988
		https://www.scribblemaps.com/maps/view/da--ga988/daga988
		https://gitlab.manjaro.org/daga988
		https://gitlab.pasteur.fr/daga988
		https://repo.getmonero.org/daga988
		https://career.habr.com/daga988
		https://gitlab.haskell.org/daga988
		https://cartoonmovement.com/cartoonist/18610
		https://forum.umbandaeucurto.com/usuario/daga988
		https://www.pinterest.com/daga988com/_saved/
		https://www.behance.net/daga988
		https://vi.gravatar.com/daga988
		https://www.jigsawplanet.com/daga988?viewas=28c954d96133
		https://binaryoptionrobotinfo.com/forums/users/daga988
		https://www.divephotoguide.com/user/daga988
		https://mootools.net/forge/profile/daga988
		https://fr.ulule.com/daga988com/#/
		https://my.archdaily.com/us/@daga988com
		http://codepad.org/users/daga988
		https://www.twitch.tv/daga988/about
		https://www.feedsfloor.com/profile/daga988
		https://glose.com/u/daga988
		https://community.tubebuddy.com/index.php?members/104507/"
    
    """
    url = f"https://da-ga-truc-tiep-xem-da-ga-cua-dao-cua-sat-thomo.p.rapidapi.com/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "da-ga-truc-tiep-xem-da-ga-cua-dao-cua-sat-thomo.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

