import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def tr_c_ti_p_g_tv(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Trực tiếp đá gà TV là website xem đá gà trực tiếp hôm nay, xem đá gà campuchia, đá gà cựa sắt thomo và chia sẻ thông tin, kinh nghiệm bổ ích về việc nuôi gà cũng như kiến thức nuôi gà hiệu quả. Các giải đá gà lớn nhỏ từ đá gà tre, gà chọi và nhiều giải khác. Hãy nhanh chóng truy cập tructiepdaga.tv để xem ngay các trận đá gà hấp dẫn nhất trong ngày.
		
		#tructiepdagatv #tructiepdaga #dagatructiep #dagathomo #dagacampuchia #đá_gà_trực_tiếp 
		Gmail: tructiepdagatv@gmail.com
		Địa chỉ 101 Đặng Tiến Đông, Đống Đa, Hà Nội
		Google map https://www.google.com/maps?cid=5950869525774567250 
		Hotline 093851 6820
		Website https://tructiepdaga.tv/
		https://sites.google.com/view/tructiepdaga-tv/home 
		https://www.facebook.com/tructiepdagatv 
		https://artmight.com/user/profile/598720
		https://wakelet.com/@tructiepdagatv
		https://www.slideserve.com/tructiepdagatv
		https://www.blackhatworld.com/members/tructiepdagatv.1622266/#about
		https://twitter.com/tructiepdagatv
		https://techbike.vn/members/tructiepdagatv.23367/#about
		https://anyflip.com/homepage/yaeiu
		https://tapas.io/tructiepdagatv
		https://www.helpforenglish.cz/profile/252528-tructiepdagatv
		https://www.historyillinois.org/UserProfile/tabid/3119/UserId/93265/Default.aspx
		https://www.imagekind.com/MemberProfile.aspx?MID=372034c5-3c7b-4c4b-87d7-d09f1636f2dc
		https://www.ultimate-guitar.com/u/tructiepdagatv
		http://tupalo.com/en/users/3433031
		https://www.thingiverse.com/tructiepdagatv/designs
		https://linkhay.com/link/5453332/truc-tiep-da-ga-tv-xem-da-ga-thomo-va-campuchia-online
		https://gettr.com/user/tructiepdagatv
		https://forum.index.hu/Article/showArticle?t=9012640
		https://www.goodreads.com/user/show/152646804-tructiepdagatv
		https://www.leetchi.com/c/tructiepdagatv
		https://www.metal-archives.com/users/tructiepdagatv
		https://os.mbed.com/users/tructiepdagatv/
		https://gfycat.com/@tructiepdagatv
		https://git.project-hobbit.eu/tructiepdagatv
		http://artistecard.com/tructiepdagatv
		https://notionpress.com/author/559447
		https://www.longisland.com/profile/tructiepdagatv
		https://www.noteflight.com/profile/2207205f374b49a3b343dee37bfc6eb15614caea
		https://lab.quickbox.io/tructiepdagatv"
    
    """
    url = f"https://tructiepdagatv.p.rapidapi.com/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tructiepdagatv.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

