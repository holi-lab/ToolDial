import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def awin_nha_cai_2022(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "AWIN68 là trang cá cược đảm bảo uy tín hàng đầu trong năm 2022, có máy chủ chính thức được đặt tại Philippines. Chuyên cung cấp đến cho người chơi nhiều sản phẩm game đổi thưởng, dịch vụ trực tuyến với chất lượng vượt trội. Truy cập awin68.info ngay để thưởng thức ngay hàng loạt hoạt động khuyến mãi đình đám hiện đang được cổng game AWIN tổ chức diễn ra.
		#AWIN  #AWIN68 #nhà_cái_AWIN68  #Đăng_nhập_AWIN68  #đăng_ký_AWIN68 #Link_vào_AWIN68_2022 #/m/033_l8  /m/033_l8 #Casino #
		Gmail: awin68info@gmail.com
		Hotline 02446617799
		Địa chỉ : 12 Ngõ 255 Lĩnh Nam, Vĩnh Hưng, Hoàng Mai, Hà Nội 10000, Việt Nam
		Website https://awin68.info/ 
		https://awin68.info/category/huong-dan-choi/ 
		https://awin68.info/category/kinh-nghiem/ 
		https://awin68.info/new-awin68/ 
		https://os.mbed.com/users/awin68/
		https://bom.so/profile/awin68
		https://gfycat.com/ko/@awin68
		https://git.project-hobbit.eu/awin68info
		http://artistecard.com/awin68
		https://notionpress.com/author/575541
		https://www.longisland.com/profile/awin68
		https://www.noteflight.com/profile/e176391d454fb86f07ded9d596d241919d14d41f
		https://lab.quickbox.io/awin68
		https://awin68.bloggersdelight.dk/2022/06/30/awin68/
		https://www.wishlistr.com/awin68/
		https://www.hulkshare.com/awin68
		https://learn.acloud.guru/profile/awin68
		https://hypothes.is/users/awin68
		http://www.rohitab.com/discuss/user/562800-awin68/
		https://dailygram.com/index.php/profile-432307
		https://www.mtbproject.com/user/201408891/awin-68
		https://mydramalist.com/profile/awin68
		https://my.desktopnexus.com/awin68/#ProfileComments
		https://worldcosplay.net/member/1058119
		https://kyteapp.mn.co/members/11814513
		https://fundrazr.com/profiles/awin-68
		https://awin68info.gitbook.io/untitled/
		https://awin68.pixnet.net/blog
		https://www.gaiaonline.com/profiles/awin68/45915543/
		https://www.bhimchat.com/awin68"
    
    """
    url = f"https://awin68.p.rapidapi.com/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "awin68.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

