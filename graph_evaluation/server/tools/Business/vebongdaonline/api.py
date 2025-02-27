import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def v_b_ng_online(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Vé bóng đá online là trang cung cấp dịch vụ mua vé bóng đá trực tuyến, giao dịch thuận tiện, hỗ trợ nhiều nền tảng thanh toán. Truy cập vebongdaonline.vn để chọn mua những trận đấu bóng đá hấp dẫn từ Việt Nam cho tới Thế Giới. Đặc biệt còn hỗ trợ qua các ví điện tử và tài khoản ngân hàng trực tuyến.
		#vebongda #vebongdaonline #worldcup #vebongdaonline.vn #giaibongda #sanvandongbongda
		Địa chỉ : Tầng 2, Tòa Nhà Vincom Tower, Quận 2, Tp Thủ Đức
		Hotline(+84)336188188
		Gmail: info@vebongdaonline.vn
		Website https://vebongdaonline.vn/ 
		https://vebongdaonline.vn/chuyen-muc/giai-bong-da/ 
		https://vebongdaonline.vn/chuyen-muc/kinh-nghiem-kien-thuc/ 
		https://vebongdaonline.vn/chuyen-muc/san-van-dong-bong-da/ 
		https://www.pinterest.com/vebongdaonline
		https://gfycat.com/@vebongdaonline
		https://git.project-hobbit.eu/vebongdaonline
		https://artistecard.com/vebongdaonline
		https://notionpress.com/author/580540
		https://www.longisland.com/profile/vebongdaonline
		https://www.noteflight.com/profile/c420b141805f5c6b22c7015ce078b6f5d9d8035c
		https://lab.quickbox.io/vebongdaonline
		https://www.wishlistr.com/vebongdaonline
		https://www.hulkshare.com/vebongdaonline
		https://learn.acloud.guru/profile/vebongdaonline
		https://hypothes.is/users/vebongdaonline
		http://www.rohitab.com/discuss/user/563832-vebongdaonline/
		https://dailygram.com/index.php/profile-432493
		https://www.mtbproject.com/user/201409924/vebongda-online
		https://www.gta5-mods.com/users/vebongdaonline
		https://mydramalist.com/profile/vebongdaonline
		https://my.desktopnexus.com/vebongdaonline/
		https://worldcosplay.net/member/1058557
		https://www.tetongravity.com/community/profile/vebongdaonline/
		https://kyteapp.mn.co/members/11832178
		https://www.gaiaonline.com/profiles/vebongdaonline/45917412/
		https://www.bhimchat.com/vebongdaonline
		https://pawoo.net/@vebongdaonline
		http://pixelhub.me/vebongdaonline
		https://www.hackathon.io/vebongdaonline
		https://guides.co/p/vebongda-online
		https://www.allmyfaves.com/vebongdaonline"
    
    """
    url = f"https://vebongdaonline.p.rapidapi.com/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "vebongdaonline.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

