"""게시판 목록 수집기.

GWDownloader의 로그인 세션으로 selectMessageGridList.do API를 호출하고,
최신 규정 선별용 후보 JSON(5필드) 리스트를 수집합니다.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, asdict
from datetime import datetime

from src.config import GW_BASE_URL, GW_BOARD_FOLDER_ID
from src.gw_downloader import GWDownloader

log = logging.getLogger(__name__)

LIST_API_URL = f"{GW_BASE_URL}/groupware/board/selectMessageGridList.do"

VIEW_URL_TEMPLATE = (
    f"{GW_BASE_URL}/groupware/layout/board_BoardView.do"
    "?CLSYS=board&CLMD=user&CLBIZ=Board&menuID=10&version=1"
    "&folderID={folder_id}&messageID={message_id}"
    "&viewType=List&boardType=Normal"
    "&startDate=&endDate=&sortBy=&searchText=&searchType=Subject"
    "&page={page}&pageSize={page_size}&rNum={r_num}"
    "&boxType=Receive&approvalStatus=R"
)


@dataclass
class BoardCandidate:
    """최신 규정 선별용 후보 1건(요청한 5필드)."""
    title: str
    reg_date: str
    source_url: str
    reg_num: int
    reg_user: str

    def to_dict(self) -> dict:
        return asdict(self)


def _to_iso_date(raw: str) -> str:
    """그룹웨어 날짜 문자열을 ISO 날짜(YYYY-MM-DD)로 변환."""
    v = (raw or "").strip()
    if not v:
        return ""

    for fmt in ("%Y.%m.%d %H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y.%m.%d", "%Y-%m-%d"):
        try:
            return datetime.strptime(v, fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    return v[:10]


def fetch_board_list(
    gw: GWDownloader,
    folder_id: str = GW_BOARD_FOLDER_ID,
    page_size: int = 10,
) -> list[BoardCandidate]:
    """게시판 전체 목록을 최신 규정 선별용 5필드 구조로 수집합니다.

    Args:
        gw: 로그인된 GWDownloader 인스턴스
        folder_id: 게시판 폴더 ID (사규 게시판 기본 8233)
        page_size: 한 페이지당 요청 건수

    Returns:
        BoardCandidate 리스트
    """
    gw.ensure_logged_in(
        return_url=(
            f"{GW_BASE_URL}/groupware/layout/board_BoardList.do"
            f"?CLSYS=Board&CLMD=user&boardType=Normal&CLBIZ=Board"
            f"&menuID=10&folderID={folder_id}"
        )
    )

    all_items: list[BoardCandidate] = []
    page_no = 1
    total_count = -1

    while True:
        payload = {
            "pageNo": str(page_no),
            "pageSize": str(page_size),
            "bizSection": "Board",
            "boardType": "Normal",
            "viewType": "List",
            "boxType": "Receive",
            "menuID": "10",
            "folderID": folder_id,
            "folderType": "Board",
            "categoryID": "",
            "searchType": "Subject",
            "searchText": "",
            "useTopNotice": "Y",
            "useUserForm": "",
            "approvalStatus": "R",
            "readSearchType": "",
            "communityID": "",
            "startDate": "",
            "endDate": "",
            "sortBy": "RegistDate desc",
        }

        headers = {
            "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8",
            "X-Requested-With": "XMLHttpRequest",
            "Accept": "application/json, text/javascript, */*; q=0.01",
        }

        resp = gw.session.post(LIST_API_URL, data=payload, headers=headers, verify=False)
        resp.raise_for_status()
        data = resp.json()

        if data.get("status") != "SUCCESS":
            log.warning("응답 상태 오류: %s", data.get("status"))
            break

        if total_count == -1:
            total_count = data.get("page", {}).get("listCount", 0)
            log.info("총 %d건의 게시글 확인", total_count)

        item_list = data.get("list", [])
        if not item_list:
            break

        for item in item_list:
            r_num_raw = item.get("RNUM", "0")
            try:
                r_num = int(float(r_num_raw))
            except ValueError:
                r_num = len(all_items) + 1

            message_id = item.get("MessageID")
            title = item.get("Subject", "").strip()
            reg_date = _to_iso_date(item.get("RegistDate", ""))
            reg_user = (
                item.get("CreatorName")
                or item.get("CreatorDept")
                or item.get("OwnerName")
                or ""
            ).strip()

            if not message_id or not title:
                continue

            url = VIEW_URL_TEMPLATE.format(
                folder_id=folder_id, message_id=message_id,
                page=page_no, page_size=page_size, r_num=r_num,
            )

            try:
                reg_num = int(str(message_id))
            except ValueError:
                reg_num = 0

            all_items.append(
                BoardCandidate(
                    title=title,
                    reg_date=reg_date,
                    source_url=url,
                    reg_num=reg_num,
                    reg_user=reg_user,
                )
            )

        page_count = data.get("page", {}).get("pageCount", 0)
        if page_no >= page_count:
            break

        page_no += 1

    log.info("게시판 목록 수집 완료: %d건", len(all_items))
    return all_items
