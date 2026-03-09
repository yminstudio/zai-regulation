"""KG그룹 그룹웨어 게시판 첨부파일 다운로더.

AES-128-CBC 암호화 로그인 후 세션을 유지하며,
게시글 상세 API에서 첨부파일 목록을 조회하고 다운로드합니다.

사용법:
    from src.gw_downloader import GWDownloader
    gw = GWDownloader()
    files = gw.download_attachments(board_url)
"""
from __future__ import annotations

import re
import hashlib
import base64
import urllib.parse
import warnings
import logging
from pathlib import Path
from dataclasses import dataclass

import requests
from requests.packages.urllib3.exceptions import InsecureRequestWarning
from Crypto.Cipher import AES
from Crypto.Util.Padding import pad

from src.config import GW_USER_ID, GW_PASSWORD, GW_BASE_URL, DATA_DIR

warnings.filterwarnings("ignore", category=InsecureRequestWarning)
log = logging.getLogger(__name__)

LOGIN_PAGE_URL = f"{GW_BASE_URL}/covicore/login.do"
LOGIN_BASE_CHK_URL = f"{GW_BASE_URL}/covicore/loginbasechk.do"
LOGIN_CHK_URL = f"{GW_BASE_URL}/covicore/loginchk.do"
FILE_DOWN_URL = f"{GW_BASE_URL}/covicore/common/fileDown.do"
MESSAGE_DETAIL_URL = f"{GW_BASE_URL}/groupware/board/selectMessageDetail.do"

AES_SALT_HEX = "18b00b2fc5f0e0ee40447bba4dabc952"
AES_IV_HEX = "4378110db6392f93e95d5159dabdee9b"
AES_PASSPHRASE = "1234"
AES_KEY_SIZE = 16
AES_ITERATIONS = 1000

BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/131.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7",
    "Accept-Encoding": "gzip, deflate, br",
    "Origin": GW_BASE_URL,
    "Referer": f"{GW_BASE_URL}/covicore/login.do",
}


def _encrypt_password(plain_password: str) -> str:
    """CryptoJS AesUtil.encrypt 호환 AES-128-CBC 암호화."""
    salt = bytes.fromhex(AES_SALT_HEX)
    iv = bytes.fromhex(AES_IV_HEX)
    key = hashlib.pbkdf2_hmac(
        "sha1", AES_PASSPHRASE.encode(), salt, AES_ITERATIONS, dklen=AES_KEY_SIZE,
    )
    cipher = AES.new(key, AES.MODE_CBC, iv)
    ct = cipher.encrypt(pad(plain_password.encode(), AES.block_size))
    return base64.b64encode(ct).decode()


@dataclass
class AttachmentInfo:
    file_id: str
    file_token: str
    file_name: str
    download_url: str


class GWDownloader:
    """그룹웨어 세션 기반 첨부파일 다운로더."""

    def __init__(
        self,
        user_id: str | None = None,
        password: str | None = None,
        download_dir: str | Path | None = None,
    ):
        self.session = requests.Session()
        self.session.headers.update(BROWSER_HEADERS)
        self.download_dir = Path(download_dir) if download_dir else DATA_DIR
        self.download_dir.mkdir(parents=True, exist_ok=True)
        self.user_id = user_id or GW_USER_ID
        self.password = password or GW_PASSWORD
        self._logged_in = False

    # ------------------------------------------------------------------
    # 로그인
    # ------------------------------------------------------------------
    def login(self, return_url: str = "") -> None:
        """2단계 로그인: loginbasechk.do -> loginchk.do."""
        self.session.get(LOGIN_PAGE_URL, verify=False)

        enc_pw = _encrypt_password(self.password)
        chk_resp = self.session.post(
            LOGIN_BASE_CHK_URL,
            data={"id": self.user_id, "pw": enc_pw, "language": "ko"},
            headers={
                **BROWSER_HEADERS,
                "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8",
                "X-Requested-With": "XMLHttpRequest",
            },
            verify=False,
        )
        chk_resp.raise_for_status()

        result = ""
        try:
            result = chk_resp.json().get("result", "")
        except Exception:
            pass

        if result != "ok":
            raise RuntimeError(
                f"로그인 실패 (loginbasechk 결과: {result}). 아이디/비밀번호를 확인하세요."
            )

        p_return = ""
        if return_url and return_url.startswith(GW_BASE_URL):
            p_return = return_url[len(GW_BASE_URL):]

        form_data = {
            "isFIDO": "N", "RelayState": "", "SamlRequest": "",
            "uid": "", "acr": "", "destination": "",
            "pReturnURL": p_return,
            "idtemp": self.user_id, "id": self.user_id,
            "passwordtemp": "", "password": "", "language": "ko",
        }
        resp = self.session.post(LOGIN_CHK_URL, data=form_data, verify=False)
        resp.raise_for_status()
        self._logged_in = True
        log.info("로그인 성공 (user: %s)", self.user_id)

    def ensure_logged_in(self, return_url: str = "") -> None:
        if not self._logged_in:
            self.login(return_url=return_url)

    # ------------------------------------------------------------------
    # 첨부파일 목록 조회
    # ------------------------------------------------------------------
    @staticmethod
    def parse_url_params(board_url: str) -> dict:
        parsed = urllib.parse.urlparse(board_url)
        return dict(urllib.parse.parse_qsl(parsed.query))

    def fetch_file_list(self, board_url: str) -> list[dict]:
        """selectMessageDetail.do API로 첨부파일 목록 반환."""
        params = self.parse_url_params(board_url)
        message_id = params.get("messageID", "")
        folder_id = params.get("folderID", "")
        version = params.get("version", "1")

        if not message_id or not folder_id:
            raise ValueError(f"URL에서 messageID/folderID를 찾을 수 없습니다: {board_url}")

        self.session.get(board_url, verify=False)

        resp = self.session.post(
            MESSAGE_DETAIL_URL,
            data={
                "bizSection": "Board", "version": version,
                "messageID": message_id, "folderID": folder_id,
                "readFlagStr": "true",
            },
            headers={
                **BROWSER_HEADERS,
                "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8",
                "X-Requested-With": "XMLHttpRequest",
                "Referer": board_url,
            },
            verify=False,
        )
        resp.raise_for_status()
        data = resp.json()

        if data.get("status") != "SUCCESS":
            raise RuntimeError(f"게시글 조회 실패: {data}")

        return data.get("fileList", [])

    # ------------------------------------------------------------------
    # 첨부파일 다운로드
    # ------------------------------------------------------------------
    def get_attachments(self, board_url: str) -> list[AttachmentInfo]:
        self.ensure_logged_in(return_url=board_url)

        file_list = self.fetch_file_list(board_url)
        attachments: list[AttachmentInfo] = []

        for item in file_list:
            file_id = str(item.get("FileID", ""))
            file_token = item.get("FileToken", "")
            file_name = item.get("FileName", f"file_{file_id}")
            if not file_id or not file_token:
                continue

            encoded_token = urllib.parse.quote(file_token, safe="")
            download_url = f"{FILE_DOWN_URL}?fileID={file_id}&fileToken={encoded_token}"
            attachments.append(AttachmentInfo(
                file_id=file_id, file_token=file_token,
                file_name=file_name, download_url=download_url,
            ))

        return attachments

    def download_file(self, attachment: AttachmentInfo) -> Path:
        resp = self.session.get(attachment.download_url, verify=False, stream=True)
        resp.raise_for_status()

        cd = resp.headers.get("Content-Disposition", "")
        filename = _parse_content_disposition(cd) or attachment.file_name
        if "%" in filename:
            filename = urllib.parse.unquote(filename)

        file_path = self.download_dir / filename
        with open(file_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)

        size_kb = file_path.stat().st_size / 1024
        log.info("다운로드 완료: %s (%.1f KB)", filename, size_kb)
        return file_path

    def download_attachments(self, board_url: str) -> list[Path]:
        """게시판 URL의 모든 첨부파일을 다운로드하고 파일 경로 리스트 반환."""
        attachments = self.get_attachments(board_url)
        if not attachments:
            log.warning("첨부파일 없음: %s", board_url[:80])
            return []

        log.info("첨부파일 %d개 발견", len(attachments))
        downloaded: list[Path] = []
        for att in attachments:
            try:
                path = self.download_file(att)
                downloaded.append(path)
            except Exception as e:
                log.error("다운로드 실패 (%s): %s", att.file_name, e)

        return downloaded


def _parse_content_disposition(header: str) -> str | None:
    """Content-Disposition 헤더에서 파일명 추출."""
    if not header:
        return None

    match = re.search(r"filename\*=(?:UTF-8|utf-8)''(.+?)(?:;|$)", header)
    if match:
        return urllib.parse.unquote(match.group(1).strip())

    match = re.search(r'filename="?([^";]+)"?', header)
    if match:
        raw = match.group(1).strip()
        try:
            return raw.encode("latin-1").decode("utf-8")
        except (UnicodeDecodeError, UnicodeEncodeError):
            return raw

    return None
