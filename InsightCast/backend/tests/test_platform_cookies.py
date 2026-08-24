from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


BACKEND_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(BACKEND_ROOT / "src"))
sys.path.insert(0, str(WORKSPACE_ROOT / "myHarness" / "myHarness-V2" / "src"))

from insightcast.platform_cookies import (
    build_ytdlp_cookie_options,
    cookie_header_from_file,
    parse_browser_cookie_spec,
)


class PlatformCookieTests(unittest.TestCase):
    def test_cookie_header_from_netscape_file_filters_domain(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cookie_path = Path(tmp) / "cookies.txt"
            cookie_path.write_text(
                "\n".join(
                    [
                        "# Netscape HTTP Cookie File",
                        ".bilibili.com\tTRUE\t/\tTRUE\t0\tSESSDATA\tabc",
                        "#HttpOnly_.bilibili.com\tTRUE\t/\tTRUE\t0\tbili_jct\tdef",
                        ".youtube.com\tTRUE\t/\tTRUE\t0\tSID\tignored",
                    ]
                ),
                encoding="utf-8",
            )

            header = cookie_header_from_file(
                cookie_path,
                domains=("bilibili.com",),
            )

        self.assertEqual(header, "SESSDATA=abc; bili_jct=def")

    def test_build_ytdlp_cookie_options_uses_file_browser_and_raw_cookie(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cookie_path = Path(tmp) / "cookies.txt"
            cookie_path.write_text("SESSDATA=abc", encoding="utf-8")
            options = build_ytdlp_cookie_options(
                {
                    "INSIGHTCAST_YTDLP_COOKIES_FILE": str(cookie_path),
                    "INSIGHTCAST_YTDLP_COOKIES_FROM_BROWSER": "edge:Default",
                    "INSIGHTCAST_YTDLP_COOKIE": "SESSDATA=raw",
                }
            )

        self.assertEqual(options["cookiefile"], str(cookie_path))
        self.assertEqual(options["cookiesfrombrowser"], ("edge", "Default", None, None))
        self.assertEqual(options["http_headers"], {"Cookie": "SESSDATA=raw"})

    def test_build_ytdlp_cookie_options_reuses_raw_bilibili_cookie_file_as_header(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cookie_path = Path(tmp) / "bilibili-cookie.txt"
            cookie_path.write_text("SESSDATA=abc; bili_jct=def", encoding="utf-8")
            options = build_ytdlp_cookie_options(
                {"INSIGHTCAST_BILIBILI_COOKIE_FILE": str(cookie_path)}
            )

        self.assertEqual(
            options["http_headers"],
            {"Cookie": "SESSDATA=abc; bili_jct=def"},
        )
        self.assertNotIn("cookiefile", options)

    def test_parse_browser_cookie_spec_supports_browser_only(self) -> None:
        self.assertEqual(parse_browser_cookie_spec("chrome"), ("chrome", None, None, None))


if __name__ == "__main__":
    unittest.main()
