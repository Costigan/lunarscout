#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import deque
from html.parser import HTMLParser
import sys
from urllib.parse import urldefrag, urljoin, urlparse
from urllib.request import Request, urlopen


class LinkParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.links: list[str] = []

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag.lower() != "a":
            return
        for name, value in attrs:
            if name.lower() == "href" and value:
                self.links.append(value)


def _without_fragment(url: str) -> str:
    return urldefrag(url)[0]


def _is_directory_like_path(path: str) -> bool:
    if not path or path.endswith("/"):
        return True
    name = path.rsplit("/", 1)[-1]
    return "." not in name


def _directory_url(url: str) -> str:
    parsed = urlparse(url)
    if not _is_directory_like_path(parsed.path) or parsed.path.endswith("/"):
        return url
    return parsed._replace(path=parsed.path + "/").geturl()


def _normalized_url(base_url: str, href: str) -> str | None:
    url = _without_fragment(urljoin(_directory_url(base_url), href.strip()))
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        return None
    return url


def _looks_like_html_url(url: str) -> bool:
    path = urlparse(url).path
    if path.endswith("/"):
        return True
    name = path.rsplit("/", 1)[-1]
    if "." not in name:
        return True
    return name.lower().endswith((".html", ".htm"))


def _scope_base_path(path: str) -> str:
    if not path or path.endswith("/"):
        return path
    if _is_directory_like_path(path):
        return path + "/"
    return path.rsplit("/", 1)[0] + "/"


def _in_scope(url: str, root_url: str, scope: str) -> bool:
    if scope == "any":
        return True

    parsed = urlparse(url)
    root = urlparse(root_url)
    if parsed.scheme != root.scheme or parsed.netloc != root.netloc:
        return False
    if scope == "host":
        return True

    return parsed.path.startswith(_scope_base_path(root.path))


def _fetch_html(url: str, timeout: float) -> tuple[str, str] | None:
    request = Request(
        url,
        headers={
            "User-Agent": "lunarscout-get-link-tree/0.1",
            "Accept": "text/html,application/xhtml+xml",
        },
    )
    with urlopen(request, timeout=timeout) as response:
        content_type = response.headers.get_content_type()
        if content_type not in {"text/html", "application/xhtml+xml"}:
            return None
        charset = response.headers.get_content_charset() or "utf-8"
        final_url = _without_fragment(response.geturl())
        return response.read().decode(charset, errors="replace"), final_url


def crawl_links(
    start_url: str,
    *,
    max_depth: int,
    scope: str,
    timeout: float,
) -> list[str]:
    start_url = _directory_url(_without_fragment(start_url))
    visited_pages: set[str] = set()
    discovered_urls: set[str] = {start_url}
    queue = deque([(start_url, 0)])

    while queue:
        page_url, depth = queue.popleft()
        if page_url in visited_pages:
            continue
        visited_pages.add(page_url)

        try:
            result = _fetch_html(page_url, timeout)
        except Exception as exc:
            print(f"warning: could not fetch {page_url}: {exc}", file=sys.stderr)
            continue
        if result is None:
            continue
        html, final_page_url = result
        final_page_url = _directory_url(final_page_url)
        discovered_urls.add(final_page_url)
        visited_pages.add(final_page_url)

        parser = LinkParser()
        parser.feed(html)
        for href in parser.links:
            url = _normalized_url(final_page_url, href)
            if url is None:
                continue
            url = _directory_url(url) if _looks_like_html_url(url) else url
            discovered_urls.add(url)
            if depth >= max_depth:
                continue
            if url in visited_pages:
                continue
            if not _looks_like_html_url(url):
                continue
            if not _in_scope(url, start_url, scope):
                continue
            queue.append((url, depth + 1))

    return sorted(discovered_urls)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Print links discovered from an HTML page tree. Only HTML pages are "
            "fetched recursively; links to other file types are printed but not "
            "followed."
        ),
    )
    parser.add_argument("url", help="starting URL")
    parser.add_argument(
        "--max-depth",
        type=int,
        default=10,
        help="maximum link depth to follow from the starting page (default: 10)",
    )
    parser.add_argument(
        "--scope",
        choices=("path", "host", "any"),
        default="path",
        help=(
            "crawl scope: path stays under the starting URL directory, host "
            "stays on the same host, any follows matching HTML links anywhere "
            "(default: path)"
        ),
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=20.0,
        help="HTTP timeout in seconds per page fetch (default: 20)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.max_depth < 0:
        parser.error("--max-depth must be non-negative")

    for url in crawl_links(
        args.url,
        max_depth=args.max_depth,
        scope=args.scope,
        timeout=args.timeout,
    ):
        print(url)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
