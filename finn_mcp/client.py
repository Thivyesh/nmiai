"""HTTP client for finn.no's public search backend.

Why this exists
---------------
The recurring problem with "finn.no via an LLM" is that generic web search
returns *stale* results — Google/DuckDuckGo cache pages for listings that are
already sold or deactivated. This client instead queries finn.no's **own**
search index (``/api/search-qf``), which only contains ads that are currently
active. Every result you get back is live on finn right now.

The endpoint is the same JSON API that powers finn.no's own frontend:

    GET https://www.finn.no/api/search-qf
        ?searchkey=SEARCH_ID_BAP_COMMON
        &search_type=SEARCH_ID_BAP_ALL
        &vertical=bap
        &q=<query>
        &sort=<sort>
        &page=<n>

``bap`` ("bruktmarked") is Torget — the general second-hand marketplace, i.e.
"things I'm interested in". Other verticals (cars, real estate, jobs) use
different ``searchkey``/``vertical`` values; :meth:`FinnClient.raw_search`
exposes those for power users without hardcoding constants we can't verify.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any

import httpx

SEARCH_ENDPOINT = "https://www.finn.no/api/search-qf"

# Torget / general second-hand marketplace ("bruktmarked").
TORGET_PARAMS = {
    "searchkey": "SEARCH_ID_BAP_COMMON",
    "search_type": "SEARCH_ID_BAP_ALL",
    "vertical": "bap",
}

# User-friendly sort aliases -> finn.no sort tokens.
SORT_ALIASES = {
    "relevance": "RELEVANCE",
    "newest": "PUBLISHED_DESC",
    "oldest": "PUBLISHED_ASC",
    "price_asc": "PRICE_ASC",
    "price_desc": "PRICE_DESC",
    "cheapest": "PRICE_ASC",
    "expensive": "PRICE_DESC",
}

# A realistic browser UA — finn.no serves the JSON API to normal clients but
# a missing/obviously-bot UA is more likely to be throttled.
DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "nb-NO,nb;q=0.9,en;q=0.8",
}

_IMG_BASE = "https://images.finncdn.no/dynamic/480x360c/"


@dataclass
class Listing:
    """A single active finn.no listing, normalized to a stable shape."""

    id: str
    title: str
    url: str
    price: int | None = None
    price_text: str | None = None
    currency: str = "NOK"
    location: str | None = None
    published: str | None = None  # ISO-8601 UTC
    image: str | None = None
    trade_type: str | None = None
    flags: list[str] = field(default_factory=list)
    is_sold: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class FinnError(RuntimeError):
    """Raised when finn.no cannot be reached or returns an unexpected shape."""


class FinnClient:
    """Async client for finn.no search.

    Parameters
    ----------
    timeout:
        Per-request timeout in seconds.
    retries:
        Number of retry attempts on transient network/5xx errors.
    """

    def __init__(self, timeout: float = 15.0, retries: int = 3) -> None:
        self._timeout = timeout
        self._retries = retries

    # -- public API -----------------------------------------------------

    async def search(
        self,
        query: str,
        *,
        sort: str = "relevance",
        max_results: int = 20,
        min_price: int | None = None,
        max_price: int | None = None,
        published_since_days: int | None = None,
        extra_params: dict[str, Any] | None = None,
    ) -> list[Listing]:
        """Search the Torget marketplace for **active** listings.

        Results are paginated internally until ``max_results`` is reached or
        finn runs out of matches.
        """
        params: dict[str, Any] = dict(TORGET_PARAMS)
        params["q"] = query
        params["sort"] = SORT_ALIASES.get(sort.lower(), sort)
        if min_price is not None:
            params["price_from"] = int(min_price)
        if max_price is not None:
            params["price_to"] = int(max_price)
        if extra_params:
            params.update(extra_params)

        listings = await self._paged_search(params, max_results)

        if published_since_days is not None:
            cutoff = datetime.now(timezone.utc).timestamp() - published_since_days * 86400
            listings = [
                l
                for l in listings
                if l.published
                and datetime.fromisoformat(l.published).timestamp() >= cutoff
            ]
        return listings

    async def raw_search(
        self, params: dict[str, Any], *, max_results: int = 20
    ) -> list[Listing]:
        """Escape hatch: hit ``search-qf`` with arbitrary parameters.

        Lets callers target other verticals (cars, real estate, jobs) or use
        filters not modeled by :meth:`search`, e.g.::

            {"searchkey": "SEARCH_ID_CAR_USED", "vertical": "car",
             "q": "tesla", "year_from": 2020}

        Only active listings are ever returned, since finn's index only holds
        live ads.
        """
        return await self._paged_search(dict(params), max_results)

    async def get_listing(self, id_or_url: str) -> dict[str, Any]:
        """Fetch one ad and report whether it is still active.

        ``id_or_url`` may be a finnkode (``"123456789"``) or a full finn URL.
        Returns metadata parsed from the ad's Open Graph tags plus an
        ``is_active`` flag — the reliable answer to "is this still for sale?".
        """
        finnkode = _extract_finnkode(id_or_url)
        if not finnkode:
            raise FinnError(f"Could not parse a finnkode from {id_or_url!r}")

        url = f"https://www.finn.no/recommerce/forsale/item/{finnkode}"
        async with self._client() as client:
            resp = await self._request(client, "GET", url, headers=DEFAULT_HEADERS)

        if resp.status_code == 404:
            return {"id": finnkode, "url": url, "is_active": False,
                    "reason": "Not found (404) — removed or never existed."}

        html = resp.text
        info = _parse_og_tags(html)
        inactive_markers = ("Annonsen finnes ikke", "er ikke lenger tilgjengelig",
                            "har blitt deaktivert")
        sold = _looks_sold(html)
        is_active = resp.status_code == 200 and not any(
            m.lower() in html.lower() for m in inactive_markers
        )
        return {
            "id": finnkode,
            "url": info.get("url") or url,
            "title": info.get("title"),
            "description": info.get("description"),
            "image": info.get("image"),
            "price": info.get("price"),
            "is_active": is_active,
            "is_sold": sold,
        }

    # -- internals ------------------------------------------------------

    def _client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(
            timeout=self._timeout,
            follow_redirects=True,
            headers=DEFAULT_HEADERS,
        )

    async def _paged_search(
        self, params: dict[str, Any], max_results: int
    ) -> list[Listing]:
        out: list[Listing] = []
        page = int(params.get("page", 1) or 1)
        async with self._client() as client:
            while len(out) < max_results:
                params["page"] = page
                data = await self._get_json(client, params)
                docs = data.get("docs") or []
                if not docs:
                    break
                out.extend(_normalize(d) for d in docs)
                last = _paging_last(data)
                if page >= last:
                    break
                page += 1
        return out[:max_results]

    async def _get_json(
        self, client: httpx.AsyncClient, params: dict[str, Any]
    ) -> dict[str, Any]:
        resp = await self._request(client, "GET", SEARCH_ENDPOINT, params=params)
        if resp.status_code != 200:
            raise FinnError(
                f"finn.no returned HTTP {resp.status_code} for {resp.url}"
            )
        try:
            return resp.json()
        except ValueError as exc:  # pragma: no cover - defensive
            raise FinnError("finn.no returned a non-JSON response") from exc

    async def _request(
        self, client: httpx.AsyncClient, method: str, url: str, **kwargs: Any
    ) -> httpx.Response:
        last_exc: Exception | None = None
        for attempt in range(self._retries):
            try:
                resp = await client.request(method, url, **kwargs)
                # Retry transient server errors, surface everything else.
                if resp.status_code in (429, 500, 502, 503, 504):
                    last_exc = FinnError(f"HTTP {resp.status_code}")
                else:
                    return resp
            except (httpx.TransportError, httpx.TimeoutException) as exc:
                last_exc = exc
            await asyncio.sleep(2**attempt)
        raise FinnError(f"finn.no request failed after {self._retries} attempts: {last_exc}")


# -- normalization helpers ----------------------------------------------


def _normalize(doc: dict[str, Any]) -> Listing:
    finn_id = str(
        doc.get("ad_id") or doc.get("id") or doc.get("finnkode") or ""
    )
    price, price_text, currency = _extract_price(doc)
    flags = [str(f) for f in (doc.get("flags") or [])]
    labels = [
        str(lbl.get("text") if isinstance(lbl, dict) else lbl)
        for lbl in (doc.get("labels") or [])
    ]
    all_flags = flags + labels
    return Listing(
        id=finn_id,
        title=doc.get("heading") or doc.get("title") or "(no title)",
        url=_extract_url(doc, finn_id),
        price=price,
        price_text=price_text,
        currency=currency,
        location=doc.get("location"),
        published=_extract_timestamp(doc),
        image=_extract_image(doc),
        trade_type=doc.get("trade_type"),
        flags=all_flags,
        is_sold=_flag_is_sold(all_flags),
    )


def _extract_price(doc: dict[str, Any]) -> tuple[int | None, str | None, str]:
    price = doc.get("price")
    currency = "NOK"
    if isinstance(price, dict):
        amount = price.get("amount")
        currency = price.get("currency_code") or currency
        text = price.get("display") or (
            f"{amount:,} {currency}".replace(",", " ") if amount is not None else None
        )
        return (int(amount) if isinstance(amount, (int, float)) else None, text, currency)
    if isinstance(price, (int, float)):
        return int(price), f"{int(price):,} NOK".replace(",", " "), currency
    # Some verticals nest prices differently (e.g. price_range / total_price).
    for key in ("total_price", "price_total"):
        nested = doc.get(key)
        if isinstance(nested, dict) and isinstance(nested.get("amount"), (int, float)):
            amt = int(nested["amount"])
            return amt, f"{amt:,} NOK".replace(",", " "), currency
    return None, None, currency


def _extract_url(doc: dict[str, Any], finn_id: str) -> str:
    canonical = doc.get("canonical_url") or doc.get("ad_link")
    if isinstance(canonical, str) and canonical.startswith("http"):
        return canonical
    if isinstance(canonical, str) and canonical.startswith("/"):
        return "https://www.finn.no" + canonical
    if finn_id:
        return f"https://www.finn.no/recommerce/forsale/item/{finn_id}"
    return "https://www.finn.no"


def _extract_image(doc: dict[str, Any]) -> str | None:
    image = doc.get("image")
    if isinstance(image, dict):
        url = image.get("url")
        if isinstance(url, str) and url.startswith("http"):
            return url
        path = image.get("path")
        if isinstance(path, str) and path:
            return _IMG_BASE + path.lstrip("/")
    if isinstance(image, str) and image.startswith("http"):
        return image
    return None


def _extract_timestamp(doc: dict[str, Any]) -> str | None:
    ts = doc.get("timestamp") or doc.get("published")
    if isinstance(ts, (int, float)):
        # finn uses millisecond epochs.
        seconds = ts / 1000 if ts > 1e11 else ts
        return datetime.fromtimestamp(seconds, tz=timezone.utc).isoformat()
    if isinstance(ts, str) and ts:
        return ts
    return None


def _flag_is_sold(flags: list[str]) -> bool:
    lowered = " ".join(flags).lower()
    return "sold" in lowered or "solgt" in lowered


def _paging_last(data: dict[str, Any]) -> int:
    meta = data.get("metadata") or {}
    paging = meta.get("paging") or {}
    last = paging.get("last")
    if isinstance(last, int) and last > 0:
        return last
    return 1


# -- ad-page parsing helpers --------------------------------------------

import re  # noqa: E402  (kept local to the parsing helpers below)

_OG_RE = re.compile(
    r'<meta[^>]+(?:property|name)=["\'](og:[\w:]+|product:price:amount)["\']'
    r'[^>]+content=["\']([^"\']*)["\']',
    re.IGNORECASE,
)


def _parse_og_tags(html: str) -> dict[str, Any]:
    tags: dict[str, str] = {}
    for prop, content in _OG_RE.findall(html):
        tags.setdefault(prop.lower(), content)
    price_raw = tags.get("product:price:amount")
    price: int | None = None
    if price_raw:
        try:
            price = int(float(price_raw))
        except ValueError:
            price = None
    return {
        "title": tags.get("og:title"),
        "description": tags.get("og:description"),
        "image": tags.get("og:image"),
        "url": tags.get("og:url"),
        "price": price,
    }


def _looks_sold(html: str) -> bool:
    low = html.lower()
    return ">solgt<" in low or '"sold"' in low


_FINNKODE_RE = re.compile(r"(?:finnkode=|/item/|/ad\.html\?finnkode=|/)?(\d{6,})")


def _extract_finnkode(id_or_url: str) -> str | None:
    id_or_url = id_or_url.strip()
    if id_or_url.isdigit():
        return id_or_url
    # Prefer an explicit finnkode/item id, else the last long number in the URL.
    matches = re.findall(r"(\d{6,})", id_or_url)
    return matches[-1] if matches else None
