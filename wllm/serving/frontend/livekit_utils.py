"""Shared helpers for the LiveKit-based app frontends.

Frontends that stream over LiveKit (LiveAvatar, Krea-Realtime + SAM3,
LongLive) need three credentials: LIVEKIT_URL, LIVEKIT_API_KEY, and
LIVEKIT_API_SECRET. Store them once in a `.env` file at the repo root
(copy `.env.example`); `load_livekit_credentials` reads that file so
every frontend picks them up without per-shell exports.

`viewer_join_url` builds a meet.livekit.io link with a freshly minted
access token, so a frontend can print one URL the user opens directly
instead of pasting the server URL and token into the site by hand.
"""

from __future__ import annotations

import os
from typing import NamedTuple
from urllib.parse import quote


class LiveKitCredentials(NamedTuple):
    url: str
    api_key: str
    api_secret: str


def _repo_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def load_livekit_credentials() -> LiveKitCredentials:
    """Return LiveKit credentials from the environment, loading `.env` first.

    Looks for `.env` in the current directory and then at the repo root.
    Values already exported in the shell take precedence over the file.
    """
    try:
        from dotenv import load_dotenv

        load_dotenv()
        load_dotenv(os.path.join(_repo_root(), ".env"))
    except ModuleNotFoundError:
        pass

    url = os.getenv("LIVEKIT_URL")
    api_key = os.getenv("LIVEKIT_API_KEY")
    api_secret = os.getenv("LIVEKIT_API_SECRET")
    if not url or not api_key or not api_secret:
        raise RuntimeError(
            "LIVEKIT_URL, LIVEKIT_API_KEY, and LIVEKIT_API_SECRET are not all set. "
            "Copy .env.example to .env at the repo root and fill them in "
            "(see docs/frontends.md)."
        )
    return LiveKitCredentials(url, api_key, api_secret)


def make_access_token(
    creds: LiveKitCredentials,
    room: str,
    identity: str,
    can_publish: bool = True,
    can_subscribe: bool = True,
    agent: bool = False,
) -> str:
    from livekit import api

    return (
        api.AccessToken(creds.api_key, creds.api_secret)
        .with_identity(identity)
        .with_name(identity)
        .with_grants(
            api.VideoGrants(
                room_join=True,
                room=room,
                can_publish=can_publish,
                can_subscribe=can_subscribe,
                agent=agent,
            )
        )
        .to_jwt()
    )


def viewer_join_url(creds: LiveKitCredentials, room: str, identity: str = "viewer") -> str:
    """A meet.livekit.io URL that joins `room` directly, token included."""
    token = make_access_token(creds, room, identity)
    return (
        "https://meet.livekit.io/custom"
        f"?liveKitUrl={quote(creds.url, safe='')}&token={quote(token, safe='')}"
    )


def print_join_instructions(creds: LiveKitCredentials, room: str, identity: str = "viewer") -> None:
    url = viewer_join_url(creds, room, identity)
    print(f"[frontend] room ready: {room}")
    print(f"[frontend] open this URL in your browser to join:\n  {url}", flush=True)
