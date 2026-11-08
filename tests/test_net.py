import httpx
import pytest

from kinetixai import net


def _client(handler):
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_retries_429_then_succeeds():
    calls = []

    def handler(req):
        calls.append(req)
        if len(calls) == 1:
            return httpx.Response(429, headers={"retry-after": "1"})
        return httpx.Response(200, json={"ok": 1})
    with _client(handler) as c:
        assert net.get_json(c, "https://x.test/a", source="X") == {"ok": 1}
    assert len(calls) == 2


def test_rate_limit_with_long_wait_is_reported_not_slept():
    with _client(lambda r: httpx.Response(429, headers={"retry-after": "120"})) as c:
        with pytest.raises(net.RateLimited) as exc:
            net.get_json(c, "https://x.test/a", source="Hugging Face")
    assert "Hugging Face" in str(exc.value) and "rate limit" in str(exc.value)


def test_github_403_quota_is_a_rate_limit():
    h = {"x-ratelimit-remaining": "0", "x-ratelimit-reset": "9999999999"}
    with _client(lambda r: httpx.Response(403, headers=h)) as c:
        with pytest.raises(net.RateLimited):
            net.get_json(c, "https://api.github.com/search/repositories", source="GitHub")


def test_timeout_becomes_source_error():
    def handler(req):
        raise httpx.ReadTimeout("slow", request=req)
    with _client(handler) as c:
        with pytest.raises(net.SourceError) as exc:
            net.get_json(c, "https://x.test", source="arXiv")
    assert "timed out" in str(exc.value)


def test_connect_error_becomes_source_error():
    def handler(req):
        raise httpx.ConnectError("no route", request=req)
    with _client(handler) as c:
        with pytest.raises(net.SourceError) as exc:
            net.get_json(c, "https://x.test", source="OpenAlex")
    assert "unreachable" in str(exc.value)


def test_404_allowed_returns_none_and_500_raises():
    with _client(lambda r: httpx.Response(404)) as c:
        assert net.get_json(c, "https://x.test", source="X", allow_404=True) is None
    with _client(lambda r: httpx.Response(500)) as c:
        with pytest.raises(net.SourceError):
            net.get_json(c, "https://x.test", source="X")


def test_non_json_body_is_a_clear_error():
    with _client(lambda r: httpx.Response(200, text="<html>")) as c:
        with pytest.raises(net.SourceError) as exc:
            net.get_json(c, "https://x.test", source="X")
    assert "not JSON" in str(exc.value)


def test_user_agent_identifies_kx():
    assert net.UA.startswith("kinetixai-cli/")
