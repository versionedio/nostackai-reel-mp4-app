"""The Reeldesk client: sign-in steps, headers, and error messages that are safe for public logs."""
import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from reeldesk import Connection, Reeldesk, ReeldeskError, check_id  # noqa: E402

CONN = Connection(base_url="https://app.example.test/", org_code="org-1", api_key="KEY", client_id="CID")


def response(status=200, headers=None, json_body=None):
    res = mock.Mock(status_code=status, headers=headers or {}, content=b"{}" if json_body is not None else b"")
    res.json.return_value = json_body or {}
    return res


class SignIn(unittest.TestCase):
    def test_two_steps_then_the_token_authorizes_calls(self):
        http = mock.Mock()
        http.post.return_value = response(headers={"oauthtoken": "OAUTH"})
        http.get.return_value = response(headers={"x-nostackai-token": "TOKEN"})
        rd = Reeldesk(CONN, http)
        rd.sign_in("render@example.test", "pw")
        login_url = http.post.call_args.args[0]
        self.assertEqual(login_url, "https://app.example.test/app/org-1/user/login")
        self.assertEqual(http.post.call_args.kwargs["json"], {"email": "render@example.test", "pwd": "pw"})
        self.assertEqual(http.get.call_args.kwargs["headers"], {"oauthtoken": "OAUTH", "x-api-key": "KEY", "client_id": "CID"})

        http.request.return_value = response(json_body={"data": []})
        rd.list_files("v1")
        self.assertEqual(http.request.call_args.kwargs["headers"]["authorization"], "TOKEN")

    def test_refused_sign_in_says_so_without_the_password(self):
        http = mock.Mock()
        http.post.return_value = response(status=401)
        with self.assertRaises(ReeldeskError) as err:
            Reeldesk(CONN, http).sign_in("render@example.test", "secret-pw")
        self.assertNotIn("secret-pw", str(err.exception))


class Errors(unittest.TestCase):
    def signed_in(self, status):
        http = mock.Mock()
        http.request.return_value = response(status=status)
        rd = Reeldesk(CONN, http)
        rd.token = "TOKEN"
        return rd

    def test_forbidden_names_the_route_not_the_record(self):
        with self.assertRaises(ReeldeskError) as err:
            self.signed_in(403).update_video("cvimunlmh4p8", {"render_status": "done"})
        self.assertIn("PUT /update/videos", str(err.exception))
        self.assertNotIn("cvimunlmh4p8", str(err.exception))

    def test_ids_are_checked_before_they_reach_a_url(self):
        self.assertEqual(check_id("c-71de"), "c-71de")
        for bad in ["", "../etc", "a/b", "x" * 65, "id?x=1"]:
            with self.assertRaises(ReeldeskError):
                check_id(bad)

    def test_a_failed_content_delete_never_fails_the_run(self):
        self.signed_in(500).delete_content("c-71de")


if __name__ == "__main__":
    unittest.main()
