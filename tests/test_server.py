from fastapi.testclient import TestClient
from refundops_rl.server import app

client = TestClient(app)


def test_health():
    assert client.get("/health").json() == {"status": "ok"}


def test_episode_lifecycle():
    created = client.post("/v1/episodes", json={"seed": 17, "family": "outside_window"})
    assert created.status_code == 201
    episode_id = created.json()["episode_id"]
    order_id = created.json()["observation"]["task"].split("order ")[1].split(".")[0]
    for tool, arguments in (
        ("get_order", {"order_id": order_id}),
        ("get_payment", {"order_id": order_id}),
        ("get_policy", {}),
    ):
        assert client.post(f"/v1/episodes/{episode_id}/step", json={"tool": tool, "args": arguments}).status_code == 200
    finished = client.post(f"/v1/episodes/{episode_id}/step", json={"tool": "finish", "args": {"resolution": "rejected"}})
    assert finished.json()["reward"] == 1.0
    assert finished.json()["terminated"]
    assert client.post(f"/v1/episodes/{episode_id}/step", json={"tool": "get_policy"}).status_code == 409
    assert client.delete(f"/v1/episodes/{episode_id}").status_code == 204
    assert client.post(f"/v1/episodes/{episode_id}/step", json={"tool": "get_policy"}).status_code == 404


def test_invalid_request_and_missing_session():
    assert client.post("/v1/episodes", json={"seed": 1, "family": "does_not_exist"}).status_code == 422
    assert client.post("/v1/episodes/fake/step", json={"tool": "get_policy"}).status_code == 404
    assert client.post("/v1/episodes", json={"seed": -1}).status_code == 422
