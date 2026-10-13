"""Tests for A2A HTTP endpoint shape (health + agent card)."""


def _make_app(agent_name: str, handler=None):
    from ares.a2a_server import make_app
    if handler is None:
        handler = lambda prompt: f"echo: {prompt}"
    return make_app(agent_name, handler)


# ── tests ─────────────────────────────────────────────────────────────────────

def test_health_endpoint():
    from fastapi.testclient import TestClient
    app = _make_app("coder")
    client = TestClient(app)
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["agent"] == "coder"


def test_agent_card_endpoint():
    from fastapi.testclient import TestClient
    app = _make_app("thinker")
    client = TestClient(app)
    resp = client.get("/.well-known/agent.json")
    assert resp.status_code == 200
    card = resp.json()
    assert "name" in card
    assert "thinker" in card["name"].lower() or "thinker" in str(card).lower()


def test_all_agent_names_have_cards():
    from ares.a2a_server import CARDS, PORTS
    assert set(CARDS.keys()) == set(PORTS.keys())
    for name, card in CARDS.items():
        assert card.name  # not empty
        assert card.url


def test_task_endpoint_delegates_to_handler():
    from fastapi.testclient import TestClient
    responses = []

    def handler(prompt):
        responses.append(prompt)
        return "handled"

    app = _make_app("runner", handler)
    client = TestClient(app)
    payload = {
        "params": {
            "message": {"parts": [{"text": "do something"}]}
        }
    }
    resp = client.post("/a2a/tasks", json=payload)
    assert resp.status_code == 200


def _send(server, text="hi"):
    return server.handle_request({"jsonrpc": "2.0", "id": "1", "method": "tasks/send",
                                  "params": {"message": {"parts": [{"text": text}]}}})


def test_task_lifecycle():
    from ares.runtime.a2a import A2AServer, AgentCard
    server = A2AServer(AgentCard(name="t"), handler=lambda p: p.upper())
    sent = _send(server)["result"]
    assert sent["state"] == "completed" and sent["output"] == "HI"
    got = server.handle_request({"id": "2", "method": "tasks/get", "params": {"id": sent["id"]}})
    assert got["result"]["id"] == sent["id"]
    cancelled = server.handle_request({"id": "3", "method": "tasks/cancel", "params": {"id": sent["id"]}})
    assert cancelled["result"]["state"] == "canceled"


def test_task_errors():
    from ares.runtime.a2a import A2AServer, AgentCard

    def boom(prompt):
        raise RuntimeError("model died")

    server = A2AServer(AgentCard(name="t"), handler=boom)
    failed = _send(server)["result"]
    assert failed["state"] == "failed" and "model died" in failed["output"]
    assert server.handle_request({"id": "1", "method": "tasks/get", "params": {"id": "nope"}})["error"]["code"] == -32602
    assert server.handle_request({"id": "1", "method": "nope"})["error"]["code"] == -32601


def test_old_tasks_are_dropped(monkeypatch):
    from ares.runtime import a2a
    monkeypatch.setattr(a2a, "MAX_TASKS", 2)
    server = a2a.A2AServer(a2a.AgentCard(name="t"), handler=str)
    ids = [_send(server)["result"]["id"] for _ in range(3)]
    assert "error" in server.handle_request({"id": "1", "method": "tasks/get", "params": {"id": ids[0]}})
