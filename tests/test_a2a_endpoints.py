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
