from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict
from wsgiref.simple_server import make_server

from .agents import BaseAgent, build_agent
from .game import ConnectFourGame, PLAYER_ONE, PLAYER_TWO


BASE_DIR = Path(__file__).resolve().parent.parent
STATIC_DIR = BASE_DIR / "static"

MODES = [
    {
        "id": "human_vs_human",
        "label": "Human vs Human",
        "description": "Local two-player mode on the same board.",
    },
    {
        "id": "human_vs_genetic_algorithm",
        "label": "Human vs Genetic Algorithm Agent",
        "description": "Faces a weighted search agent with evolved-style scoring.",
    },
    {
        "id": "human_vs_self_play",
        "label": "Human vs Self-Play Agent",
        "description": "Faces the strongest agent, powered by minimax-style lookahead.",
    },
    {
        "id": "human_vs_semi_random_rl",
        "label": "Human vs Semi-Random RL Agent",
        "description": "Faces the persisted best RL checkpoint.",
    },
]


@dataclass
class GameSession:
    mode: str
    game: ConnectFourGame
    agent: BaseAgent | None = None


SESSIONS: Dict[str, GameSession] = {}


def json_response(status: str, data: Dict[str, Any]) -> tuple[str, list[tuple[str, str]], bytes]:
    payload = json.dumps(data).encode("utf-8")
    headers = [
        ("Content-Type", "application/json; charset=utf-8"),
        ("Content-Length", str(len(payload))),
        ("Access-Control-Allow-Origin", "*"),
    ]
    return status, headers, payload


def file_response(path: Path, content_type: str) -> tuple[str, list[tuple[str, str]], bytes]:
    data = path.read_bytes()
    return "200 OK", [("Content-Type", content_type), ("Content-Length", str(len(data)))], data


def bad_request(message: str) -> tuple[str, list[tuple[str, str]], bytes]:
    return json_response("400 Bad Request", {"error": message})


def not_found() -> tuple[str, list[tuple[str, str]], bytes]:
    return json_response("404 Not Found", {"error": "Not found"})


def get_request_json(environ: dict) -> dict:
    length = int(environ.get("CONTENT_LENGTH") or 0)
    body = environ["wsgi.input"].read(length) if length else b""
    if not body:
        return {}
    return json.loads(body.decode("utf-8"))


def session_payload(session_id: str, session: GameSession) -> dict:
    mode = next(item for item in MODES if item["id"] == session.mode)
    payload = session.game.to_dict(session.mode)
    payload["session_id"] = session_id
    payload["active_mode_id"] = mode["id"]
    payload["active_mode_label"] = mode["label"]
    payload["players"] = {
        "1": "human",
        "2": "human" if session.mode == "human_vs_human" else "agent",
    }
    return payload


def maybe_make_agent_move(session: GameSession) -> None:
    if session.mode == "human_vs_human" or session.game.is_over():
        return
    if session.game.current_player != PLAYER_TWO:
        return
    agent = session.agent
    if agent is None:
        return
    move = agent.choose_move(session.game.clone(), PLAYER_TWO)
    session.game.drop_piece(move)


def handle_create_game(data: dict) -> tuple[str, list[tuple[str, str]], bytes]:
    mode = data.get("mode", "human_vs_human")
    valid_modes = {item["id"] for item in MODES}
    if mode not in valid_modes:
        return bad_request("Unsupported mode.")
    session_id = uuid.uuid4().hex
    session = GameSession(mode=mode, game=ConnectFourGame(), agent=build_agent(mode))
    SESSIONS[session_id] = session
    return json_response("201 Created", session_payload(session_id, session))


def handle_get_game(session_id: str) -> tuple[str, list[tuple[str, str]], bytes]:
    session = SESSIONS.get(session_id)
    if session is None:
        return not_found()
    return json_response("200 OK", session_payload(session_id, session))


def handle_move(session_id: str, data: dict) -> tuple[str, list[tuple[str, str]], bytes]:
    session = SESSIONS.get(session_id)
    if session is None:
        return not_found()
    if session.game.is_over():
        return bad_request("This game is already over.")
    column = data.get("column")
    if not isinstance(column, int):
        return bad_request("Column must be an integer.")
    if session.mode != "human_vs_human" and session.game.current_player != PLAYER_ONE:
        return bad_request("Wait for the agent to move.")
    try:
        session.game.drop_piece(column)
    except ValueError as error:
        return bad_request(str(error))
    maybe_make_agent_move(session)
    return json_response("200 OK", session_payload(session_id, session))


def application(environ: dict, start_response) -> list[bytes]:
    method = environ["REQUEST_METHOD"]
    path = environ.get("PATH_INFO", "")

    if method == "OPTIONS":
        status, headers, payload = json_response("204 No Content", {})
        start_response(status, headers)
        return [payload]

    if method == "GET" and path == "/":
        status, headers, payload = file_response(STATIC_DIR / "index.html", "text/html; charset=utf-8")
    elif method == "GET" and path == "/app.js":
        status, headers, payload = file_response(STATIC_DIR / "app.js", "application/javascript; charset=utf-8")
    elif method == "GET" and path == "/styles.css":
        status, headers, payload = file_response(STATIC_DIR / "styles.css", "text/css; charset=utf-8")
    elif method == "GET" and path == "/api/modes":
        status, headers, payload = json_response("200 OK", {"modes": MODES})
    elif method == "POST" and path == "/api/games":
        status, headers, payload = handle_create_game(get_request_json(environ))
    elif method == "GET" and path.startswith("/api/games/"):
        session_id = path.rsplit("/", 1)[-1]
        status, headers, payload = handle_get_game(session_id)
    elif method == "POST" and path.startswith("/api/games/") and path.endswith("/move"):
        session_id = path.split("/")[-2]
        status, headers, payload = handle_move(session_id, get_request_json(environ))
    else:
        status, headers, payload = not_found()

    start_response(status, headers)
    return [payload]


def main() -> None:
    host = "127.0.0.1"
    port = 8000
    print(f"Serving Connect Four on http://{host}:{port}")
    with make_server(host, port, application) as server:
        server.serve_forever()


if __name__ == "__main__":
    main()
