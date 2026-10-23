"""Minimal ComfyUI HTTP client: queue an API-format graph, wait, report per-node timing from the websocket."""
import json
import time
import urllib.request
import uuid

import websocket  # websocket-client, in ComfyUI's venv


class Client:
    def __init__(self, host="127.0.0.1:8199"):
        self.host = host
        self.client_id = uuid.uuid4().hex

    def _post(self, path, body):
        req = urllib.request.Request(f"http://{self.host}{path}", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req) as resp:
            return json.loads(resp.read() or b"{}")

    def _get(self, path):
        with urllib.request.urlopen(f"http://{self.host}{path}") as resp:
            return json.loads(resp.read())

    def free(self):
        self._post("/free", {"unload_models": True, "free_memory": True})

    def run(self, graph, verbose=True):
        """Queue `graph`, follow it on the websocket; returns (history entry, {node: seconds}, wall seconds)."""
        ws = websocket.create_connection(f"ws://{self.host}/ws?clientId={self.client_id}", timeout=3600)
        try:
            resp = self._post("/prompt", {"prompt": graph, "client_id": self.client_id})
        except urllib.error.HTTPError as exc:
            raise RuntimeError(exc.read().decode()) from None
        prompt_id = resp["prompt_id"]
        start = time.time()
        node_start, current, timings = None, None, {}
        while True:
            msg = ws.recv()
            if not isinstance(msg, str):
                continue
            data = json.loads(msg)
            kind, body = data.get("type"), data.get("data", {})
            if body.get("prompt_id") not in (None, prompt_id):
                continue
            if kind == "executing":
                now = time.time()
                if current is not None:
                    timings[current] = timings.get(current, 0.0) + now - node_start
                current, node_start = body.get("node"), now
                if current is None:
                    break
                if verbose:
                    print(f"  [{now - start:7.1f}s] node {current} ({graph[current]['class_type']})", flush=True)
            elif kind == "execution_error":
                ws.close()
                raise RuntimeError(f"node {body.get('node_id')} {body.get('node_type')}: {body.get('exception_message')}\n"
                                   + "".join(body.get("traceback", [])[-6:]))
            elif kind == "progress" and verbose and body.get("value") == body.get("max"):
                print(f"            steps {body.get('value')}/{body.get('max')} done", flush=True)
        ws.close()
        wall = time.time() - start
        history = self._get(f"/history/{prompt_id}")[prompt_id]
        return history, timings, wall
