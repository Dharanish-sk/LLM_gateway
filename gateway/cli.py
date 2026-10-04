import argparse
import hashlib
import json
import os
import secrets

import uvicorn


def main():
    parser = argparse.ArgumentParser(description="Intelligent LLM Gateway")
    commands = parser.add_subparsers(dest="command", required=True)
    serve = commands.add_parser("serve")
    serve.add_argument("--config", default="config/gateway.yaml")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", default=8000, type=int)
    key = commands.add_parser("keygen", help="Print a new secret and hashed file record")
    key.add_argument("--id", default="client-1")
    args = parser.parse_args()
    if args.command == "keygen":
        raw = "gw_" + secrets.token_urlsafe(32)
        print("Secret (save securely):", raw)
        print("Credential file record:")
        print(json.dumps({"id": args.id, "sha256": hashlib.sha256(raw.encode()).hexdigest(), "enabled": True}, indent=2))
    else:
        os.environ["GATEWAY_CONFIG"] = args.config
        uvicorn.run("gateway.app:create_app", factory=True, host=args.host, port=args.port, access_log=False)


if __name__ == "__main__":
    main()
