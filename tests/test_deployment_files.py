import json
from pathlib import Path

import yaml

from gateway.config import load_settings


def test_deployment_config_and_dashboard_metric_references():
    config = load_settings("config/gateway.yaml")
    assert config.mode == "production" and config.state == "redis"
    compose = yaml.safe_load(Path("compose.yaml").read_text())
    assert {"gateway", "redis", "ollama", "grafana", "prometheus"} <= set(compose["services"])
    dashboard = json.loads(Path("ops/grafana/dashboard.json").read_text())
    assert dashboard["uid"] == "intelligent-llm-gateway"
    assert len(dashboard["panels"]) == 8
    # Ensure dashboards don't silently point to nonexistent metric names.
    from gateway.telemetry import Telemetry
    telemetry = Telemetry()
    try:
        names = {metric.name for metric in telemetry.registry.collect()}
        for panel in dashboard["panels"]:
            for target in panel["targets"]:
                import re
                for metric in re.findall(r"gateway_[a-z_]+", target["expr"]):
                    base = re.sub(r"_(total|bucket|count|sum)$", "", metric)
                    assert base in names
    finally:
        telemetry.close()


def test_provider_catalog_native_and_container_endpoints(monkeypatch):
    monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)
    native = load_settings("config/providers.yaml")
    assert {tier.name for tier in native.tiers} >= {"claude", "gemini", "qwen", "deepseek"}
    assert native.cache.embedding_url == "http://localhost:11434"
    original_hash = native.policy_hash()
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://ollama:11434")
    container = load_settings("config/providers.yaml")
    assert container.cache.embedding_url == "http://ollama:11434"
    assert next(tier for tier in container.tiers if tier.name == "local").base_url == "http://ollama:11434"
    assert next(tier for tier in container.tiers if tier.name == "claude").base_url == "https://api.anthropic.com/v1"
    assert container.policy_hash() != original_hash
