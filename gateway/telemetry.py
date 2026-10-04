import base64
import os
from contextlib import contextmanager

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from prometheus_client import CollectorRegistry, Counter, Histogram


class Telemetry:
    def __init__(self):
        self.registry = CollectorRegistry()
        self.requests = Counter("gateway_requests_total", "Completed gateway requests", ["tier", "outcome"], registry=self.registry)
        self.latency = Histogram("gateway_request_latency_seconds", "End-to-end request latency", ["tier"], buckets=(.01, .05, .1, .25, .5, 1, 2, 4, 8, 16, 32, 60, 120, 240), registry=self.registry)
        self.ttft = Histogram("gateway_backend_ttft_seconds", "Backend time to first output", ["tier"], buckets=(.01, .05, .1, .25, .5, 1, 2, 4, 8, 16, 32, 60), registry=self.registry)
        self.tokens = Counter("gateway_tokens_total", "Backend tokens (estimated when usage unavailable)", ["tier", "direction", "source"], registry=self.registry)
        self.cost = Counter("gateway_cost_usd_total", "Configured token cost; excludes infrastructure and invisible failed attempts", ["tier"], registry=self.registry)
        self.cache = Counter("gateway_cache_total", "Cache lookups", ["result"], registry=self.registry)
        self.rejections = Counter("gateway_rate_limit_rejections_total", "Rate-limit rejections", registry=self.registry)
        self.backend_calls = Counter("gateway_backend_calls_total", "Backend attempt outcomes", ["tier", "outcome"], registry=self.registry)
        self.fallbacks = Counter("gateway_fallbacks_total", "Fallback attempts", ["from_tier", "to_tier"], registry=self.registry)
        self.circuit_skips = Counter("gateway_circuit_skips_total", "Open circuit skips", ["tier"], registry=self.registry)
        self.capability_rejections = Counter("gateway_capability_rejections_total", "Incompatible candidate tiers during accepted preflight", ["tier", "reason"], registry=self.registry)
        self.capability_reroutes = Counter("gateway_capability_reroutes_total", "Initial selection changed by capability or client policy", ["from_tier", "to_tier"], registry=self.registry)
        self.provider = TracerProvider(resource=Resource.create({"service.name": "intelligent-llm-gateway"}))
        endpoint = os.getenv("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT")
        if endpoint:
            headers = {}
            public, secret = os.getenv("LANGFUSE_PUBLIC_KEY"), os.getenv("LANGFUSE_SECRET_KEY")
            if public and secret:
                headers["Authorization"] = "Basic " + base64.b64encode(f"{public}:{secret}".encode()).decode()
            self.provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint, headers=headers)))
        self.tracer = self.provider.get_tracer("gateway", "0.1.0")

    @contextmanager
    def span(self, name: str, parent=None, **attributes):
        context = trace.set_span_in_context(parent) if parent else None
        # Do not keep a context attached across async-generator yields (different ASGI tasks).
        with self.tracer.start_span(name, context=context) as span:
            span.set_attribute("langfuse.observation.type", "generation" if name == "backend" else "span")
            for key, value in attributes.items():
                if value is not None:
                    span.set_attribute(key, value)
            yield span

    def charge(self, tier, usage: dict, source: str):
        incoming, outgoing = usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0)
        self.tokens.labels(tier.name, "input", source).inc(max(0, incoming))
        self.tokens.labels(tier.name, "output", source).inc(max(0, outgoing))
        self.cost.labels(tier.name).inc(max(0, incoming * tier.price.input_per_million + outgoing * tier.price.output_per_million) / 1e6)

    def close(self):
        self.provider.shutdown()
