"""Shared flask-caching instance — imported by all pages and shared modules."""
from flask_caching import Cache

cache = Cache(config={"CACHE_TYPE": "SimpleCache"})
