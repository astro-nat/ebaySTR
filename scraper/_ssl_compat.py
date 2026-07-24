"""SSL context helper for httpx.

Some Windows boxes run AV products (Norton, Kaspersky, ESET) or sit
behind a corp proxy (ZScaler, Netskope, Bluecoat) that perform TLS
inspection. These rewrite every server cert with a private root CA
that lives in the OS certificate store but NOT in certifi's bundle.
httpx by default uses certifi explicitly, which means
`truststore.inject_into_ssl()` alone does not rescue it — httpx still
passes `cafile=certifi.where()` to ssl.create_default_context().

The fix: build an `ssl.SSLContext` via the `truststore` library and
hand it to httpx as `verify=ctx`. truststore's context defers to the
OS native store (Windows CryptoAPI / macOS Security / Linux ca-certs),
so the inspector's root is trusted.

`make_ssl_context()` returns:
  - a truststore-backed SSLContext when truststore is importable
  - True (httpx's default = certifi) otherwise

Either is a valid `verify=` argument for httpx.AsyncClient.
"""
from __future__ import annotations

import ssl
from typing import Union


def make_ssl_context() -> Union[ssl.SSLContext, bool]:
    """Build an OS-native SSL context for httpx, falling back to certifi."""
    try:
        import truststore  # type: ignore
    except ImportError:
        return True  # httpx default → certifi
    ctx = truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    return ctx
