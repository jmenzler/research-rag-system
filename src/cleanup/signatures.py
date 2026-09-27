"""Fail-page literal-token signatures.

Single source of truth — referenced by extractor (real-time gates) and audit
(post-extraction sweep). Multilingual: English, German, French, Spanish.
"""

from __future__ import annotations

# Literal-token detectors for known-bad pages: Cloudflare WAF challenge, 404s,
# vendor-specific block notices, login walls. Each entry is matched as a
# substring against the lowercased first 1500 chars of fetched text.
FAIL_PAGE_SIGNATURES: tuple[str, ...] = (
    # Cloudflare English
    "just a moment",
    "performing security verification",
    "enable javascript and cookies to continue",
    "privacy and security by cloudflare",
    "ray id:",
    "checking your browser",
    "ddos protection by cloudflare",
    "this website uses a security service to protect",
    "verifying you are human",
    # Cloudflare German
    "sicherheitsüberprüfung wird durchgeführt",
    "diese website nutzt einen sicherheitsservice",
    "überprüfung erfolgreich",
    # Cloudflare French
    "vérification de sécurité",
    "vous êtes un humain",
    # Cloudflare Spanish
    "comprobando que eres humano",
    "verificación de seguridad",
    # Generic block / dead pages
    "access denied",
    "this site can",  # browser-error "this site can't be reached"
    "404 not found",
    "403 forbidden",
    "401 unauthorized",
    "page not found",
    "are you a robot",
    "captcha",
    "before you continue to google",
    "we'll remember this choice",
    # Vendor-specific block pages observed in corpus audit
    "marketchameleon",
    "possible reasons for receiving this error",
    "error 1020",
    "using a vpn or other security product",
)


# Subset of FAIL_PAGE_SIGNATURES that specifically indicate a Cloudflare bot
# challenge. Used to decide whether the stealth tiers should attempt CF solve,
# vs giving up (e.g. for genuine 404s or vendor-specific blocks).
CLOUDFLARE_SIGNATURES: frozenset[str] = frozenset({
    "just a moment",
    "performing security verification",
    "enable javascript and cookies to continue",
    "privacy and security by cloudflare",
    "ray id:",
    "checking your browser",
    "ddos protection by cloudflare",
    "this website uses a security service to protect",
    "verifying you are human",
    "sicherheitsüberprüfung wird durchgeführt",
    "diese website nutzt einen sicherheitsservice",
    "überprüfung erfolgreich",
    "vérification de sécurité",
    "vous êtes un humain",
    "comprobando que eres humano",
    "verificación de seguridad",
    "are you a robot",
    "error 1020",
})


def is_cloudflare_signature(sig: str) -> bool:
    """True if a fail signature indicates a Cloudflare challenge (escalate-able)."""
    return sig in CLOUDFLARE_SIGNATURES
