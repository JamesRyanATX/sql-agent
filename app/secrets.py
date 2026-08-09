"""How a warehouse password is written down.

Registering a connection means handing the agent credentials to somebody else's
database, and those credentials have to survive a restart — so they land in
`connection.password` on agent-db. This module is the only thing that reads or
writes that column's contents.

**Be precise about what encryption buys here.** With `CONNECTION_SECRET` set, a
stolen agent-db dump is useless on its own, and `make psql-agent` over your
shoulder shows a token rather than a password. It buys nothing at all against a
compromised API process: that process holds the key by definition, and unsealing
is the job it exists to do. Encryption at rest is not access control.

Empty `CONNECTION_SECRET` stores plaintext and the server says so at startup.
That is the same bargain `API_TOKEN` already ships — insecure defaults are fine
when they are loud, and a second policy for the second secret would be one more
thing to keep true.
"""

from __future__ import annotations

from functools import lru_cache

from app.settings import settings

_PLAIN = "plain:"
_FERNET = "fernet:"


class SecretError(RuntimeError):
    """A stored password could not be read back."""


@lru_cache
def _cipher(key: str):
    """Built once per key. Fernet does scrypt-free key setup, but the import is
    heavy enough that a per-request construction shows up under a fan-out."""
    from cryptography.fernet import Fernet

    return Fernet(key.encode())


def seal(password: str | None) -> str | None:
    """Encode a password for storage. Returns a tagged string, or None.

    Tagged in the value rather than flagged in a column, so turning encryption
    on is a config change and not a migration, and so a mixed table — some rows
    written before the key existed — reads correctly without a backfill.
    """
    if password is None:
        return None
    key = settings().connection_secret
    if not key:
        return _PLAIN + password
    return _FERNET + _cipher(key).encrypt(password.encode()).decode()


def unseal(stored: str | None) -> str | None:
    """Decode what `seal` wrote.

    A decrypt failure raises rather than returning something. The alternative —
    handing back a garbage password — surfaces as an authentication failure at
    the warehouse, which reads like the user typed their password wrong and
    sends them off fixing the wrong thing.
    """
    if stored is None:
        return None
    if stored.startswith(_PLAIN):
        return stored[len(_PLAIN) :]
    if stored.startswith(_FERNET):
        key = settings().connection_secret
        if not key:
            raise SecretError(
                "this connection's password was encrypted, but CONNECTION_SECRET "
                "is unset — set it to the key it was written with"
            )
        from cryptography.fernet import InvalidToken

        try:
            return _cipher(key).decrypt(stored[len(_FERNET) :].encode()).decode()
        except InvalidToken as e:
            raise SecretError(
                "this connection's password does not decrypt with the current "
                "CONNECTION_SECRET — it was written with a different key"
            ) from e
    # Rows predating the tag, if any ever exist. Reading them as plaintext is
    # the only interpretation that can be right.
    return stored
