"""How a registered warehouse's password is written down.

The only module that reads or writes `connection.password` on agent-db.

With `CONNECTION_SECRET` set, a stolen agent-db dump is useless on its own. It
buys nothing against a compromised API process, which holds the key by
definition. Empty stores plaintext, and the server warns at startup.
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
    """Built once per key — the cryptography import is heavy under a fan-out."""
    from cryptography.fernet import Fernet

    return Fernet(key.encode())


def seal(password: str | None) -> str | None:
    """Encode a password for storage. Returns a tagged string, or None.

    Tagged in the value rather than flagged in a column, so turning encryption on
    is a config change rather than a migration, and a mixed table still reads.
    """
    if password is None:
        return None
    key = settings().connection_secret
    if not key:
        return _PLAIN + password
    return _FERNET + _cipher(key).encrypt(password.encode()).decode()


def unseal(stored: str | None) -> str | None:
    """Decode what `seal` wrote.

    A decrypt failure raises. Handing back a garbage password instead surfaces as
    an authentication failure at the warehouse, which blames the user's typing.
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
    # Untagged rows, if any exist. Plaintext is the only reading that can be right.
    return stored
