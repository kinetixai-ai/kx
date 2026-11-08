# Security

Please report security problems privately, not in public issues:
use the "Report a vulnerability" button on this repository's Security tab.

Include what you found, how to reproduce it, and which version of kx
(`kx --version`). We aim to reply within three working days.

What kx does to keep you safe:
- Your API token is kept in the OS keychain when `keyring` is installed,
  otherwise in an owner-only file. kx never prints it and never sends it
  over plain http except to localhost.
- `kx rag` runs on your machine; with a local Ollama nothing leaves it.
- Release files on GitHub come with `SHA256SUMS`.
