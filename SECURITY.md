# Security

Maestro runs on your machine and holds your provider keys, so security reports matter.

**Please report a vulnerability privately**, through this repository's **Security** tab ("Report a vulnerability"). Do not open a public issue for it. Include what you found, how to reproduce it, and the version (`maestro-router --version`).

What is in scope: anything that could expose a key from `~/.maestro/keys.env`, the admin token, or prompt text; requests reaching a provider other than the one a key belongs to; `maestro-router serve` accepting requests it should refuse (another origin, beyond loopback without `MAESTRO_API_KEY`); and the website at https://maestro.this.live.

How Maestro handles secrets is described in [README.md](README.md#privacy-and-security) and in the contributor rules in [CLAUDE.md](CLAUDE.md).
