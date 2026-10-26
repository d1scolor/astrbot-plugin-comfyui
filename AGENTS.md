# Project instructions

- This is a public repository. Never commit credentials, private service URLs,
  hardware/account identifiers, diagnostic bundles or private conversation logs.
- Do not use SSH keys to access other machines without explicit permission.
  Do not assume authorization to access deployments or send QQ messages.
- Do not dump container environments or provider configuration.
- Use uv for Python dependency and development environment management.
- Keep shell execution disabled for bot users. Media delivery must not require
  granting shell access, administrator status or arbitrary filesystem access.
- Read HANDOFF.md before implementation. Verify current AstrBot SDK interfaces
  against the supported version before writing integration code.
