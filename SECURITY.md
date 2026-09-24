# Security Policy

## Supported versions

| Version | Supported          |
| ------- | ------------------ |
| 2.x     | :white_check_mark: |

## Reporting a vulnerability

Please **do not** open a public issue. Report suspected vulnerabilities
privately to **mosesnetto@gmail.com** with the subject
`[MERIDIAN-SECURITY]`.

We aim to acknowledge reports within 48 hours and to ship a fix in the next
release.

## Security notes

- `.env` files and LLM API keys must never be committed. `llm_integration.py`
  reads `OPENROUTER_API_KEYS`, `AI_MODEL` and `AI_FALLBACK_MODEL` from the
  environment only.
- The web app is intended to run on a trusted LAN. When exposed publicly,
  put it behind an authenticated reverse proxy.
- User-uploaded PDFs are processed locally; generated artifacts inside
  `uploads/` and `web_output/` are gitignored.