# R3 guest chat

`RESEARCH_R3_GUEST_ENABLED=true` enables anonymous R3 testing independently of
Google/email account login. Disabled by default. It does not authenticate a
visitor for account APIs, research reviews, or the private corpus/evidence host.

A separate HttpOnly, same-site cookie lasts 30 days. Its signed random identity
is hashed in conversation metadata; no email, password or account is created.
Read, list, delete, continuation and transcript feedback require that identity.
Clearing/expiring the cookie loses access to its chats. Existing account and
legacy anonymous conversations retain their original access behavior.

Guest R3 limits are 20 turns/hour per session, 40/hour per ingress IP and 100/hour
globally, plus one active turn per session. These counters are per app process
and reset on restart; they are not a durable spending cap. Caddy is the only
public ingress and app ports must remain loopback-only. The existing private
gateway has its own concurrency and request bounds. Guests do not consent to
training by chatting. Checkpoint validation, default model and raw decoding are
unchanged.

Verification: `npm run test:research`, type checking, lint, and candidate/live
tests with two distinct cookie jars. Confirm cross-session GET/DELETE/POST are
denied, refresh preserves history, own follow-up has three messages, account APIs
stay unauthenticated and the private corpus endpoint still returns 401.

Rollback: disable the flag; existing owner cookies can still read their chats.
No schema migration is required. Reverting to an older application image that
does not enforce guest metadata ownership is unsafe while guest records remain.
