# Security Policy

## Supported versions

| Version | Supported |
|---|---|
| 0.1.x | yes |
| < 0.1 | no |

Only the latest minor release receives security fixes.

## Reporting a vulnerability

Please do not open a public issue. Use GitHub private vulnerability reporting:
open the repository's **Security** tab and choose **Report a vulnerability**
(https://github.com/thnhdatle/footnote/security/advisories/new).

Include the affected version, a description, reproduction steps and the impact.

## Response targets

- Acknowledgement within 72 hours.
- Initial assessment and a fix or mitigation plan as soon as it is triaged;
  we will keep you informed and credit you in the advisory unless you prefer
  otherwise.

## Scope

Tenant isolation (row-level security), authentication, document parsing,
webhook delivery and the container image are in scope. Findings in
third-party dependencies should be reported upstream; tell us if Footnote's
usage makes them exploitable.
