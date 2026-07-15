# Hosted UI

This directory is the static browser interface for the AWS-hosted product. It is intentionally independent of `web_dashboard.py`, which remains a local-only CLI dashboard.

The checked-in `config.js` uses `preview` mode. Preview mode exercises navigation, responsive layout, file validation, audit progress, history filters, and comparison interactions without sending data or pretending that authentication exists. Build an exact preview bundle with:

```powershell
python scripts/build_web.py --mode preview
```

Live mode accepts only public browser configuration: API base URL, Cognito domain, client ID, and redirect URL. Never place credentials, client secrets, access tokens, dataset keys, or AWS keys in this directory or its generated bundle.

Live mode uses Cognito Authorization Code with PKCE, direct owner-scoped S3 uploads, asynchronous job polling, owner history, and short-lived artifact downloads. Browser tokens are held in session storage and authorization remains enforced by API Gateway and the API ownership checks, never by hidden controls.
