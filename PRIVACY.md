# Hosted alpha privacy notice

Last updated: 2026-07-17

This notice applies to the hosted Dataset Quality Analyzer private alpha. The local CLI processes data on your own machine and does not send it to the hosted service.

## Data handled

The hosted service receives the ZIP archive you choose to upload, the email address attached to your invited account, job settings and status, generated audit reports, and limited operational logs. Logs use job and account identifiers for diagnosis; dataset contents and passwords are not intentionally logged.

Data is processed in AWS `us-east-1`. It is used only to authenticate access, run the requested audit, return results, prevent abuse, and operate the alpha. It is not sold or used to train a model.

## Access and retention

Uploaded archives and generated artifacts are stored in private, encrypted S3 buckets. Access to jobs and downloads is scoped to the authenticated account. Workers are isolated one-shot tasks and treat the uploaded source as read-only.

- Unfinished multipart uploads are abandoned after one day.
- Source archives are normally deleted after a completed job and have a two-day storage lifecycle backstop.
- Successful report downloads are offered for seven days; artifact objects have an eight-day lifecycle backstop.
- Failed-job artifacts are logically unavailable after 48 hours and are removed by the artifact lifecycle backstop.
- Job metadata expires after 30 days.
- Development operational logs are retained for 7 or 14 days, depending on the component.

Lifecycle deletion is asynchronous, so an expired object can remain briefly while AWS processes the rule. The hosted alpha is not permanent storage.

## Your choices

Do not upload secrets, health records, financial records, government identifiers, export-controlled data, or any dataset you are not authorized to process. Use the local CLI when data must not leave your machine.

After a job reaches a terminal state, use **Delete source** in the dashboard for an immediate source-deletion request. To request account removal or help with deletion, email `atikul.munna@northsouth.edu` and include the job ID when relevant. Do not email dataset contents, passwords, or access tokens.

This is a small, selected-tester alpha with no availability or recovery guarantee. Material changes to this notice will be recorded in this repository before broader availability.
