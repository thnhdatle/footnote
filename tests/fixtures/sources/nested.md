# Operations Guide

This guide covers the daily operations of the service.

## Deployment

Deployments follow a fixed sequence of steps.

### Preparation

- Review the release notes
- Back up the database
- Notify the on-call engineer

### Rollout

The rollout proceeds in stages across regions.

- Deploy to staging first
- Verify the health checks
- Promote to production

#### Rollback Details

If a stage fails the release is rolled back automatically.

## Monitoring

Dashboards show latency, errors and saturation.

### Alerts

- Page the engineer on sustained errors
- Open a ticket on slow degradation
