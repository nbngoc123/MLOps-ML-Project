# Use the official PostgreSQL image
FROM postgres:latest

# Set environment variables for PostgreSQL
# These match the settings in the docker-compose.yaml
ENV POSTGRES_USER=postgres
ENV POSTGRES_PASSWORD=aivn2025
ENV POSTGRES_DB=image_db

# The volume for PostgreSQL data is handled by docker-compose.yaml (or will be handled by Kubernetes PVC).
# The base image already sets up the necessary data directory at /var/lib/postgresql/data.
# No explicit VOLUME instruction is needed here if we rely on the base image's configuration
# and the external volume mapping.

# Expose the default PostgreSQL port
EXPOSE 5432
