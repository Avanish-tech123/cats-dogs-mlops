# CI/CD Pipeline Setup Guide

## Overview
This project uses GitHub Actions for continuous integration and deployment.

## Pipeline Stages

### 1. Test Stage
- Runs on every push and pull request
- Executes unit tests using pytest
- Generates code coverage report
- Must pass before build stage runs

### 2. Build Stage
- Runs after tests pass
- Builds Docker image using Dockerfile
- Uses Docker BuildKit for efficient caching
- Validates image was built successfully

### 3. Publish Stage
- Only runs on pushes to main branch
- Pushes Docker image to Docker Hub
- Tags image with:
  - Branch name
  - Git commit SHA
  - "latest" tag for main branch

## Required GitHub Secrets

Add these secrets in your GitHub repository settings:

```
DOCKER_USERNAME=<your-dockerhub-username>
DOCKER_PASSWORD=<your-dockerhub-access-token>
```

### How to create Docker Hub access token:
1. Log in to [Docker Hub](https://hub.docker.com/)
2. Go to **Account Settings** → **Security**
3. Click **New Access Token**
4. Give it a description (e.g., "GitHub Actions")
5. Set permissions to **Read, Write, Delete**
6. Copy the token and add it as `DOCKER_PASSWORD` secret in GitHub

### How to add secrets to GitHub:
1. Go to your GitHub repository
2. Navigate to **Settings** → **Secrets and variables** → **Actions**
3. Click **New repository secret**
4. Add `DOCKER_USERNAME` with your Docker Hub username
5. Add `DOCKER_PASSWORD` with your Docker Hub access token

## Local Testing

Before pushing, run tests locally:

```bash
# Install dependencies
pip install -r requirements.txt

# Run unit tests
pytest test_app.py -v

# Run with coverage
pytest test_app.py -v --cov=app --cov-report=html

# Build Docker image locally
docker build -t cat-dog-classifier:local .

# Test Docker image locally
docker run -d --name test-api -p 8000:8000 cat-dog-classifier:local
curl http://localhost:8000/health
```

## Triggering the Pipeline

The pipeline triggers automatically on:
- Push to main or develop branches
- Pull requests to main branch

## Monitoring Pipeline Status

View pipeline status:
1. Go to your GitHub repository
2. Click the "Actions" tab
3. Select a workflow run to see details

## Docker Hub Integration

Successfully built images are pushed to:
```
docker.io/<your-username>/cat-dog-classifier:latest
docker.io/<your-username>/cat-dog-classifier:<branch>-<sha>
```

Pull the latest image:
```bash
docker pull <your-username>/cat-dog-classifier:latest
docker run -d -p 8000:8000 <your-username>/cat-dog-classifier:latest
```

## Unit Tests Coverage

The test suite includes:
- **Data Preprocessing Tests** (5 tests)
  - Image shape validation
  - Tensor type verification
  - Device placement
  - Grayscale to RGB conversion
  - Normalization validation

- **Model Architecture Tests** (3 tests)
  - Model initialization
  - Forward pass validation
  - Output range verification

- **Inference Function Tests** (5 tests)
  - Return type validation
  - Required keys verification
  - Class validity check
  - Confidence range validation
  - Probability sum verification

- **Image Format Tests** (8 tests)
  - Various image sizes
  - Multiple color modes

## Workflow File Location
`.github/workflows/ci-pipeline.yml`

## Pipeline Workflow
```
Code Push → Run Tests → Build Docker Image → Push to Registry
              ↓              ↓                    ↓
           (Pass)         (Pass)            (Main branch only)
```

## Troubleshooting

### Tests Failing Locally
- Ensure all dependencies are installed: `pip install -r requirements.txt`
- Check if model file exists: `models/cnn_best.pt`
- Verify Python version: `python --version` (should be 3.9+)

### Docker Build Failing
- Check Dockerfile syntax
- Ensure all required files exist (app.py, requirements.txt, models/)
- Verify Docker is running: `docker --version`

### GitHub Actions Failing
- Check if secrets are configured correctly
- Review Actions logs in GitHub UI
- Ensure branch protection rules allow workflow runs

## Alternative CI Platforms

### GitLab CI (.gitlab-ci.yml)
```yaml
stages:
  - test
  - build
  - publish

test:
  stage: test
  image: python:3.9-slim
  script:
    - pip install -r requirements.txt
    - pytest test_app.py -v --cov=app
```

### Jenkins (Jenkinsfile)
```groovy
pipeline {
    agent any
    stages {
        stage('Test') {
            steps {
                sh 'pip install -r requirements.txt'
                sh 'pytest test_app.py -v'
            }
        }
        stage('Build') {
            steps {
                sh 'docker build -t cat-dog-classifier .'
            }
        }
    }
}
```

## Best Practices

1. **Always run tests locally before pushing**
2. **Use feature branches and pull requests**
3. **Monitor pipeline failures and fix immediately**
4. **Keep Docker images small** (use multi-stage builds if needed)
5. **Tag images properly** (version, commit SHA, latest)
6. **Rotate Docker Hub tokens regularly**
7. **Review test coverage reports**

## Next Steps

1. Configure GitHub secrets (DOCKER_USERNAME, DOCKER_PASSWORD)
2. Commit all changes to your repository
3. Push to GitHub to trigger the CI pipeline
4. Monitor the pipeline in the Actions tab
5. Once successful, verify Docker image on Docker Hub

