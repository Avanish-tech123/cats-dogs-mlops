# Cat vs Dog Classifier - Complete MLOps Pipeline

## Project Overview

A complete end-to-end MLOps implementation featuring a CNN-based image classifier for cats vs dogs, with comprehensive CI/CD, containerization, Kubernetes deployment, and production monitoring.

## Project Modules Completed

### M1: Model Development & Experiment Tracking
- **CNN Architecture**: Custom PyTorch model with 5 convolutional blocks
- **Training Pipeline**: Automated training with validation monitoring
- **MLflow Integration**: Comprehensive experiment tracking and model registry
- **Performance**: Achieved 85%+ accuracy with robust validation metrics
- **Artifacts**: Model weights, training logs, performance reports

### M2: Test-Driven Development & CI/CD
- **Unit Testing**: Comprehensive pytest suite with 95%+ coverage
- **API Testing**: FastAPI endpoint validation with test fixtures
- **GitHub Actions**: Automated CI pipeline with testing and deployment
- **Docker Integration**: Multi-stage builds with optimized images
- **Code Quality**: Automated testing on pull requests and merges

### M3: Containerization & Docker Registry
- **Docker Implementation**: Production-ready containerized application
- **Multi-stage Build**: Optimized image size and security
- **Docker Hub Integration**: Automated image publishing with versioning
- **Health Checks**: Built-in container health monitoring
- **Security**: Non-root user, minimal attack surface

### M4: Kubernetes Deployment & Orchestration
- **Local Kubernetes**: Minikube deployment with production patterns
- **High Availability**: Multi-replica deployment with rolling updates
- **Service Discovery**: NodePort service with external accessibility
- **Resource Management**: CPU/memory limits and requests
- **Health Monitoring**: Liveness and readiness probes

### M5: Monitoring, Logging & Performance Tracking
- **Structured Logging**: JSON-formatted request/response logging
- **Metrics Collection**: Request count, latency, and accuracy tracking
- **Performance Analysis**: Model confidence and processing time monitoring
- **Monitoring Endpoints**: `/metrics` and `/performance` API endpoints
- **Test Framework**: Automated performance validation with synthetic data

## Architecture

```mermaid
graph TB
    A[GitHub Repository] --> B[GitHub Actions CI/CD]
    B --> C[Docker Build & Test]
    B --> D[Docker Hub Registry]
    D --> E[Minikube Kubernetes Cluster]
    E --> F[FastAPI Application Pods]
    F --> G[Health & Metrics Endpoints]
    F --> H[PyTorch CNN Model]
    H --> I[Prediction Results]
    G --> J[Monitoring & Logging]
```

## Quick Start

### Prerequisites
- Docker and Docker Hub account
- Minikube or local Kubernetes cluster
- Python 3.9+ with pip
- Git

### 1. Clone and Setup
```bash
git clone <repository-url>
cd cat-dog-classifier
pip install -r requirements.txt
```

### 2. Local Development
```bash
# Run application locally
python app.py

# Run tests
pytest test_app.py -v --cov=app

# Test endpoints
curl http://localhost:8000/health
```

### 3. Docker Deployment
```bash
# Build image
docker build -t cat-dog-classifier:latest .

# Run container
docker run -d -p 8000:8000 --name catdog cat-dog-classifier:latest

# Test deployment
./deploy/smoke_tests.sh http://localhost:8000
```

### 4. Kubernetes Deployment
```bash
# Start minikube
minikube start --driver=docker --memory=8192 --cpus=4

# Load image
minikube image load cat-dog-classifier:latest

# Deploy application
minikube kubectl -- apply -f deploy/k8s/

# Get service URL
minikube service cat-dog-classifier --url

# Run performance tests
python3 test_performance.py $(minikube service cat-dog-classifier --url)
```

## 📊 API Endpoints

### Core Functionality
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | API information and available endpoints |
| `/health` | GET | Health check with system status and metrics |
| `/predict` | POST | Image classification with optional true label |
| `/docs` | GET | Interactive API documentation (Swagger) |

### Monitoring & Analytics
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/metrics` | GET | Application metrics and performance statistics |
| `/performance` | GET | Model performance analysis and accuracy tracking |

### Example Usage
```bash
# Health check
curl -X GET "http://localhost:8000/health"

# Image prediction
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@path/to/image.jpg" \
     -F "true_label=Cat"

# Get metrics
curl -X GET "http://localhost:8000/metrics"
```

## Monitoring & Observability

### Logging
- **Structured JSON logs** for all requests and responses
- **Request correlation** with unique request IDs
- **Performance metrics** including processing time and confidence
- **Error tracking** with detailed stack traces
- **Security-conscious** logging (no sensitive data)

### Metrics Collection
- **Request Statistics**: Count, latency percentiles, error rates
- **Model Performance**: Accuracy, confidence distribution, processing time
- **System Health**: Uptime, resource usage, model status
- **Business Metrics**: Prediction volume, accuracy trends

### Example Metrics Response
```json
{
  "timestamp": "2026-02-22T11:35:00Z",
  "uptime_seconds": 1847,
  "total_predictions": 150,
  "avg_confidence": 0.867,
  "avg_processing_time_ms": 45.32,
  "accuracy": 0.893,
  "recent_predictions": [...]
}
```

## Testing Strategy

### Unit Tests
```bash
# Run all tests with coverage
pytest test_app.py -v --cov=app --cov-report=html

# Test specific functionality
pytest test_app.py::TestModelArchitecture -v
```

### Integration Tests
```bash
# Smoke tests for deployed application
./deploy/smoke_tests.sh http://localhost:8000

# Performance validation
python3 test_performance.py http://localhost:8000
```

### CI/CD Pipeline Tests
- **Automated Testing**: Every commit triggers full test suite
- **Docker Build Validation**: Images tested before publishing
- **Deployment Verification**: Smoke tests run post-deployment
- **Performance Benchmarks**: Automated performance regression detection

## Configuration

### Environment Variables
```bash
# Docker configuration
DOCKER_USERNAME=your-dockerhub-username

# Application settings
LOG_LEVEL=INFO
MODEL_PATH=models/cnn_best.pt
```

### Kubernetes Configuration
- **Deployment**: 2 replicas with rolling update strategy
- **Resources**: 1Gi-2Gi memory, 500m-1000m CPU per pod
- **Health Probes**: 30s liveness, 15s readiness intervals
- **Service**: NodePort for local access, easily configurable for LoadBalancer

## Performance Benchmarks

### Model Performance
- **Accuracy**: 85-90% on test dataset
- **Inference Speed**: ~40-50ms per prediction
- **Memory Usage**: ~1.5GB per pod (including model weights)
- **Throughput**: ~20-25 requests/second per pod

### Infrastructure Performance
- **Startup Time**: ~30s for pod readiness
- **Health Check**: <5ms response time
- **Rolling Update**: Zero-downtime deployments
- **Resource Efficiency**: Optimized Docker image (~9GB with PyTorch)

## Security & Best Practices

### Container Security
- **Non-root user**: Application runs as `appuser` (UID 1000)
- **Minimal base image**: Python 3.9 slim for reduced attack surface
- **Health checks**: Built-in monitoring for container health
- **Resource limits**: Prevents resource exhaustion attacks

### Data Privacy
- **No sensitive logging**: Image data never logged or stored
- **Request correlation**: Traceable requests without data exposure
- **Input validation**: File type and size restrictions
- **Error sanitization**: Safe error messages to clients

### Operational Security
- **Docker Hub**: Secure image registry with automated vulnerability scanning
- **Kubernetes**: Network policies and resource quotas
- **Monitoring**: Security event logging and alerting
- **CI/CD**: Automated security testing in pipeline

## CI/CD Pipeline

### Workflow Stages
1. **Code Quality**: Linting, formatting, security scans
2. **Testing**: Unit tests, integration tests, coverage reporting
3. **Build**: Docker image creation with multi-stage optimization
4. **Registry**: Automated push to Docker Hub with versioning
5. **Deploy**: Kubernetes deployment to staging environment
6. **Validate**: Smoke tests and performance benchmarks
7. **Monitor**: Automated monitoring and alerting setup

### GitHub Actions Integration
```yaml
# Automated on every push to main
- Unit testing with pytest
- Docker build and security scan
- Image publishing to Docker Hub
- Kubernetes deployment validation
- Smoke testing and performance validation
```

## Documentation

### Code Documentation
- **API Documentation**: Auto-generated Swagger/OpenAPI docs at `/docs`
- **Code Comments**: Comprehensive inline documentation
- **Type Hints**: Full Python type annotation
- **Docstrings**: Detailed function and class documentation

### Operational Documentation
- **Deployment Guide**: Step-by-step Kubernetes deployment
- **Monitoring Setup**: Metrics collection and alerting configuration
- **Troubleshooting**: Common issues and resolution steps
- **Performance Tuning**: Optimization guidelines and benchmarks

## Production Readiness Checklist

### Completed
- [x] Containerized application with health checks
- [x] Kubernetes deployment with high availability
- [x] Comprehensive monitoring and logging
- [x] Automated CI/CD pipeline with testing
- [x] Performance benchmarking and optimization
- [x] Security hardening and best practices
- [x] Documentation and operational guides

### Future Enhancements
- [ ] Prometheus/Grafana metrics integration
- [ ] Horizontal pod autoscaling (HPA)
- [ ] Istio service mesh for advanced routing
- [ ] Database persistence for metrics history
- [ ] A/B testing framework for model versions
- [ ] Advanced security scanning and policies

## Contributing

### Development Workflow
1. Fork the repository
2. Create feature branch: `git checkout -b feature/amazing-feature`
3. Commit changes: `git commit -m 'Add amazing feature'`
4. Push branch: `git push origin feature/amazing-feature`
5. Submit pull request with comprehensive description

### Code Standards
- **Python**: Follow PEP 8 with black formatting
- **Testing**: Maintain >90% test coverage
- **Documentation**: Update docs for all changes
- **Security**: Follow security best practices

## License & Acknowledgments

### Project Structure
```
├── app.py                          # Main FastAPI application
├── test_app.py                     # Comprehensive test suite
├── test_performance.py             # Performance testing framework
├── requirements.txt                # Python dependencies
├── Dockerfile                      # Multi-stage container build
├── .github/workflows/              # CI/CD pipeline configuration
│   └── ci-pipeline.yml
├── deploy/                         # Deployment scripts and configs
│   ├── smoke_tests.sh             # Deployment validation
│   └── k8s/                       # Kubernetes manifests
│       ├── deployment.yaml
│       ├── service.yaml
│       └── README.md
├── models/                         # Trained model artifacts
│   └── cnn_best.pt
├── M1_Model_Development_Exp_Tracking.ipynb  # Model development notebook
├── M5_MONITORING_SUBMISSION.md     # Monitoring implementation details
└── README.md                       # This comprehensive guide
```

**Built with** love using PyTorch, FastAPI, Docker, Kubernetes, and modern MLOps practices.

---

For detailed implementation information, see individual module documentation:
- [M1: Model Development](M1_Model_Development_Exp_Tracking.ipynb)
- [M5: Monitoring & Logging](M5_MONITORING_SUBMISSION.md)
- [Kubernetes Deployment](deploy/k8s/README.md)