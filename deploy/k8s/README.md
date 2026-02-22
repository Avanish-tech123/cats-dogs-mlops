Kubernetes Deployment for Cat-Dog Classifier

This directory contains Kubernetes manifests for deploying the Cat-Dog Classifier to a local cluster (kind/minikube/microk8s) or a remote Kubernetes cluster.

Files:
- `deployment.yaml` - Deployment manifest (2 replicas, resource limits, liveness/readiness probes)
- `service.yaml` - Service manifest (NodePort for local access)

Quick start (minikube):

1. Start cluster:

```bash
minikube start
```

2. Make sure the image is accessible to the cluster. For minikube you can load the image directly:

```bash
minikube image load <your-username>/cat-dog-classifier:latest
```

Or ensure the cluster can pull images from Docker Hub (set `DOCKER_USERNAME` in `.env` and push the image).

3. Apply manifests:

```bash
minikube kubectl -- apply -f deploy/k8s/deployment.yaml
minikube kubectl -- apply -f deploy/k8s/service.yaml
```

4. Access the service:

```bash
# For minikube
minikube service cat-dog-classifier --url

# Or port-forward
minikube kubectl -- port-forward svc/cat-dog-classifier 8000:8000
```

5. Run smoke tests:

```bash
./deploy/smoke_tests.sh all
```

Notes:
- For cloud environments, change `Service` type to `LoadBalancer` and configure Ingress accordingly.
- Update image name in `deployment.yaml` or set `DOCKER_USERNAME` in your environment before applying.
