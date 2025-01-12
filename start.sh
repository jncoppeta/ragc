start_milvus() {
    cat << EOF > embedEtcd.yaml
listen-client-urls: http://0.0.0.0:2379
advertise-client-urls: http://0.0.0.0:2379
quota-backend-bytes: 4294967296
auto-compaction-mode: revision
auto-compaction-retention: '1000'
EOF

    cat << EOF > user.yaml
# Extra config to override default milvus.yaml
EOF

    sudo docker run -d \
        --name milvus-standalone \
        --network $1 \
        --security-opt seccomp:unconfined \
        -e ETCD_USE_EMBED=true \
        -e ETCD_DATA_DIR=/var/lib/milvus/etcd \
        -e ETCD_CONFIG_PATH=/milvus/configs/embedEtcd.yaml \
        -e COMMON_STORAGETYPE=local \
        -v $(pwd)/volumes/milvus:/var/lib/milvus \
        -v $(pwd)/embedEtcd.yaml:/milvus/configs/embedEtcd.yaml \
        -v $(pwd)/user.yaml:/milvus/configs/user.yaml \
        -p 19530:19530 \
        -p 9091:9091 \
        -p 2379:2379 \
        --health-cmd="curl -f http://localhost:9091/healthz" \
        --health-interval=30s \
        --health-start-period=90s \
        --health-timeout=20s \
        --health-retries=3 \
        milvusdb/milvus:v2.5.2 \
        milvus run standalone  1> /dev/null
}

# Define the network name
NETWORK_NAME="rag-network"
API_IMAGE="ragc-api:latest"

# Check if the network exists, create it if not
if ! docker network inspect "$NETWORK_NAME" >/dev/null 2>&1; then
  echo "Creating network: $NETWORK_NAME"
  docker network create "$NETWORK_NAME"
fi

start_milvus $NETWORK_NAME;

# Start the services using docker-compose
echo "Building image with custom PDFs..."
docker-compose build --build-arg API_IMAGE=$API_IMAGE
docker-compose up -d

# Wait for the services to start
echo "Waiting for containers to start..."
sleep 10

# Get container IDs for the services
OLLAMA_CONTAINER=$(docker ps -q --filter "name=ollama")
API_CONTAINER=$(docker ps -q --filter "name=api")

# Connect the containers to the network
if [ -n "$OLLAMA_CONTAINER" ]; then
  echo "Connecting Ollama container to $NETWORK_NAME"
  docker network connect "$NETWORK_NAME" "$OLLAMA_CONTAINER"
else
  echo "Ollama container not found. Please check if it's running."
fi

if [ -n "$API_CONTAINER" ]; then
  echo "Connecting API container to $NETWORK_NAME"
  docker network connect "$NETWORK_NAME" "$API_CONTAINER"
else
  echo "API container not found. Please check if it's running."
fi

echo "Containers connected to $NETWORK_NAME successfully."

echo "Waiting for API to be ready..."
FASTAPI_HEALTH_URL="http://localhost:8000/health"
while true; do
    # Check the health endpoint of FastAPI
    res=$(curl --write-out "%{http_code}" --silent --output /dev/null "$FASTAPI_HEALTH_URL")
    if [ "$res" -eq 200 ]; then
        echo "FastAPI is ready."
        break
    fi
    echo "API is not ready yet... Waiting..."
    sleep 10  # Wait for 10 seconds before retrying
done

echo """
==================================================
All containers have started successfully!
You can access the API documentation at: http://localhost:8000/docs
==================================================
"""
