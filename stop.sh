OLLAMA_CONTAINER=$(docker ps -q --filter "name=ollama")
API_CONTAINER=$(docker ps -q --filter "name=api")
MILVUS_CONTIANER=$(docker ps -q --filter "name=milvus-standalone")

docker stop $OLLAMA_CONTAINER $API_CONTAINER $MILVUS_CONTIANER
docker rm $OLLAMA_CONTAINER $API_CONTAINER $MILVUS_CONTIANER