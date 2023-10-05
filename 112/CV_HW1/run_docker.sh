
xhost +local:root

XAUTH=/tmp/.docker.xauth

# docker run -it \
#     --name=ntut_cv_hw1 \
#     -d \
#     --rm \
#     --env="DISPLAY=unix$DISPLAY" \
#     --env="QT_X11_NO_MITSHM=1" \
#     --env="XAUTHORITY=$XAUTH" \
#     --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
#     --volume="$XAUTH:$XAUTH" \
#     --volume="$(pwd):/app/" \
#     --net=host \
#     --privileged \
#     --gpus all \
#     tokohsun/opencv:4.8.1

docker run -it \
    --name=ntut_cv_hw1 \
    -d \
    --rm \
    --env="DISPLAY=unix$DISPLAY" \
    --env="QT_X11_NO_MITSHM=1" \
    --env="XAUTHORITY=$XAUTH" \
    --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
    --volume="$XAUTH:$XAUTH" \
    --volume="$(pwd):/app/" \
    --net=host \
    --privileged \
    tokohsun/opencv:4.8.1 

echo "Done!!"