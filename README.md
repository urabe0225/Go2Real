# Go2Real

- [docker](https://docs.docker.com/engine/install/ubuntu/#install-using-the-repository)
- [nvidia container toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html#with-apt-ubuntu-debian)

## Go2の検出
'''
cd ./Go2Real/setup/
docker compose up -d
docker exec -it setup python /workspace/networkTest.py
'''

## exampleの実行
'''
cd ./Go2Real/setup/
docker compose up -d
docker exec -it setup python /workspace/unitree_sdk2_python/example/go2/low_level/go2_stand_example.py enp2s0
'''

## 学習環境の構築
- 前進機能
'''
cd ./Go2Real/training/
docker compose up -d
docker exec -it training-go2_controller-1 python3 Genesis/examples/locomotion/go2_train.py
'''

## 検証
'''
xhost +local:docker
cd ./Go2Real/training/
docker compose up -d
docker exec -it training-go2_controller-1 python3 Genesis/examples/locomotion/go2_eval.py
'''
> 本来はGUIが起動するが対応できていない

## Sim2Real ※未検証
'''
cd ./Go2Real/sim2real/
docker compose up -d
docker exec -it sim2real-go2_controller-1 python3 /workspace/sim2real_walk.py enp2s0
'''