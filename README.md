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
docker compose up
'''

## 検証
'''
xhost +local:docker
cd ./Go2Real/training/
docker compose up -d
docker exec -it go2_controller python3 Genesis/examples/locomotion/go2_eval.py -e go2-walking --ckpt 1000
'''
> 本来はGUIが起動するが対応できていない

## Sim2Real
'''
cd ./Go2Real/sim2real/
docker compose up -d
docker exec -it go2real python3 /workspace/sim2real_walk.py
'''