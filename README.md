# Go2Real

## Go2の検出
'''
cd ./Go2Real/
docker compose up -d
docker exec -it go2real-go2_controller-1 python /workspace/networkTest.py
'''

## exampleの実行
'''
cd ./Go2Real/
docker compose up -d
docker exec -it go2real-go2_controller-1 python python /workspace/unitree_sdk2_python/example/go2/low_level/go2_stand_example.py enp2s0
'''

## 学習環境の構築
- 前進機能
'''
cd ./Go2Real/training/
docker compose up -d
docker exec -it training-go2_controller-1 python3 Genesis/examples/locomotion/go2_train.py
'''