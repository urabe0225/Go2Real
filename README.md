# Go2Real

- [docker](https://docs.docker.com/engine/install/ubuntu/#install-using-the-repository)
- [nvidia container toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html#with-apt-ubuntu-debian)

## GenesisのExampleを(前進機能)

### Go2の検出
- Go2を起動しイーサネットケーブルで接続
```
cd ./Go2Real/setup/
docker compose up -d
docker exec -it setup python /workspace/networkTest.py
```

### 初期姿勢コマンドの実行
- Go2を起動しイーサネットケーブルで接続
```
cd ./Go2Real/setup/
docker compose up -d
docker exec -it setup python /workspace/unitree_sdk2_python/example/go2/low_level/go2_stand_example.py enp2s0
```

### Trainingの実行
```
cd ./Go2Real/training/
docker compose up
docker exec -it go2_controller python3 Genesis/examples/locomotion/go2_train.py --exp_name go2-walking --num_envs 4096 --max_iterations 1001 &
```

### Training結果の検証
```
xhost +local:docker
cd ./Go2Real/eval/
docker compose up -d
docker exec -it go2_controller python3 Genesis/examples/locomotion/go2_eval.py --exp_name go2-walking --ckpt 1000
```

### 実機での検証
- Go2を起動しイーサネットケーブルで接続
```
cd ./Go2Real/sim2real/
docker compose up -d
docker exec -it go2real python3 /workspace/sim2real_walk.py --exp_name go2-walking --ckpt 1000 --net enp2s0
```

## GenesisのExampleを改良し床の反発係数を考慮した環境で学習(前進機能)

### Go2の検出
> 省略

### 初期姿勢コマンドの実行
> 省略

### Trainingの実行
```
cd ./Go2Real/training/
docker compose up
docker exec -it go2_controller python3 Genesis/examples/locomotion/friction_train.py --exp_name go2-friction --num_envs 4096 --max_iterations 1001 &
```

### Training結果の検証
```
xhost +local:docker
cd ./Go2Real/eval/
docker compose up -d
docker exec -it go2_controller python3 Genesis/examples/locomotion/go2_eval.py --exp_name go2-friction --ckpt 1000
```

### 実機での検証
- Go2を起動しイーサネットケーブルで接続
```
cd ./Go2Real/sim2real/
docker compose up -d
docker exec -it go2real python3 /workspace/sim2real_walk.py --exp_name go2-friction --ckpt 1000 --net enp2s0
```

