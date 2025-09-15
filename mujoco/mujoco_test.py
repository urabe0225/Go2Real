import mujoco
from mujoco.viewer import launch

# XMLモデルファイルを読み込む
model = mujoco.MjModel.from_xml_path('/opt/mujoco/model/humanoid/humanoid.xml')
data = mujoco.MjData(model)

# ビューアを起動
launch(model, data)
