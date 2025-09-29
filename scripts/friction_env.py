import torch
import math
import genesis as gs
from genesis.utils.geom import quat_to_xyz, transform_by_quat, inv_quat, transform_quat_by_quat
from go2_env import Go2Env

def gs_rand_float(lower, upper, shape, device):
    return (upper - lower) * torch.rand(size=shape, device=device) + lower

class FrictionEnv(Go2Env):
    def __init__(self, num_envs, env_cfg, obs_cfg, reward_cfg, command_cfg, show_viewer=False):
        super().__init__(num_envs, env_cfg, obs_cfg, reward_cfg, command_cfg, show_viewer)

        # 足のリンクのインデックスを取得
        self.foot_links = []
        foot_names = ["FR_foot", "FL_foot", "RR_foot", "RL_foot"]
        for foot_name in foot_names:
            try:
                foot_link = self.robot.get_link(foot_name)
                self.foot_links.append(foot_link)
            except:
                print(f"Warning: Could not find foot link {foot_name}")

        # 摩擦係数の範囲設定
        self.friction_range = env_cfg.get("friction_range", [0.2, 1.5])  # [最小, 最大]
        self.restitution_range = env_cfg.get("restitution_range", [0.0, 0.3])
        
        # 各環境の現在の摩擦係数を記録
        self.current_friction = torch.zeros(num_envs, device=self.device)
        self.current_restitution = torch.zeros(num_envs, device=self.device)

    def _resample_commands(self, envs_idx):
        super()._resample_commands(envs_idx)
    
    def _randomize_friction(self, envs_idx):
        """指定された環境の摩擦係数をランダムに設定"""
        if len(envs_idx) == 0:
            return
            
        # 新しい摩擦係数とrestitutionをランダムに生成
        new_friction = gs_rand_float(
            self.friction_range[0], self.friction_range[1], 
            (len(envs_idx),), self.device
        )
        new_restitution = gs_rand_float(
            self.restitution_range[0], self.restitution_range[1],
            (len(envs_idx),), self.device
        )
    
        # 現在の値を更新
        self.current_friction[envs_idx] = new_friction
        self.current_restitution[envs_idx] = new_restitution
    
        # 地面と足の摩擦係数を設定
        for i, env_idx in enumerate(envs_idx):
            env_idx_val = env_idx.item() if torch.is_tensor(env_idx) else env_idx
            
            friction_val = new_friction[i].item() if torch.is_tensor(new_friction[i]) else new_friction[i]
            restitution_val = new_restitution[i].item() if torch.is_tensor(new_restitution[i]) else new_restitution[i]
            
            # 地面の物理パラメータを設定
            self.plane.set_friction(friction_val)#(friction_val, [env_idx_val])
            #self.plane.set_restitution(restitution_val, envs_idx=[env_idx_val])
            
            # 足の物理パラメータを設定
            for foot_link in self.foot_links:
                foot_link.set_friction(friction_val, envs_idx=[env_idx_val])
                foot_link.set_restitution(restitution_val, envs_idx=[env_idx_val])

    def step(self, actions):
        # 親クラスのstepメソッドを呼び出す前に、摩擦係数の定期的な変更を処理
        # episode_length_bufを更新するために一時的に処理
        temp_episode_length = getattr(self, 'episode_length_buf', torch.zeros((self.num_envs,), device=self.device, dtype=gs.tc_int))
        temp_episode_length += 1
        
        # 摩擦係数の定期的な変更（オプション）
        friction_resample_time = self.env_cfg.get("friction_resample_time_s", 8.0)  # 8秒ごと
        friction_envs_idx = (
            (temp_episode_length % int(friction_resample_time / self.dt) == 0)
            .nonzero(as_tuple=False)
            .reshape((-1,))
        )
        if len(friction_envs_idx) > 0:
            self._randomize_friction(friction_envs_idx)

        # 親クラスのstepメソッドを呼び出し
        return super().step(actions)

    def reset_idx(self, envs_idx):
        super().reset_idx(envs_idx)
        # リセット時に摩擦係数もランダマイズ
        self._randomize_friction(envs_idx)

    def reset(self):
        result = super().reset()
        # リセット時に全環境の摩擦係数をランダマイズ
        self._randomize_friction(torch.arange(self.num_envs, device=self.device))
        return result

    # ------------ reward functions----------------
    def _reward_tracking_lin_vel(self):
        return super()._reward_tracking_lin_vel()

    def _reward_tracking_ang_vel(self):
        return super()._reward_tracking_ang_vel()

    def _reward_lin_vel_z(self):
        return super()._reward_lin_vel_z()

    def _reward_action_rate(self):
        return super()._reward_action_rate()

    def _reward_similar_to_default(self):
        return super()._reward_similar_to_default()

    def _reward_base_height(self):
        return super()._reward_base_height()
