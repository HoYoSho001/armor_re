# 视觉 → 电控：云台控制指令
#
# 对应 auto-aim-master 里串口【上行】21 字节帧（AutoArmor::Transmit 打包）：
#     AA BB | yaw(int32) | pitch(int32) | yaw_vel(int32) | 0 | found | fire | BB
# 其中三个角度量单位是 度×10⁴（大端 int32）。
# 本 msg 用【弧度 / 弧度每秒】表达，串口节点内部负责
#   ( rad * 180/π + compensate ) * 1e4  → int32 大端
float64 yaw      # 控制角（manage_yaw），弧度，通常会加弹道/安装补偿
float64 pitch    # 目标 pitch，弧度
float64 yaw_vel  # 控制角速度（前馈），弧度/秒
int32 found      # 1=识别到目标  0=丢失
int32 fire       # 2=允许开火   0=不开火
