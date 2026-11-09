# 电控 → 视觉：云台姿态 + 模式
#
# 对应 auto-aim-master 里串口【下行】17 字节帧解析出的 Data_Save：
#     [3:6] yaw  [7:10] pitch  [11:14] roll   （int32 大端，单位 度×10⁴）
#     [15]  color   [16] mode
# 本 msg 统一用【弧度】表示角度，换算已经在该节点内部完成。
std_msgs/Header header   # stamp = 本帧从串口读到的时刻（用于和图像做时间对齐）

float64 yaw      # 云台 yaw，弧度，已归一化到 (-π, π]
float64 pitch    # 云台 pitch，弧度
float64 roll     # 云台 roll，弧度（原协议里有 -π 偏移，节点内已减掉）
int32 color      # 敌方颜色：0=蓝 1=红
int32 mode       # 1=自瞄  2=打符
