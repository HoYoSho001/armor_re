#include <iostream>
#include <rclcpp/rclcpp.hpp>
#include <Eigen/Dense>
int x, y;
double compute(double theta, double omega, double t) {
    Eigen::VectorXf X(3);
    Eigen::MatrixXf A(6, 6);
    Eigen::VectorXf B(6);

    B << 0, omega, 0, theta, 0, 0;
    A << 1, t, t*t, t*t*t, t*t*t*t,t*t*t*t*t,
         0, 1, 2*t, 3*t*t, 4*t*t*t, 5*t*t*t*t,
         0, 0, 2, 6*t, 12*t*t, 20*t*t*t;
    X = A.colPivHouseholderQr().solve(B);
    return X(1); //解出的电机速度
}

// ============================================================================
//  与电机（电控）收发数据
// ============================================================================
//  ★ 这一层不碰串口字节。串口的 17 字节收 / 21 字节发 由 serial_node.cpp 负责，
//    它已经把协议翻译成了两条 ROS 消息：
//
//        电控 ──串口17字节──▶ SerialNode ──/gimbal_state──▶ planner
//        电控 ◀──串口21字节── SerialNode ◀──/gimbal_cmd──── planner
//
//    所以在 planner 里，所谓"跟电机收发"就是：
//        收 = 订阅 /gimbal_state   → onMotorData()
//        发 = 发布 /gimbal_cmd     → sendToMotor()
//
//  ★ 为什么不让 planner 自己开串口：
//    同一个 /dev/ttyACM0 只能被一个进程打开。planner 和 serial_node 抢同一个
//    设备的结果是其中一个 open 失败、或者收到半帧乱码。所以串口只能有一处。
//
//  ★ 单位约定（最容易出错的地方）：
//    ROS 消息里【全部是弧度】，度×10⁴ 的换算是 serial_node 做的。
//    所以下面两个函数的参数/成员都是弧度，不要在这里再乘 180/π。
// ============================================================================

#include <cmath>
#include <cstdint>

#include "armor_re/msg/gimbal_cmd.hpp"
#include "armor_re/msg/gimbal_state.hpp"

// ----------------------------------------------------------------------------
//  电机回传的最近一帧状态（接收函数的产物）
// ----------------------------------------------------------------------------
//  对应 auto-aim-master 里 io/src/SerialPort.cpp 解析出来的 Data_Save：
//      17 字节帧：AA BB | 保留 | yaw | pitch | roll | color | mode
//  区别是这里已经换成弧度、并且带了一个 ROS 时间戳。
// ----------------------------------------------------------------------------
struct MotorState
{
  rclcpp::Time stamp{0, 0, RCL_ROS_TIME};   // 这帧从串口读到的时刻，用来和图像做时间对齐
  double yaw   = 0.0;    // 云台 yaw，弧度，范围 (-π, π]
  double pitch = 0.0;    // 云台 pitch，弧度，抬头为正
  double roll  = 0.0;    // 云台 roll，弧度（协议里的 -π 偏移 serial_node 已减掉）
  int    color = 0;      // 敌方颜色：0=蓝 1=红
  int    mode  = 1;      // 1=自瞄 2=打符
  bool   valid = false;  // false = 一帧都还没收到，此时不要用上面的角度
};

// ============================================================================
class PlannerNode : public rclcpp::Node
{
public:
  PlannerNode() : Node("planner_node")
  {
    // 话题名做成参数，方便和 serial_node 用 -p 对齐（默认两边一致）
    const std::string state_topic = declare_parameter<std::string>("state_topic", "gimbal_state");
    const std::string cmd_topic   = declare_parameter<std::string>("cmd_topic", "gimbal_cmd");

    // ---- 接收：电控 → 视觉 ----
    //  QoS: 电控回传一般 100 Hz 以上，队列给 50 吸收抖动；
    //       这里永远只要新数据，所以队列满时丢旧的（keep_last）。
    motor_sub_ = create_subscription<armor_re::msg::GimbalState>(
      state_topic, rclcpp::QoS(50),
      [this](const armor_re::msg::GimbalState::SharedPtr msg) { onMotorData(*msg); });

    // ---- 发送：视觉 → 电控 ----
    motor_pub_ = create_publisher<armor_re::msg::GimbalCmd>(cmd_topic, rclcpp::QoS(10));

    RCLCPP_INFO(get_logger(), "PlannerNode: 收 %s / 发 %s",
                state_topic.c_str(), cmd_topic.c_str());
  }

  // ==========================================================================
  //  ① 接收函数：电控 → 视觉
  // ==========================================================================
  //  入口：/gimbal_state 的订阅回调，每来一帧调用一次（约 100 Hz）
  //  产出：把 latest_motor_ 刷成最新的云台姿态
  //
  //  参数：
  //    msg —— 云台状态。角度【已经是弧度】、帧【已经解好】，
  //           这里不需要再做任何单位换算或字节解析。
  //
  //  ★ 一定要存 stamp。后面拿一帧图像（像素）解算世界坐标时，
  //    必须用【拍这张图那一刻】的云台角。用"当前"云台角的话，
  //    云台一转，世界坐标系就跟着转，EKF 会把云台的运动误判成目标在动。
  // ==========================================================================
  void onMotorData(const armor_re::msg::GimbalState & msg)
  {
    // 时间戳：没有就用收到消息的时刻兜底（精度差一点，但不至于让下游崩）
    const bool no_stamp = (msg.header.stamp.sec == 0 && msg.header.stamp.nanosec == 0);
    latest_motor_.stamp = no_stamp ? now() : rclcpp::Time(msg.header.stamp);

    // 角度：直接存，msg 里已经是弧度
    latest_motor_.yaw   = msg.yaw;
    latest_motor_.pitch = msg.pitch;
    latest_motor_.roll  = msg.roll;   // 协议里的 -π 偏移已在 serial_node 里减掉

    // 模式与颜色
    latest_motor_.color = msg.color;
    latest_motor_.mode  = msg.mode;

    latest_motor_.valid = true;

    ++rx_count_;
    // 第一次收到时打一行，用来确认协议真的对上了（角度量级合理就说明没错）
    if (rx_count_ == 1) {
      RCLCPP_INFO(get_logger(),
                  "收到第一帧电机数据: yaw=%.2f° pitch=%.2f° roll=%.2f° color=%d mode=%d",
                  latest_motor_.yaw * 180.0 / M_PI,
                  latest_motor_.pitch * 180.0 / M_PI,
                  latest_motor_.roll * 180.0 / M_PI,
                  latest_motor_.color, latest_motor_.mode);
    }
  }

  // ==========================================================================
  //  ② 发送函数：视觉 → 电控
  // ==========================================================================
  //  参数（全部【弧度】，不要传度；度×10⁴ 的换算在 serial_node 里做）：
  //
  //    ┌──────────┬────────┬──────────────────────────────────────────────┐
  //    │ 形参      │ 单位    │ 含义                                         │
  //    ├──────────┼────────┼──────────────────────────────────────────────┤
  //    │ yaw      │ rad    │ 云台【控制角】manage_yaw（世界系绝对角）。     │
  //    │          │        │ ★ 传 manage_yaw，不是 target_yaw。反小陀螺时  │
  //    │          │        │   云台要"抢跑"，真正让它转到的就是这个角。     │
  //    │          │        │ 对应串口帧 [2:6]，度×10⁴ 大端 int32。         │
  //    ├──────────┼────────┼──────────────────────────────────────────────┤
  //    │ pitch    │ rad    │ 目标俯仰角，来自弹道解算，抬头为正。           │
  //    │          │        │ ★ 只有一条弹道解，没有速度/加速度前馈 ——      │
  //    │          │        │   俯仰不参与五次多项式规划（规划只做 yaw）。   │
  //    │          │        │ 对应串口帧 [6:10]。                          │
  //    ├──────────┼────────┼──────────────────────────────────────────────┤
  //    │ yaw_vel  │ rad/s  │ 前馈角速度（五次多项式在命中时刻的一阶导）。   │
  //    │          │        │ 电控拿它做速度前馈，PID 少追一点；            │
  //    │          │        │ 不需要就传 0。对应串口帧 [10:14]。           │
  //    ├──────────┼────────┼──────────────────────────────────────────────┤
  //    │ found    │ 0/1    │ 1=识别到目标 0=丢失。对应串口帧 [18]。        │
  //    │          │        │ 电控靠它判断要不要让云台跟视觉走。            │
  //    ├──────────┼────────┼──────────────────────────────────────────────┤
  //    │ fire     │ 0/2    │ 2=允许开火 0=不开火。对应串口帧 [19]。        │
  //    │          │        │ ★ 是 2 不是 1，照抄原工程协议。               │
  //    └──────────┴────────┴──────────────────────────────────────────────┘
  //
  //  不在这两个函数里做的事（都已经被别处承担了）：
  //    · 角度补偿 compensate_yaw / compensate_pitch → serial_node 的参数
  //    · 弧度 → 度×10⁴                              → serial_node 的 onCmd()
  //    · 打包 21 字节帧头帧尾                        → serial_node 的 onCmd()
  //
  //  例：
  //    // 识别到目标、对准了、允许开火
  //    sendToMotor(manage_yaw, target_pitch, yaw_vel, 1, 2);
  //    // 丢失目标：保持上一次的角度不动，不点火
  //    publishHold();
  // ==========================================================================
  void sendToMotor(double yaw, double pitch, double yaw_vel, int found, int fire)
  {
    // ---- 保护：NaN/inf 一旦发下去，电控会算出乱七八糟的角度，把云台直接甩飞 ----
    if (!std::isfinite(yaw) || !std::isfinite(pitch) || !std::isfinite(yaw_vel)) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 1000,
                           "sendToMotor: 参数含 NaN/inf (yaw=%.3f pitch=%.3f vel=%.3f)，本帧拒发",
                           yaw, pitch, yaw_vel);
      return;
    }

    // ---- yaw 归一化到 (-π, π]，避免下发 720° 这种离谱角度 ----
    yaw = std::remainder(yaw, 2.0 * M_PI);

    armor_re::msg::GimbalCmd cmd;
    cmd.yaw     = yaw;                  // 控制角 manage_yaw
    cmd.pitch   = pitch;                // 目标俯仰
    cmd.yaw_vel = yaw_vel;              // 前馈角速度
    cmd.found   = (found != 0) ? 1 : 0;  // 只允许 0/1
    cmd.fire    = (fire == 2) ? 2 : 0;   // 只允许 0/2

    motor_pub_->publish(cmd);
    ++tx_count_;

    // 记一份，丢失目标时可以原地保持
    last_cmd_ = cmd;
    has_cmd_  = true;
  }

  // ==========================================================================
  //  便利接口
  // ==========================================================================
  const MotorState & motor() const { return latest_motor_; }

  // 丢失目标时用：保持上一次的控制角，不点火
  void publishHold()
  {
    if (!has_cmd_) return;              // 还没发过，就没有"上一次"可保持
    sendToMotor(last_cmd_.yaw, last_cmd_.pitch, 0.0, 0, 0);
  }

private:
  // ---- 接收侧 ----
  MotorState latest_motor_;
  rclcpp::Subscription<armor_re::msg::GimbalState>::SharedPtr motor_sub_;
  uint64_t rx_count_ = 0;

  // ---- 发送侧 ----
  rclcpp::Publisher<armor_re::msg::GimbalCmd>::SharedPtr motor_pub_;
  uint64_t tx_count_ = 0;
  armor_re::msg::GimbalCmd last_cmd_;
  bool has_cmd_ = false;
};

// ============================================================================
//  main
// ============================================================================
//  原来的空 main 换成了 ROS 的标准启动。
//  真正接算法的时候，流程是：
//      收到一帧装甲板 → PnP → EKF → 弹道 → 五次多项式 → sendToMotor(...)
//  骨架大致是：
//
//      rclcpp::spin(node);                       // 或 spin_some 放在循环里
//      if (!node->motor().valid) continue;       // 还没收到电机数据，什么都不做
//      double gimbal_yaw = node->motor().yaw;    // ← 时间对齐时要用这一帧的角
//      ... 解算 ...
//      node->sendToMotor(manage_yaw, target_pitch, yaw_vel, 1, 2);
//
// ============================================================================
int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<PlannerNode>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
