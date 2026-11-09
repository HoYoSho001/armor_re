// ============================================================================
//  serial_node.cpp —— 串口节点（SerialNode）
//
//  职责：只做"电控 ↔ 视觉"的协议搬运，不做任何算法。
//        等价于 auto-aim-master 里 io/src/SerialPort.cpp + AutoArmor::Transmit 的打包部分。
//
//  数据流（两个方向）：
//
//    ┌──────── 下行：收 ────────┐
//    │ 电控 发送 17 字节帧        │
//    │  AA BB | ? | yaw | pitch | roll | color | mode │
//    │        → 本节点解析        │
//    │        → 发布 gimbal_state (armor_re/msg/GimbalState) │
//    └──────────────────────────┘
//
//    ┌──────── 上行：发 ────────┐
//    │ 订阅 gimbal_cmd (armor_re/msg/GimbalCmd)            │
//    │        → 本节点打包 21 字节帧 → write() → 电控       │
//    │  AA BB | yaw | pitch | yaw_vel | 0 | found | fire | BB │
//    └──────────────────────────┘
//
//  ★ 协议细节（与 auto-aim-master 完全一致）：
//    · 角度量：int32 大端，单位 = 度 × 10⁴（即 0.0001° 分辨率）
//    · 下行帧 17 字节，上行帧 21 字节，两个方向不对称
//    · 上行 [18]=found(1/0)  [19]=fire(2=开火/0=不开火)  [20]=0xBB 帧尾
//
//  ★ 为什么上行帧里 yaw 用的是"控制角"而不是"目标角"：
//    这是原工程反小陀螺的约定 —— 规划器给出平滑后的 manage_yaw 作为控制量，
//    而 target_yaw 只用于规划器内部判开火。所以 GimbalCmd.yaw 应当传控制角。
//
//  需要的库（CMakeLists.txt 已配好）：
//    rclcpp                          —— 节点
//    armor_re::msg::GimbalState/Cmd  —— 自定义消息
//    termios / poll / write          —— 系统调用，【不需要链接额外库】
//
//  运行示例：
//    ros2 run armor_re SerialNode
//    ros2 run armor_re SerialNode --ros-args -p port:=/dev/ttyUSB0 -p compensate_yaw:=0.5
// ============================================================================

#include <atomic>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include <fcntl.h>
#include <poll.h>
#include <termios.h>
#include <unistd.h>

#include <rclcpp/rclcpp.hpp>
#include "armor_re/msg/gimbal_state.hpp"
#include "armor_re/msg/gimbal_cmd.hpp"

namespace
{
// ---- 协议常量（与 auto-aim-master 保持一致）----
constexpr uint8_t  kHead0      = 0xAA;
constexpr uint8_t  kHead1      = 0xBB;
constexpr uint8_t  kTail       = 0xBB;
constexpr int      kRxFrameLen = 17;      // 下行帧长
constexpr int      kTxFrameLen = 21;      // 上行帧长

// 角度换算：int32(度×10⁴) ↔ 弧度
constexpr double kDeg1e4ToRad = M_PI / 180.0 / 10000.0;
constexpr double kRadToDeg1e4 = 180.0 / M_PI * 10000.0;

// 大端读写
inline int32_t be32(const uint8_t * p)
{
  return static_cast<int32_t>(
    (static_cast<uint32_t>(p[0]) << 24) | (static_cast<uint32_t>(p[1]) << 16) |
    (static_cast<uint32_t>(p[2]) <<  8) |  static_cast<uint32_t>(p[3]));
}
inline void putBe32(uint8_t * p, int32_t v)
{
  p[0] = static_cast<uint8_t>((v >> 24) & 0xFF);
  p[1] = static_cast<uint8_t>((v >> 16) & 0xFF);
  p[2] = static_cast<uint8_t>((v >>  8) & 0xFF);
  p[3] = static_cast<uint8_t>((v >>  0) & 0xFF);
}
}  // namespace

// ============================================================================
class SerialNode : public rclcpp::Node
{
public:
  SerialNode() : Node("serial_node")
  {
    port_             = declare_parameter<std::string>("port", "/dev/ttyACM0");
    baud_             = declare_parameter<int>("baud", 115200);
    state_topic_      = declare_parameter<std::string>("state_topic", "gimbal_state");
    cmd_topic_        = declare_parameter<std::string>("cmd_topic", "gimbal_cmd");
    compensate_yaw_   = declare_parameter<double>("compensate_yaw", 0.0);     // 度
    compensate_pitch_ = declare_parameter<double>("compensate_pitch", 0.0);   // 度
    publish_state_    = declare_parameter<bool>("publish_state", true);

    // ---- 发布：电控下发的云台状态 ----
    state_pub_ = create_publisher<armor_re::msg::GimbalState>(state_topic_, 10);

    // ---- 订阅：要发给电控的控制指令 ----
    cmd_sub_ = create_subscription<armor_re::msg::GimbalCmd>(
      cmd_topic_, 10,
      [this](const armor_re::msg::GimbalCmd::SharedPtr msg) { onCmd(*msg); });

    RCLCPP_INFO(get_logger(), "串口 %s @ %d, 发布 %s, 订阅 %s, yaw补偿 %.2f° pitch补偿 %.2f°",
                port_.c_str(), baud_, state_topic_.c_str(), cmd_topic_.c_str(),
                compensate_yaw_, compensate_pitch_);

    if (!openPort()) {
      RCLCPP_ERROR(get_logger(), "打开串口失败: %s", port_.c_str());
      throw std::runtime_error("serial open failed");
    }

    // 读线程：阻塞在 poll 上，收到数据就解析并发布
    read_thread_ = std::thread([this] { readLoop(); });
  }

  ~SerialNode() override
  {
    quit_ = true;
    if (read_thread_.joinable()) read_thread_.join();
    std::lock_guard<std::mutex> lk(fd_mutex_);
    if (fd_ >= 0) { ::close(fd_); fd_ = -1; }
    RCLCPP_INFO(get_logger(), "SerialNode 已退出：收到 %lu 帧, 发出 %lu 帧",
                static_cast<unsigned long>(rx_count_.load()),
                static_cast<unsigned long>(tx_count_.load()));
  }

private:
  // ==========================================================================
  //  打开 / 重开串口
  // ==========================================================================
  bool openPort()
  {
    std::lock_guard<std::mutex> lk(fd_mutex_);

    if (fd_ >= 0) { ::close(fd_); fd_ = -1; }

    // O_NOCTTY: 不把串口当控制终端；O_NONBLOCK 先开起来方便配置
    fd_ = ::open(port_.c_str(), O_RDWR | O_NOCTTY | O_NONBLOCK);
    if (fd_ < 0) return false;

    // 配置 8N1、raw 模式、无软硬流控
    termios tio{};
    if (tcgetattr(fd_, &tio) < 0) { ::close(fd_); fd_ = -1; return false; }

    speed_t spd = B115200;
    switch (baud_) {
      case 9600:    spd = B9600;    break;
      case 19200:   spd = B19200;   break;
      case 38400:   spd = B38400;   break;
      case 57600:   spd = B57600;   break;
      case 115200:  spd = B115200;  break;
      case 230400:  spd = B230400;  break;
      case 460800:  spd = B460800;  break;
      case 921600:  spd = B921600;  break;
      default:
        RCLCPP_WARN(get_logger(), "不支持的波特率 %d，回退 115200", baud_);
        break;
    }
    cfsetispeed(&tio, spd);
    cfsetospeed(&tio, spd);

    tio.c_cflag &= ~PARENB;          // 无校验
    tio.c_cflag &= ~CSTOPB;          // 1 位停止位
    tio.c_cflag &= ~CSIZE;
    tio.c_cflag |=  CS8;             // 8 数据位
#ifdef CRTSCTS
    tio.c_cflag &= ~CRTSCTS;         // 无硬件流控
#endif
    tio.c_cflag |= (CLOCAL | CREAD); // 忽略调制解调器线、使能接收

    tio.c_lflag &= ~(ICANON | ECHO | ECHOE | ISIG);   // raw 模式
    tio.c_iflag &= ~(IXON | IXOFF | IXANY);           // 无软件流控
    tio.c_iflag &= ~(INLCR | ICRNL | IGNCR);          // 别动 \r\n（二进制协议必须）
    tio.c_oflag &= ~OPOST;                            // 不做输出处理
    tio.c_cc[VMIN]  = 0;                              // 配合 poll 使用：不阻塞等足字节
    tio.c_cc[VTIME] = 0;

    if (tcsetattr(fd_, TCSANOW, &tio) < 0) { ::close(fd_); fd_ = -1; return false; }
    tcflush(fd_, TCIOFLUSH);

    // 恢复阻塞语义：写操作不要因为缓冲区满而立刻失败
    int flags = fcntl(fd_, F_GETFL, 0);
    fcntl(fd_, F_SETFL, flags & ~O_NONBLOCK);

    rx_buf_.clear();      // 重开时清掉残留字节，避免半帧污染
    return true;
  }

  // ==========================================================================
  //  读线程：poll → read → 拆包 → 发布
  // ==========================================================================
  void readLoop()
  {
    using namespace std::chrono_literals;
    int err_count = 0;

    while (rclcpp::ok() && !quit_) {
      int fd;
      {
        std::lock_guard<std::mutex> lk(fd_mutex_);
        fd = fd_;
      }
      if (fd < 0) { std::this_thread::sleep_for(100ms); continue; }

      // ★ poll 超时 20ms：既能及时响应，又不会像原工程那样 timeout=0 把 CPU 跑满
      pollfd pfd{fd, POLLIN, 0};
      int pr = ::poll(&pfd, 1, 20 /*ms*/);
      if (pr < 0) {
        if (errno == EINTR) continue;               // 被信号打断，正常
        RCLCPP_WARN(get_logger(), "poll 失败: %s", std::strerror(errno));
        ++err_count;
      } else if (pr == 0) {
        continue;                                    // 超时，没有数据
      } else if (pfd.revents & (POLLERR | POLLHUP | POLLNVAL)) {
        RCLCPP_WARN(get_logger(), "串口异常 (revents=0x%x)，尝试重连", pfd.revents);
        ++err_count;
      } else if (pfd.revents & POLLIN) {
        uint8_t chunk[128];
        ssize_t n = ::read(fd, chunk, sizeof(chunk));
        if (n > 0) {
          err_count = 0;
          rx_buf_.insert(rx_buf_.end(), chunk, chunk + n);
          parseBuffer();                             // 拆包 + 发布
        } else if (n < 0 && errno != EAGAIN && errno != EWOULDBLOCK && errno != EINTR) {
          RCLCPP_WARN(get_logger(), "read 失败: %s", std::strerror(errno));
          ++err_count;
        }
      }

      // 连续出错 → 重连（对应原工程的 restart_serial）
      if (err_count > 5) {
        RCLCPP_WARN(get_logger(), "连续 %d 次串口错误，重连…", err_count);
        if (!openPort()) std::this_thread::sleep_for(200ms);
        err_count = 0;
        continue;
      }

      // 防御：如果长期收不到合法帧，缓冲区不能无限增长
      if (rx_buf_.size() > 4096) {
        RCLCPP_WARN(get_logger(), "接收缓冲区异常增长(%zu)，清空重来", rx_buf_.size());
        rx_buf_.clear();
        ++bad_frame_count_;
      }
    }
  }

  // 在 rx_buf_ 里找 AA BB 帧头，凑够 17 字节就解析
  void parseBuffer()
  {
    while (rx_buf_.size() >= static_cast<size_t>(kRxFrameLen)) {
      // ---- 1. 定位帧头 ----
      size_t head = rx_buf_.size();                    // 找不到时用作哨兵
      for (size_t i = 0; i + 1 < rx_buf_.size(); ++i) {
        if (rx_buf_[i] == kHead0 && rx_buf_[i + 1] == kHead1) { head = i; break; }
      }
      if (head == rx_buf_.size()) {
        // 没有帧头：只保留最后 1 个字节（可能是被截断的 0xAA）
        rx_buf_.erase(rx_buf_.begin(), rx_buf_.end() - 1);
        ++bad_frame_count_;
        return;
      }
      if (head > 0) {                                   // 丢掉帧头之前的垃圾
        rx_buf_.erase(rx_buf_.begin(), rx_buf_.begin() + head);
        if (rx_buf_.size() < static_cast<size_t>(kRxFrameLen)) return;
      }

      // ---- 2. 取出完整一帧 ----
      const uint8_t * f = rx_buf_.data();

      // 帧格式（17 字节）：
      //   [0]=0xAA [1]=0xBB [2]=保留
      //   [3:6]=yaw [7:10]=pitch [11:14]=roll  （int32 大端，度×10⁴）
      //   [15]=敌方颜色 [16]=模式
      const int32_t yaw_i   = be32(f + 3);
      const int32_t pitch_i = be32(f + 7);
      const int32_t roll_i  = be32(f + 11);

      armor_re::msg::GimbalState st;
      st.header.stamp = now();
      st.yaw   = static_cast<double>(yaw_i)   * kDeg1e4ToRad;
      st.pitch = static_cast<double>(pitch_i) * kDeg1e4ToRad;
      // ★ 原协议里 roll 带了 -π 的偏移，这里还原回去（与原工程一致）
      st.roll  = static_cast<double>(roll_i)  * kDeg1e4ToRad - M_PI;
      st.color = f[15];
      st.mode  = f[16];

      if (publish_state_) state_pub_->publish(st);
      ++rx_count_;

      rx_buf_.erase(rx_buf_.begin(), rx_buf_.begin() + kRxFrameLen);
    }
  }

  // ==========================================================================
  //  写：收到 GimbalCmd → 打包 21 字节 → write
  // ==========================================================================
  void onCmd(const armor_re::msg::GimbalCmd & msg)
  {
    // 弧度 → 度，加补偿，再 ×1e4 转 int32
    const int32_t yaw_i   = static_cast<int32_t>(
      (msg.yaw         * 180.0 / M_PI + compensate_yaw_)   * 10000.0);
    const int32_t pitch_i = static_cast<int32_t>(
      (msg.pitch       * 180.0 / M_PI + compensate_pitch_) * 10000.0);
    const int32_t vel_i   = static_cast<int32_t>(
      (msg.yaw_vel     * 180.0 / M_PI)                     * 10000.0);

    uint8_t frame[kTxFrameLen];
    frame[0] = kHead0;
    frame[1] = kHead1;
    putBe32(frame +  2, yaw_i);        // 控制角 manage_yaw
    putBe32(frame +  6, pitch_i);      // 目标 pitch
    putBe32(frame + 10, vel_i);        // 前馈角速度
    putBe32(frame + 14, 0);            // 保留
    frame[18] = static_cast<uint8_t>(msg.found ? 1 : 0);
    frame[19] = static_cast<uint8_t>(msg.fire);
    frame[20] = kTail;

    std::lock_guard<std::mutex> lk(fd_mutex_);
    if (fd_ < 0) return;

    ssize_t n = ::write(fd_, frame, kTxFrameLen);
    if (n != kTxFrameLen) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 1000,
                           "write 失败(%zd/%d): %s", n, kTxFrameLen, std::strerror(errno));
      // 写失败通常意味着设备掉线 → 交给读线程的 err_count 机制触发重连
      ::close(fd_);
      fd_ = -1;
      return;
    }
    ++tx_count_;
  }

private:
  // ---- 参数 ----
  std::string port_;
  std::string state_topic_;
  std::string cmd_topic_;
  int    baud_             = 115200;
  double compensate_yaw_   = 0.0;
  double compensate_pitch_ = 0.0;
  bool   publish_state_    = true;

  // ---- ROS ----
  rclcpp::Publisher<armor_re::msg::GimbalState>::SharedPtr state_pub_;
  rclcpp::Subscription<armor_re::msg::GimbalCmd>::SharedPtr cmd_sub_;

  // ---- 串口 ----
  int fd_ = -1;
  mutable std::mutex fd_mutex_;      // 保护 fd_：读线程 poll/read 与写回调会并发
  std::vector<uint8_t> rx_buf_;      // 收包缓冲（处理粘包/半包）
  std::thread read_thread_;
  std::atomic<bool> quit_{false};
  std::atomic<unsigned long> rx_count_{0};
  std::atomic<unsigned long> tx_count_{0};
  unsigned long bad_frame_count_ = 0;
};

// ============================================================================
int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);

  try {
    auto node = std::make_shared<SerialNode>();
    rclcpp::spin(node);       // 单线程 executor：写回调在 spin 里，读线程独立
  } catch (const std::exception & e) {
    RCLCPP_ERROR(rclcpp::get_logger("serial_node"), "启动失败: %s", e.what());
    rclcpp::shutdown();
    return 1;
  }

  rclcpp::shutdown();
  return 0;
}
