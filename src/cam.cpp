// ============================================================================
//  cam.cpp —— 相机节点（CameraNode），海康 MVS 专用
//
//  职责：把相机帧取出来，发布成 ROS2 标准图像话题。不做任何识别。
//
//      CameraNode ──/image_raw (sensor_msgs/msg/Image, bgr8)──► ArmorRe
//
//  ★ 只支持海康 USB3 工业相机（MV-CS016-10UC 等），走 MVS SDK。
//    早期版本还有一个"读 avi 视频文件"的降级分支，已经删掉：
//    视频回放该由检测节点自己做，不该混进相机节点 ——
//    混在一起的代价是相机节点里到处是 using_mvs_ 判断，
//    而且真实相机和视频文件的时间戳、帧率语义完全不同。
//
//  ★ 依赖（CMakeLists.txt 里已是硬性依赖，找不到直接编不过）
//    rclcpp / sensor_msgs / cv_bridge   —— ROS 侧
//    OpenCV                             —— cv::Mat、Bayer→BGR
//    MVS SDK                            —— MV_CC_* 采集（/opt/MVS）
//    libusb                             —— 相机僵死时的 USB 硬复位
//
//  ★ 参数
//    topic             发布话题名                    默认 image_raw
//    frame_id          图像 frame_id                 默认 camera_optical_frame
//    exposure_us       曝光时间（微秒，手动）        默认 2000
//    gain             增益（dB，auto_gain=false 时生效） 默认 0
//    frame_rate        目标帧率（fps）               默认 165
//    auto_gain         自动增益                      默认 true
//    auto_white_balance 自动白平衡                   默认 true
//    serial_number     指定相机序列号（空=取第一台）  默认 ""
//    usb_vid/usb_pid   仅用于 libusb 复位            默认 0x2bdf/0x0001
//
//  运行：
//    ros2 run armor_re CameraNode
//    ros2 run armor_re CameraNode --ros-args -p exposure_us:=10000.0 -p auto_gain:=false
//    ros2 run armor_re CameraNode --ros-args -p serial_number:=DA4784466
// ============================================================================

#include <atomic>
#include <chrono>
#include <cstring>
#include <memory>
#include <string>
#include <thread>

#include <opencv2/opencv.hpp>

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <cv_bridge/cv_bridge.hpp>

#include "MvCameraControl.h"            // 海康 MVS SDK
#include <libusb-1.0/libusb.h>          // 只用于相机掉线后的 USB 硬复位

namespace
{
using namespace std::chrono_literals;

// 取一帧的超时。165 fps 下一帧约 6 ms，100 ms 足够判定"卡住了"。
constexpr unsigned kGrabTimeoutMs = 100;
}  // namespace

// ============================================================================
class CameraNode : public rclcpp::Node
{
public:

  CameraNode() : Node("camera_node")
  {
    // ---------------- 参数 ----------------
    exposure_us_ = declare_parameter<double>("exposure_us", 2000.0);
    gain_        = declare_parameter<double>("gain", 0.0);
    frame_rate_  = declare_parameter<double>("frame_rate", 165.0);
    auto_gain_   = declare_parameter<bool>("auto_gain", true);
    auto_wb_     = declare_parameter<bool>("auto_white_balance", true);
    serial_no_   = declare_parameter<std::string>("serial_number", "");
    frame_id_    = declare_parameter<std::string>("frame_id", "camera_optical_frame");
    topic_       = declare_parameter<std::string>("topic", "image_raw");
    vid_         = declare_parameter<int>("usb_vid", 0x2bdf);
    pid_         = declare_parameter<int>("usb_pid", 0x0001);

    // ---------------- 发布者 ----------------
    // SensorDataQoS = BestEffort + 深度 5：
    // 图像是高频大流量数据（1440×1080 bgr8 一帧 4.4 MB，165 fps ≈ 730 MB/s），
    // 丢一两帧无所谓；用 Reliable 反而会积压延迟。
    pub_ = create_publisher<sensor_msgs::msg::Image>(topic_, rclcpp::SensorDataQoS().keep_last(5));
    RCLCPP_INFO(get_logger(), "发布话题: %s", topic_.c_str());

    // ---------------- 打开相机 ----------------
    if (!mvsStart()) {
      mvsClose();
      throw std::runtime_error("打开海康相机失败（检查 USB3 口 / udev 规则 / 是否被 MVS 客户端占用）");
    }

    // ---------------- 启动采集线程 ----------------
    // 为什么用独立线程而不是 timer：MV_CC_GetImageBuffer 是【阻塞】调用，
    // 放进 timer 回调会把整个 executor 卡住；rclcpp 的 publisher 本身线程安全，
    // 所以在采集线程里直接 publish 是标准做法。
    capture_thread_ = std::thread([this] { captureLoop(); });
  }

  ~CameraNode() override
  {
    quit_ = true;
    if (capture_thread_.joinable()) capture_thread_.join();
    mvsClose();
    RCLCPP_INFO(get_logger(), "CameraNode 已退出，共发布 %lu 帧",
                static_cast<unsigned long>(frame_count_.load()));
  }

private:
  // ==========================================================================
  //  采集主循环
  // ==========================================================================
  void captureLoop()
  {
    using clock = std::chrono::steady_clock;
    auto last_report = clock::now();
    unsigned long last_frames = 0;
    
    cv::VideoWriter writer("test.avi", cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), 30, cv::Size(1440, 1080));
    while (rclcpp::ok() && !quit_) {
      cv::Mat frame;  
      rclcpp::Time stamp;

      if (!grabMvs(frame, stamp)) {
        if (quit_ || !rclcpp::ok()) break;

        // 相机掉线 / 僵死 → 收尾 → USB 硬复位 → 重新打开
        RCLCPP_WARN(get_logger(), "取帧失败，尝试恢复相机…");
        std::this_thread::sleep_for(100ms);
        if (!recoverMvs()) {
          RCLCPP_ERROR(get_logger(), "相机恢复失败，退出采集");
          break;
        }
        continue;
      }

      cv::imshow("frame", frame);
      cv::waitKey(5);
      writer.write(frame);
      std::cout<<frame.size()<<std::endl;
      // ---- cv::Mat → sensor_msgs/msg/Image ----
      // 统一发 "bgr8"：下游 cv_bridge::toCvShare(msg, "bgr8") 直接拿到 BGR。
      auto msg = cv_bridge::CvImage(std_msgs::msg::Header(), "bgr8", frame).toImageMsg();
      msg->header.stamp    = stamp;
      msg->header.frame_id = frame_id_;
      pub_->publish(*msg);
      ++frame_count_;

      // 每 5 秒报一次实测帧率：曝光时间会卡住帧率上限，这个数字最直观
      const auto now_t = clock::now();
      if (now_t - last_report >= std::chrono::seconds(5)) {
        const double secs = std::chrono::duration<double>(now_t - last_report).count();
        RCLCPP_INFO(get_logger(), "实测 %.1f fps（已发 %lu 帧）",
                    static_cast<double>(frame_count_.load() - last_frames) / secs,
                    static_cast<unsigned long>(frame_count_.load()));
        last_report = now_t;
        last_frames = frame_count_.load();
      }
    }
  }

  // ==========================================================================
  //  打开 + 开始取流：枚举 → 选设备 → 建句柄 → 打开 → 设参数 → StartGrabbing
  // ==========================================================================
  bool mvsStart()
  {
    MV_CC_DEVICE_INFO_LIST list;
    std::memset(&list, 0, sizeof(list));
    if (MV_CC_EnumDevices(MV_USB_DEVICE, &list) != MV_OK) {
      RCLCPP_ERROR(get_logger(), "MV_CC_EnumDevices 失败");
      return false;
    }
    if (list.nDeviceNum == 0) {
      RCLCPP_ERROR(get_logger(), "没有找到 USB 相机");
      return false;
    }

    // ---- 选设备：serial_number 为空取第一台，否则按序列号精确匹配 ----
    // ★ handle_ 一旦创建就代表这一台相机，之后所有 MV_CC_* 都作用在它身上。
    //MV_CC_DEVICE_INFO * target = nullptr;
    MV_CC_DEVICE_INFO * target = list.pDeviceInfo[0];
    for (unsigned i = 0; i < list.nDeviceNum; ++i) {
      MV_CC_DEVICE_INFO * info = list.pDeviceInfo[i];
      if (!info || info->nTLayerType != MV_USB_DEVICE) continue;

      const auto & usb = info->SpecialInfo.stUsb3VInfo;
      const std::string sn(reinterpret_cast<const char *>(usb.chSerialNumber));

      if (serial_no_.empty() || sn == serial_no_) {
        target = info;
        RCLCPP_INFO(get_logger(), "选中相机: %s %s  序列号=%s",
                    reinterpret_cast<const char *>(usb.chManufacturerName),
                    reinterpret_cast<const char *>(usb.chModelName), sn.c_str());
        break;
      }
      RCLCPP_INFO(get_logger(), "跳过相机（序列号 %s ≠ 目标 %s）", sn.c_str(), serial_no_.c_str());
    }
    if (target == nullptr) {
      RCLCPP_ERROR(get_logger(), "枚举到 %u 台设备，但没有 %s",
                   list.nDeviceNum, serial_no_.c_str());
      return false;
    }

    if (MV_CC_CreateHandle(&handle_, target) != MV_OK) {
      RCLCPP_ERROR(get_logger(), "MV_CC_CreateHandle 失败");
      handle_ = nullptr;
      return false;
    }
    if (MV_CC_OpenDevice(handle_) != MV_OK) {
      RCLCPP_ERROR(get_logger(), "MV_CC_OpenDevice 失败（相机可能被 MVS 客户端占用）");
      return false;
    }

    // ---- 相机参数（与原工程同一套策略）----
    //   曝光【手动固定】→ 灯条亮度稳定 → 检测端固定阈值才可靠
    //   增益【自动】    → 由相机吸收整体明暗变化
    setEnum("BalanceWhiteAuto",
            auto_wb_ ? MV_BALANCEWHITE_AUTO_CONTINUOUS : MV_BALANCEWHITE_AUTO_OFF);
    setEnum("ExposureAuto", MV_EXPOSURE_AUTO_MODE_OFF);
    setEnum("GainAuto", auto_gain_ ? MV_GAIN_MODE_CONTINUOUS : MV_GAIN_MODE_OFF);
    setFloat("ExposureTime", exposure_us_);
    if (!auto_gain_) setFloat("Gain", gain_);
    if (frame_rate_ > 0.0) MV_CC_SetFrameRate(handle_, static_cast<float>(frame_rate_));

    if (MV_CC_StartGrabbing(handle_) != MV_OK) {
      RCLCPP_ERROR(get_logger(), "MV_CC_StartGrabbing 失败");
      return false;
    }
    mvs_ready_ = true;
    RCLCPP_INFO(get_logger(), "MVS 相机已启动（曝光 %.0f us，目标帧率 %.0f fps）",
                exposure_us_, frame_rate_);
    return true;
  }

  // ==========================================================================
  //  取一帧：GetImageBuffer → 时间戳 → 转 BGR → FreeImageBuffer
  // ==========================================================================
  bool grabMvs(cv::Mat & out, rclcpp::Time & stamp)
  {
    if (!mvs_ready_) return false;

    MV_FRAME_OUT raw;
    std::memset(&raw, 0, sizeof(raw));
    if (MV_CC_GetImageBuffer(handle_, &raw, kGrabTimeoutMs) != MV_OK) return false;

    stamp = now();                       // ★ 越接近取帧时刻越好
    const auto & info = raw.stFrameInfo;

    // cv::Mat 只是 SDK 缓冲区的视图，不拥有像素 —— 下面的 cvtColor 顺便完成拷贝，
    // 所以必须在 MV_CC_FreeImageBuffer 之前做完转换。
    cv::Mat view(cv::Size(static_cast<int>(info.nWidth), static_cast<int>(info.nHeight)),
                 CV_8U, raw.pBufAddr);

    // ★ 关于 Bayer 那几个 code，别再"顺手改成 2BGR"：
    //   OpenCV 的 Bayer 枚举是按【马赛克排布 + 输出通道序】成对命名的，
    //   COLOR_BayerRG2RGB 输出的确实是 BGR 排列（和 COLOR_BayerBG2BGR 同一个枚举值），
    //   所以下面这套写法解出来的 Mat 就是 channel0=B、channel2=R，
    //   与这里发布的 "bgr8" 以及原工程按 channels[0]=B 取蓝通道的约定一致。
    //   实测：给一幅"RG 排布、只有 R 位有值"的马赛克，BayerRG2RGB 把值解到下标2(R=200)，
    //   而 BayerRG2BGR 解到下标0 → 后者会把红蓝换掉。
    switch (info.enPixelType) {
      case PixelType_Gvsp_BayerGR8: cv::cvtColor(view, out, cv::COLOR_BayerGR2RGB); break;
      case PixelType_Gvsp_BayerRG8: cv::cvtColor(view, out, cv::COLOR_BayerRG2RGB); break;
      case PixelType_Gvsp_BayerGB8: cv::cvtColor(view, out, cv::COLOR_BayerGB2RGB); break;
      case PixelType_Gvsp_BayerBG8: cv::cvtColor(view, out, cv::COLOR_BayerBG2RGB); break;
      case PixelType_Gvsp_Mono8:    cv::cvtColor(view, out, cv::COLOR_GRAY2BGR);    break;
      case PixelType_Gvsp_BGR8_Packed: out = view.clone(); break;   // 相机已直出 BGR
      case PixelType_Gvsp_RGB8_Packed: cv::cvtColor(view, out, cv::COLOR_RGB2BGR); break;
      default: out = view.clone(); break;                            // 未知格式：原样拷贝，别崩
    }

    // 第一帧打一行真实格式，用来核对"相机到底出的什么像素格式"
    if (!first_frame_logged_) {
      first_frame_logged_ = true;
      RCLCPP_INFO(get_logger(), "首帧: %ux%u  payload=%u  像素类型=0x%x",
                  info.nWidth, info.nHeight, info.nFrameLen,
                  static_cast<unsigned>(info.enPixelType));
    }

    MV_CC_FreeImageBuffer(handle_, &raw);    // ★ 拷贝完成后才能归还缓冲
    return true;
  }

  // ==========================================================================
  //  收尾：StopGrabbing → CloseDevice → DestroyHandle（顺序不能乱）
  // ==========================================================================
  void mvsClose()
  {
    mvs_ready_ = false;
    if (handle_ == nullptr) return;
    MV_CC_StopGrabbing(handle_);
    MV_CC_CloseDevice(handle_);
    MV_CC_DestroyHandle(handle_);
    handle_ = nullptr;
  }

  // 掉线恢复：收尾 → USB 硬复位 → 重新初始化
  bool recoverMvs()
  {
    mvsClose();
    resetUsb();
    return mvsStart();
  }

  // USB 硬件复位：相机僵死时靠这个救回来
  // VID:PID 只用来让 libusb 找到这颗 USB 设备做复位，与"选哪台相机"无关
  void resetUsb()
  {
    if (vid_ < 0 || pid_ < 0) return;
    libusb_device_handle * h =
      libusb_open_device_with_vid_pid(nullptr, static_cast<uint16_t>(vid_),
                                      static_cast<uint16_t>(pid_));
    if (h == nullptr) {
      RCLCPP_WARN(get_logger(), "libusb 打不开设备，跳过复位");
      return;
    }
    if (libusb_reset_device(h) != 0) RCLCPP_WARN(get_logger(), "USB 复位失败");
    else                             RCLCPP_INFO(get_logger(), "USB 复位成功");
    libusb_close(h);
  }

  // ---- 相机参数写入的薄封装（底层是 GenICam 特征树按【名字】查找节点）----
  void setEnum(const std::string & name, unsigned int value)
  {
    const unsigned int ret = MV_CC_SetEnumValue(handle_, name.c_str(), value);
    if (ret != MV_OK)
      RCLCPP_WARN(get_logger(), "MV_CC_SetEnumValue(\"%s\", %u) 失败: 0x%x",
                  name.c_str(), value, ret);
  }

  void setFloat(const std::string & name, double value)
  {
    const unsigned int ret = MV_CC_SetFloatValue(handle_, name.c_str(), value);
    if (ret != MV_OK)
      RCLCPP_WARN(get_logger(), "MV_CC_SetFloatValue(\"%s\", %.1f) 失败: 0x%x",
                  name.c_str(), value, ret);
  }

  // ---- 参数 ----
  double      exposure_us_ = 2000.0;
  double      gain_        = 0.0;
  double      frame_rate_  = 165.0;
  bool        auto_gain_   = true;
  bool        auto_wb_     = true;
  int         vid_         = 0x2bdf;
  int         pid_         = 0x0001;
  std::string serial_no_;
  std::string frame_id_;
  std::string topic_;

  // ---- ROS ----
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr pub_;

  // ---- MVS ----
  void * handle_ = nullptr;
  bool   mvs_ready_ = false;
  bool   first_frame_logged_ = false;

  // ---- 采集线程 ----
  std::thread capture_thread_;
  std::atomic<bool> quit_{false};
  std::atomic<unsigned long> frame_count_{0};
};

// ============================================================================
int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);

  try {
    auto node = std::make_shared<CameraNode>();
    rclcpp::spin(node);
  } catch (const std::exception & e) {
    RCLCPP_ERROR(rclcpp::get_logger("camera_node"), "启动失败: %s", e.what());
    rclcpp::shutdown();
    return 1;
  }

  rclcpp::shutdown();
  return 0;
}
