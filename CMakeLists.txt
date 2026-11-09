cmake_minimum_required(VERSION 3.20)
project(armor_re)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ============================================================================
#  依赖
# ============================================================================
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(example_interfaces REQUIRED)
find_package(rosidl_default_generators REQUIRED)

# ---- 【新增】发布图像所需 ----
#   sensor_msgs : sensor_msgs/msg/Image 消息类型
#   cv_bridge   : cv::Mat <-> sensor_msgs/msg/Image 互转
find_package(sensor_msgs REQUIRED)
find_package(cv_bridge REQUIRED)

# ---- 相机节点用到的系统库 ----
find_package(PkgConfig REQUIRED)
pkg_check_modules(LIBUSB REQUIRED libusb-1.0)   # 相机掉线后的 USB 硬复位

# ============================================================================
#  海康 MVS SDK —— 现在是【硬性依赖】
#  ---------------------------------------------------------------------------
#  早期版本把 SDK 做成可选的：找不到 /opt/MVS 就让 cam.cpp 退回 cv::VideoCapture
#  读 avi 文件。那条降级分支已经从 cam.cpp 删掉了，所以这里也改成必需：
#  找不到就直接在 configure 阶段报错，而不是编出一个"永远打不开相机"的节点。
# ============================================================================
find_path(MVS_INCLUDE_DIR
  NAMES MvCameraControl.h
  PATHS /opt/MVS/include /opt/MVS/include/64
  NO_DEFAULT_PATH)
find_library(MVS_LIBRARY
  NAMES MvCameraControl libMvCameraControl.so
  PATHS /opt/MVS/lib /opt/MVS/lib/64
  NO_DEFAULT_PATH)

if(NOT MVS_INCLUDE_DIR OR NOT MVS_LIBRARY)
  message(FATAL_ERROR
    "armor_re: 找不到海康 MVS SDK（/opt/MVS/include/MvCameraControl.h 或 libMvCameraControl.so）。\n"
    "  CameraNode 依赖它，没有 SDK 编不出可用的相机节点。\n"
    "  安装方式：下载 MVS Linux x86_64 包 → sudo ./install.sh（装到 /opt/MVS）。")
endif()
message(STATUS "armor_re: 找到海康 MVS SDK")
message(STATUS "   头文件: ${MVS_INCLUDE_DIR}")
message(STATUS "   库文件: ${MVS_LIBRARY}")

# ============================================================================
#  接口（消息定义）
# ============================================================================
rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/Detect.msg"
  "msg/GimbalState.msg"      # 【新增】电控 -> 视觉
  "msg/GimbalCmd.msg"        # 【新增】视觉 -> 电控
  DEPENDENCIES std_msgs      # GimbalState 里用了 std_msgs/Header
)
rosidl_get_typesupport_target(cpp_typesupport_target ${PROJECT_NAME} "rosidl_typesupport_cpp")

# ============================================================================
#  原有节点
# ============================================================================
add_executable(ArmorRe src/detect_armor.cpp)
add_executable(ArmorEKF src/EKF.cpp)

target_link_libraries(ArmorRe ${rclcpp_LIBRARIES} ${OpenCV_LIBS} Eigen3::Eigen ${example_interfaces} "${cpp_typesupport_target}")
target_include_directories(ArmorRe PRIVATE ${rclcpp_INCLUDE_DIRS})
target_include_directories(ArmorRe PRIVATE ${OpenCV_INCLUDE_DIRS})

target_link_libraries(ArmorEKF ${rclcpp_LIBRARIES} ${OpenCV_LIBS} Eigen3::Eigen ${example_interfaces} "${cpp_typesupport_target}")
target_include_directories(ArmorEKF PRIVATE ${rclcpp_INCLUDE_DIRS})
target_include_directories(ArmorEKF PRIVATE ${OpenCV_INCLUDE_DIRS})

# ============================================================================
#  相机节点 CameraNode
#  ---------------------------------------------------------------------------
#  需要四组库：
#    1) ROS2 侧 : rclcpp + sensor_msgs + cv_bridge   ← 发布 sensor_msgs/msg/Image
#    2) 图像侧 : OpenCV                              ← cv::Mat / Bayer→BGR
#    3) 硬件侧 : MVS SDK + libusb                     ← 海康采集 + USB 掉线复位
# ============================================================================
add_executable(CameraNode src/cam.cpp)

# ★ 这里必须用 imported target（新式写法）而不是 ${xxx_LIBRARIES}/${xxx_INCLUDE_DIRS}：
#   cv_bridge 的自带头文件目录 /opt/ros/<distro>/include/cv_bridge 只挂在
#   cv_bridge::cv_bridge 这个 target 的 INTERFACE_INCLUDE_DIRECTORIES 上，
#   老式的 cv_bridge_INCLUDE_DIRS 变量只汇总了它依赖的包，不含自身，会报
#   "cv_bridge/cv_bridge.hpp: 没有那个文件或目录"。
#   sensor_msgs 同理：用聚合 target sensor_msgs::sensor_msgs。
target_link_libraries(CameraNode
  rclcpp::rclcpp
  sensor_msgs::sensor_msgs
  cv_bridge::cv_bridge
  ${OpenCV_LIBS}
)
target_include_directories(CameraNode PRIVATE ${OpenCV_INCLUDE_DIRS})

# MVS SDK（硬性依赖，前面已 FATAL_ERROR 兜底）
target_include_directories(CameraNode PRIVATE ${MVS_INCLUDE_DIR})
target_link_libraries(CameraNode ${MVS_LIBRARY})

# libusb（硬性依赖）—— 相机掉线后的 USB 硬复位
target_include_directories(CameraNode PRIVATE ${LIBUSB_INCLUDE_DIRS})
target_link_libraries(CameraNode ${LIBUSB_LIBRARIES})
target_link_directories(CameraNode PRIVATE ${LIBUSB_LIBRARY_DIRS})

# --------------------------------------------------------------------------
#  【必须】把 MVS 的库目录写进 RPATH
#  --------------------------------------------------------------------------
#  ${MVS_LIBRARY} 是绝对路径，但链接器只会把 soname（libMvCameraControl.so）
#  记进 DT_NEEDED，【不会】记住它在哪个目录。于是运行期靠动态加载器去搜：
#      · LD_LIBRARY_PATH（~/.bashrc 里的 set_env_path.sh 就是干这个的）
#      · /etc/ld.so.conf
#      · 二进制自带的 RPATH/RUNPATH
#  只靠第一种的后果：换个不加载 bashrc 的启动方式（launch 文件 / systemd /
#  IDE 的任务），就会报
#      error while loading shared libraries: libMvCameraControl.so
#  写死 RPATH 之后，二进制自己知道去哪找，跟环境变量无关。
#     BUILD_RPATH   : build/ 目录下的那份（直接跑 build/armor_re/CameraNode 时用）
#     INSTALL_RPATH : install/ 目录下的那份（ros2 run / ros2 launch 用的）
# --------------------------------------------------------------------------
get_filename_component(MVS_LIB_DIR "${MVS_LIBRARY}" DIRECTORY)
set_target_properties(CameraNode PROPERTIES
  BUILD_RPATH   "${MVS_LIB_DIR}"
  INSTALL_RPATH "${MVS_LIB_DIR}"
)

# ============================================================================
#  【新增】串口节点 SerialNode
#  ---------------------------------------------------------------------------
#  只需要 rclcpp + 本包生成的消息类型。
#  termios / poll / write 都是系统调用，【不需要链接任何额外库】。
#  刻意不链接 OpenCV / Eigen / MVS —— 保持这个节点最小、最好部署。
# ============================================================================
add_executable(SerialNode src/serial_node.cpp)

target_link_libraries(SerialNode
  rclcpp::rclcpp
  "${cpp_typesupport_target}"
)
# _DEFAULT_SOURCE: 让 termios 的 B115200 / CRTSCTS 等常量在 -std=c++17 下可见
target_compile_definitions(SerialNode PRIVATE _DEFAULT_SOURCE)

# ============================================================================
#  安装
# ============================================================================
install(TARGETS ArmorEKF   DESTINATION lib/${PROJECT_NAME})
install(TARGETS ArmorRe    DESTINATION lib/${PROJECT_NAME})
install(TARGETS CameraNode DESTINATION lib/${PROJECT_NAME})
install(TARGETS SerialNode DESTINATION lib/${PROJECT_NAME})
install(DIRECTORY launch DESTINATION share/${PROJECT_NAME})

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
