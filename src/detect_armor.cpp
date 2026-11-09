#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>

#include <sys/ipc.h>
#include <sys/shm.h>

#include <cstdlib>
#include <algorithm>
#include <iostream>
#include <vector>
#include <queue>
#include <string>

#include <Eigen/Dense>
#include <cmath>

#include <rclcpp/rclcpp.hpp>
#include "armor_re/msg/detect.hpp"

// 数字分类器：从 auto-aim-master 的 ArmorDetector::ExtractNumbers/Classify 移植，
// 模型就是原工程的 model/mlp.onnx + label.txt（已复制到本包 model/ 目录）。
// 放在 src/ 下，同目录直接引号包含即可。
#include "armor_classifier.hpp"

key_t key = ftok("/tmp/shm", 65);
int shmid = shmget(key, 4*sizeof(double), IPC_CREAT | 0644);
double* arr = (double*)shmat(shmid, nullptr, 0);


struct lightbar{
  int x, y, height;
  int width = 0;          // 新增：灯条宽度。分类装甲板编号时要用（原代码把它丢掉了）
  double angel;
  bool operator<(const lightbar &a)const{
    return x < a.x;
  }
};
bool cmp(lightbar a, lightbar b) {
  return a.x < b.x;
}
struct armor{
  lightbar leftbar, rightbar;
  std::string number;         // 新增：识别出的编号 "1"~"4"/outpost/guard/base；空串=未识别
  double confidence = 0.0;    // 新增：softmax 置信度
};
bool match_judge(lightbar a, lightbar b) {
  if (b.x - a.x < a.height || b.x - a.x < b.height) { //假设板子为长方形
    //std::cerr<<"特判一号不同意"<<std::endl;
    return false;
  }
  if (a.angel<45 || b.angel < 45)
    return false;//假设版角度大于45
  if (b.y-b.height/2 > a.y+a.height/2 || a.y-a.height/2 > b.y ) { //假设板子两灯条的高低落差不大
    //std::cerr<<"特判二号不同意"<<std::endl;
    return false;
  }  
  if ((b.x - a.x) / a.height > 4 || (b.x - a.x) / b.height > 4 ) { //限制长宽比
    ///std::cerr<<"特判三号不同意"<<std::endl;
    return false;
  }
  if (a.height < 10 || b.height < 10 || a.height*(b.x-a.x) < 15 || a.height*(b.x-a.x) > 20000) {
    return false;
  }
  return true;
}

//std::priority_qu.eue<lightbar> lb;

std::vector<std::vector<armor>> amlist;


class myNode:public rclcpp::Node{
  public:
    myNode(const std::string& node_name):Node(node_name) {
      publisher = this -> create_publisher<armor_re::msg::Detect>("armor_location", 10);
    }
    void fsend(armor a) {
      armor_re::msg::Detect message;
      message.lh = a.leftbar.height;
      message.lx = a.leftbar.x + a.leftbar.width/2;
      message.ly = a.leftbar.y + a.leftbar.height/2;
      message.rh = a.rightbar.height;
      message.rx = a.rightbar.x + a.rightbar.width/2;
      message.ry = a.rightbar.y + a.rightbar.height/2;
      message.la = a.leftbar.angel;
      message.ra = a.rightbar.angel;
      message.header.stamp = this -> now();
      publisher -> publish(message);
    }
  private:
    rclcpp::Publisher<armor_re::msg::Detect>::SharedPtr publisher;
};
std::shared_ptr<myNode> node;

// ★ 原来这里有一行 cv::dnn::Net net = cv::dnn::readNetFromONNX("mlp.onnx");
//   它有两个问题，已注释掉，改用下面的 armor_cls::Classifier（见 main 里 load）：
//     1) 这是【全局变量初始化】，在 main() 之前执行。模型路径 "mlp.onnx" 是相对路径，
//        而 ros2 run 的工作目录不是包目录 → 找不到文件 → readNetFromONNX 抛异常 →
//        没人接 → terminate called after throwing an instance of 'cv::Exception'。
//        （实测就是这么崩的：Can't read ONNX file: mlp.onnx）
//     2) 光有一个 Net 还不够：还得自己做 透视变换→28×28 ROI→二值化→softmax→阈值判断，
//        这些装甲板专用预处理已经在 armor_classifier.hpp 里实现好了。
// cv::dnn::Net net = cv::dnn::readNetFromONNX("mlp.onnx");

void frame_process() {
  cv::VideoCapture cap;
  cap.open("//home//hoyosho//ros2_ws//opencv_minimal//3.avi");
  cv::Mat frame;
  cap.set(cv::CAP_PROP_POS_FRAMES, 30);
  cap.read(frame);

  cv::VideoWriter writer("//home//hoyosho//ros2_ws//opencv_minimal//res3.avi", cv::VideoWriter::fourcc('p', 'J', 'P', 'G'), 30, frame.size());

  arr[0] = arr[1] = arr[2] = arr[3] = 0;

  while (cv::waitKey(5)!=27 && !frame.empty()) {
    //cv::imshow("frame", frame);

    cv::Mat src;
    cap.read(src);
    if (src.empty()) break;          // 视频读完了：不加这行，下面 cvtColor 会因空图断言崩溃
    src.convertTo(frame, CV_8U, 1.1, 50);
    //cv::imshow("frame", frame);
    //cv::imshow("src", src);
    
    
          char buf[64];
          snprintf(buf, sizeof(buf), "%.3f", arr[0]);
          cv::putText(src, buf, cv::Point(720, 540), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255, 0, 255), 2);
          snprintf(buf, sizeof(buf), "%.3f", arr[1]);
          cv::putText(src, buf, cv::Point(720, 640), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255, 0, 255), 2);
          snprintf(buf, sizeof(buf), "%.3f", arr[2]);
          cv::putText(src, buf, cv::Point(720, 740), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255, 0, 255), 2);
          
    cv::Mat gray, blurred, mask;
    cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
    std::vector <cv::Mat> channels;
    cv::split(frame, channels);
    cv::GaussianBlur(gray, blurred, cv::Size(5, 5), 1.5);
    //cv::imshow("blurred", channels[0]);
    cv::inRange(blurred, cv::Scalar(150), cv::Scalar(250), mask);
    //cv::imshow("mask", mask);
    std::vector <std::vector <cv::Point>> contours;
    cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    
    lightbar lb[100];
    int len = 0;
    for (long unsigned i=0; i<contours.size(); i++) {
      if(contours[i].size() < 3)
        continue;
      cv::Rect rect = cv::boundingRect(contours[i]);
      cv::RotatedRect rect_ = cv::minAreaRect(contours[i]);
      
      if (rect.width >= rect.height)
        continue;
      cv::rectangle(src, rect, cv::Scalar(0, 255, 0), 2);
      cv::Point2f p;
      int length = rect_.size.height*rect_.size.height+rect_.size.width*rect_.size.width;
      length = pow(length, 0.5);
      p.x = rect_.center.x - rect_.size.height * sin(rect_.angle*3.14/180);
      p.y = rect_.center.y + rect_.size.height * cos(rect_.angle*3.14/180);
      
      cv::line(src, rect_.center, p, cv::Scalar(0, 0, 255), 3, 8, 0);


      lightbar a;
      a.x = rect.x;
      a.y = rect.y+rect.height/2;
      a.height = rect.height;
      a.width = rect.width;      // 新增：保存灯条宽度，分类时要用
      a.angel = rect_.angle;
      lb[len++] = a;
    }
    //cv::imshow("frame1", frame);
    std::sort(lb, lb+len);

    // ---------------------------------------------------------------------
    // 本帧识别出的装甲板列表：amlist[0] 即本帧结果，每帧重建。
    // 想改成"累积历史"就把 amlist.clear() 去掉；但注意 armorEKFupdata 是
    // 每帧一次的单步更新，累积历史会让它反复处理旧数据。
    // ---------------------------------------------------------------------
    amlist.clear();
    std::vector<armor> frame_armors;

    // 是否只收集"识别成功"的装甲板。改成 false 则未识别的也一并收进 amlist
    const bool only_classified = true;

    for (int i = 0; i < len; i++) {
      for (int j = i+1; j < len; j++) {
        //std::cout<<"is matching!"<<std::endl;
        if (match_judge(lb[i], lb[j])) {
          cv::Rect rect;
          rect.x = lb[i].x;
          rect.width = lb[j].x - lb[i].x;
          rect.y = lb[i].y - lb[i].height/2;
          rect.height = lb[i].height;
          
          //std::cerr<<"is drawing! S:"<<rect.width*rect.height<<std::endl;
          cv::rectangle(src, rect, cv::Scalar(0, 0, 255), 2);

          // ★ 新增：识别装甲板编号，并放进 amlist
          armor am;
          am.leftbar  = lb[i];
          am.rightbar = lb[j];

          // 由 lightbar 还原两个灯条的外接矩形交给 classifyArmor。
          // 内部会取灯条【中线】构造角点，与原项目 ArmorDetector::GetPoints 语义一致。
          cv::Rect left_rect (lb[i].x, lb[i].y - lb[i].height/2, lb[i].width, lb[i].height);
          cv::Rect right_rect(lb[j].x, lb[j].y - lb[j].height/2, lb[j].width, lb[j].height);
          //am.number = armor_cls::classifyArmor(frame, left_rect, right_rect, am.confidence);
          /* 接入神经网络识别数字*/
          // ---- 1. 分类：左右灯条的外接矩形 → 装甲板编号 ----
          //   内部做的事和原工程完全一致：
          //     取两灯条的【中线】构造 4 角点 → 透视变换到 28×(32小板/52大板)
          //     → 取中央 28×28 ROI → 灰度 → adaptiveThreshold(19,-3) → mlp.onnx 前向
          //     → softmax → 取最大类；置信度 < 阈值的、判成 "negative" 的都返回空串
          //   返回值："1"~"5" / "outpost" / "guard" / "base"，失败返回 ""（空串）
          am.number = armor_cls::classifyArmor(frame, left_rect, right_rect, am.confidence);

          // 调试用：stderr 不被 freopen 到 armorlist.txt，所以这里看得到
          //std::cerr << "[classify] " << (am.number.empty() ? "未识别" : am.number) << "  conf=" << cv::format("%.3f", am.confidence) << std::endl;


          std::cout<<lb[i].x<<" "<<lb[i].y<<" "<<lb[j].x<<" "<<lb[j].y<<" "<<lb[i].height<<" "<<lb[j].height<<std::endl;

          if (!am.number.empty()) {
            frame_armors.push_back(am);

            node -> fsend(am);
            
            cv::putText(src, am.number,
                        cv::Point(rect.x, rect.y - 8),
                        cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255, 0, 255), 2);
          } else if (!only_classified) {
            frame_armors.push_back(am);   // 编号为空也收
          }
        }
      }
    }

    // 本帧有可用装甲板才 push；amlist 为空表示本帧无目标
    if (!frame_armors.empty())
      amlist.push_back(frame_armors);

    cv::imshow("result", src);
    writer.write(src);
  }

} // 尚未处理装甲板的识别分类 与车辆装甲板的分类

int main(int argc, char ** argv) {
  freopen("armorlist.txt", "w", stdout);

  // ---- 加载数字分类模型（auto-aim-master 的 mlp.onnx）----
  //   路径优先级：环境变量 > 本包 model/ 目录
  //   ★ 置信度阈值 0.3 是原工程 AutoArmor 调过之后的值。
  //     原代码里写的 0.7 会把大量"其实认对了但只有 0.4~0.6 置信度"的真板丢掉。
  {
    const char * m = std::getenv("ARMOR_MODEL");
    const char * l = std::getenv("ARMOR_LABEL");
    const std::string mp = (m && m[0]) ? m : "/home/hoyosho/test/tmp_ws/src/armor_re/model/mlp.onnx";
    const std::string lp = (l && l[0]) ? l : "/home/hoyosho/test/tmp_ws/src/armor_re/model/label.txt";

    auto & cls = armor_cls::Classifier::instance();
    cls.conf_threshold = 0.3;
    if (!cls.load(mp, lp)) {
      std::cerr << "加载数字分类模型失败: " << mp << std::endl;
      return 1;
    }
    std::cerr << "数字分类模型已加载: " << mp << "  (置信度阈值 "
              << cls.conf_threshold << ")" << std::endl;
  }

  rclcpp::init(argc, argv);
  node = std::make_shared<myNode>("publisher");
  /*std::thread spin_thread([node](){
    rclcpp::spin(node);
  });*/
  std::cerr<<"running successfully"<<std::endl;
  frame_process();
  shmdt(arr);
  rclcpp::shutdown();
  //spin_thread.join();
  cv::destroyAllWindows();
  return 0;
}
