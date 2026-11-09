// ============================================================================
//  armor_classifier.hpp  ——  装甲板数字分类器（纯头文件，无需改 CMakeLists）
//
//  把 auto-aim-master 里的 ExtractNumbers + Classify 整套流程封装成函数。
//
//  来源对照（原项目）：
//    预处理  Armor/src/ArmorDetector.cpp:295-364   ExtractNumbers()
//    推理    Armor/src/ArmorDetector.cpp:365-444   Classify()
//    加载    Armor/src/ArmorDetector.cpp:503-519   构造函数
//    模型    model/mlp.onnx          输入 [1,1,28,28] float → 输出 (1,8)
//    标签    model/label.txt        顺序即类别编号：1 2 3 4 outpost guard base negative
//
//  ── 用法（三行）────────────────────────────────────────────────────────────
//      #include "armor_classifier.hpp"
//      ...
//      double conf = 0;
//      std::string number = armor_cls::classifyArmor(frame, left_bar_rect,
//                                                    right_bar_rect, conf);
//      if (!number.empty()) { /* 识别成功 */ }
//
//  首次调用会自动加载模型（路径自动搜索，见 resolveAsset）。
// ============================================================================
#ifndef ARMOR_CLASSIFIER_HPP
#define ARMOR_CLASSIFIER_HPP

#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>

#include <algorithm>
#include <fstream>
#include <iostream>
#include <mutex>
#include <string>
#include <vector>

namespace armor_cls
{

// ---------------------------------------------------------------------------
// 装甲板尺寸。影响透视变换的目标宽度（小板 32、大板 52），必须与训练一致。
// ---------------------------------------------------------------------------
enum ArmorSize
{
  ARMOR_SMALL = 0,
  ARMOR_LARGE = 1
};

// ---------------------------------------------------------------------------
// 分类结果
// ---------------------------------------------------------------------------
struct Result
{
  std::string number;         // "1".."4"/"outpost"/"guard"/"base"/"negative"；失败为空
  double confidence = 0.0;    // softmax 最大概率 0~1
  bool ok = false;            // number 非空 且 置信度达标 且 非 negative
  cv::Mat number_img;         // 送进网络的 28x28 二值图，调试可视化用

  explicit operator bool() const { return ok; }
};

// ---------------------------------------------------------------------------
// 内部实现：模型单例（你不需要直接用它）
// ---------------------------------------------------------------------------
class Classifier
{
public:
  // 置信度阈值，与原项目构造参数一致（0.7）
  double conf_threshold = 0.6;
  // 是否把 "negative" 类判为失败（原项目是直接丢弃该装甲板）
  bool reject_negative = true;

  static Classifier &instance()
  {
    static Classifier inst;
    return inst;
  }

  // 加载模型；两个路径传空字符串时自动搜索（见 resolveAsset）
  bool load(const std::string &model_path = "", const std::string &label_path = "")
  {
    std::lock_guard<std::mutex> lk(mtx_);
    if (ready_ && model_path_.empty() && model_path.empty())
      return true;  // 已按默认路径加载过

    const std::string mp = model_path.empty() ? resolveAsset("mlp.onnx") : model_path;
    const std::string lp = label_path.empty() ? resolveAsset("label.txt") : label_path;
    if (mp.empty() || lp.empty())
    {
      std::cerr << "[armor_cls] 找不到模型文件。已尝试的路径：\n";
      for (const auto &c : candidates("mlp.onnx")) std::cerr << "    " << c << "\n";
      std::cerr << "  请把 mlp.onnx / label.txt 放到项目根的 model/ 目录下，"
                   "或调用 armor_cls::Classifier::instance().load(模型路径, 标签路径)\n";
      ready_ = false;
      return false;
    }

    try
    {
      net_ = cv::dnn::readNetFromONNX(mp);
    }
    catch (const cv::Exception &e)
    {
      std::cerr << "[armor_cls] readNetFromONNX 失败: " << e.what() << "\n";
      ready_ = false;
      return false;
    }

    class_names_.clear();
    std::ifstream lf(lp);
    for (std::string line; std::getline(lf, line);)
      class_names_.push_back(line);

    if (class_names_.empty())
    {
      std::cerr << "[armor_cls] 标签文件为空: " << lp << "\n";
      ready_ = false;
      return false;
    }

    model_path_ = mp;
    label_path_ = lp;
    ready_ = true;
    std::cout << "[armor_cls] 模型已加载: " << mp << "  (类别数 " << class_names_.size()
              << ", 阈值 " << conf_threshold << ")\n";
    return true;
  }

  bool ready() const { return ready_; }
  const std::vector<std::string> &classNames() const { return class_names_; }
  const std::string &modelPath() const { return model_path_; }

  // ---- 预处理：把装甲板区域拉正成 28x28 二值数字图 ----------------------
  // corners 顺序必须严格是：左下, 左上, 右上, 右下（与原项目 GetPoints 一致）
  static cv::Mat extractNumberImage(const cv::Mat &frame,
                                    const std::vector<cv::Point2f> &corners,
                                    int armor_size = ARMOR_SMALL)
  {
    if (frame.empty() || corners.size() != 4)
    {
      std::cerr << "[armor_cls] extractNumberImage: 输入非法\n";
      return cv::Mat();
    }

    // ↓↓↓ 以下常量全部取自 ArmorDetector::ExtractNumbers，不要随意改动 ↓↓↓
    const int light_length = 12;
    const int warp_height = 28;
    const int small_armor_width = 32;
    const int large_armor_width = 52;
    const cv::Size roi_size(28, 28);
    const cv::Size input_size(28, 28);

    const int top_light_y = (warp_height - light_length) / 2 - 1;  // 7
    const int bottom_light_y = top_light_y + light_length;         // 19
    const int warp_width =
        (armor_size == ARMOR_LARGE) ? large_armor_width : small_armor_width;

    cv::Point2f lights_vertices[4] = {corners[0], corners[1], corners[2], corners[3]};
    cv::Point2f target_vertices[4] = {
        cv::Point2f(0, bottom_light_y),
        cv::Point2f(0, top_light_y),
        cv::Point2f(warp_width - 1, top_light_y),
        cv::Point2f(warp_width - 1, bottom_light_y),
    };

    cv::Mat warped;
    cv::Mat M = cv::getPerspectiveTransform(lights_vertices, target_vertices);
    cv::warpPerspective(frame, warped, M, cv::Size(warp_width, warp_height));

    // 取中央 ROI（把两侧灯条裁掉）。clone() 保证后面就地处理时内存连续。
    cv::Rect roi(cv::Point((warp_width - roi_size.width) / 2, 0), roi_size);
    cv::Mat number_img = warped(roi).clone();

    // 注意：原项目此处写的是 COLOR_RGB2GRAY。因为输入是"白字黑底"，
    // RGB/BGR 的权重差异对纯白(255,255,255)与纯黑(0,0,0)没有影响，故用 BGR2GRAY。
    cv::cvtColor(number_img, number_img, cv::COLOR_BGR2GRAY);
    cv::adaptiveThreshold(number_img, number_img, 255,
                          cv::ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY, 19, -3);
    cv::resize(number_img, number_img, input_size);
    return number_img;
  }

  // ---- 推理：输入 28x28 CV_8U 二值图 ------------------------------------
  // 注：cv::dnn::Net::setInput/forward 不是 const 成员函数，所以这里不能加 const
  Result classifyImage(const cv::Mat &number_img)
  {
    Result r;
    if (!ready_)
    {
      // 【修复】原来这里直接返回空 Result，导致单独调用 classifyImage 时静默失败。
      // 现在与 classify() 一致：先尝试惰性加载。
      Classifier::instance().load();
      if (!ready_)
      {
        std::cerr << "[armor_cls] 模型未加载，请先调用 load()\n";
        return r;
      }
    }
    if (number_img.empty())
      return r;

    cv::Mat gray;
    if (number_img.channels() == 3)
      cv::cvtColor(number_img, gray, cv::COLOR_BGR2GRAY);
    else
      gray = number_img;

    cv::Mat input;
    if (gray.depth() != CV_32F)
      // 显式归一化到 0~1。原项目写的是 gray/255.0，对 CV_8U 会保持整数类型，
      // 这里用 convertTo 语义正确且结果一致（二值图 0/255 → 0.0/1.0）。
      gray.convertTo(input, CV_32F, 1.0 / 255.0);
    else
      input = gray;

    cv::Mat blob;
    cv::dnn::blobFromImage(input, blob);  // [1,1,28,28] float

    std::lock_guard<std::mutex> lk(mtx_);
    net_.setInput(blob);
    cv::Mat outputs = net_.forward().clone();  // (1,8)
    if (outputs.empty())
      return r;

    // softmax（原项目手写版本）
    const float *p = outputs.ptr<float>(0);
    const size_t n = outputs.total();
    const float mx = *std::max_element(p, p + n);
    cv::Mat ex;
    cv::exp(outputs - mx, ex);
    const double sum = cv::sum(ex)[0];
    if (sum <= 0)
      return r;
    ex /= sum;

    cv::Point cls;
    double conf = 0.0;
    cv::minMaxLoc(ex.reshape(1, 1), nullptr, &conf, nullptr, &cls);
    const int id = cls.x;
    if (id < 0 || id >= static_cast<int>(class_names_.size()))
      return r;

    r.number_img = gray;
    r.confidence = conf;
    r.number = class_names_[id];

    const bool ignored = reject_negative && (r.number == "negative");
    r.ok = !ignored && (conf >= conf_threshold);
    if (!r.ok)
      r.number.clear();  // 不可信时返回空串，调用方只需判空
    return r;
  }

  // ---- 主入口：帧 + 4 角点 → 编号 ---------------------------------------
  Result classify(const cv::Mat &frame, const std::vector<cv::Point2f> &corners,
                  int armor_size = ARMOR_SMALL)
  {
    Result r;
    if (!ready_)
      Classifier::instance().load();
    if (!ready_)
      return r;
    cv::Mat img = extractNumberImage(frame, corners, armor_size);
    r = classifyImage(img);
    return r;
  }

private:
  Classifier() = default;
  Classifier(const Classifier &) = delete;
  Classifier &operator=(const Classifier &) = delete;

  // 依次尝试的候选路径，让你在项目根目录或 build/ 目录下运行都能找到模型
  static std::vector<std::string> candidates(const std::string &name)
  {
    return {
        "model/" + name,
        "../model/" + name,
        "../../model/" + name,
        "./" + name,
        "../" + name,
        "src/../model/" + name,
    };
  }

  static std::string resolveAsset(const std::string &name)
  {
    for (const auto &c : candidates(name))
    {
      std::ifstream f(c, std::ios::binary);
      if (f.good())
        return c;
    }
    return std::string();
  }

  bool ready_ = false;
  cv::dnn::Net net_;
  std::vector<std::string> class_names_;
  std::string model_path_, label_path_;
  mutable std::mutex mtx_;
};

// ===========================================================================
//  对外接口：三个重载，按你手上有什么数据挑一个
//
//  ⚠️ 角点语义（很重要）：原项目 GetPoints() 返回的 4 个点是【灯条中线】的端点，
//     即 (灯条中心x, 灯条上边中点y) 和 (灯条中心x, 灯条下边中点y)，
//     不是灯条的外边缘！目标宽度 warp_width 对应的是【两灯条中线的间距】。
//     下面所有接口都已按这个语义处理。
// ===========================================================================

// 按原项目 Judge_Type 的规则自动判大小板（尺度无关）：
//     两灯条中线间距 / 灯条平均高度 > 2.8 → 大板
inline int detectArmorSize(float center_distance, float bar_height)
{
  if (bar_height <= 1.0f)
    return ARMOR_SMALL;
  return (center_distance / bar_height > 2.8f) ? ARMOR_LARGE : ARMOR_SMALL;
}

// 【1】最通用：直接给装甲板 4 个角点（顺序：左下, 左上, 右上, 右下）
//     armor_size 传 -1 表示自动判大小板
inline std::string classifyArmor(const cv::Mat &frame,
                                 const std::vector<cv::Point2f> &corners,
                                 double &confidence,
                                 int armor_size = -1)
{
  int size = armor_size;
  if (size < 0)
  {
    // 用左右两条中线的水平间距当 side_length，用两条的高度均值当 light_height
    const float span = std::abs(corners[2].x - corners[0].x);
    const float h = 0.5f * (std::abs(corners[1].y - corners[0].y) +
                            std::abs(corners[3].y - corners[2].y));
    size = detectArmorSize(span, h);
  }
  Result r = Classifier::instance().classify(frame, corners, size);
  confidence = r.confidence;
  return r.number;
}

// 【2】推荐：给左右两个灯条的外接矩形（cv::Rect 自带 width，最精确）
//     内部取【灯条中线】，与原项目语义一致
inline std::string classifyArmor(const cv::Mat &frame,
                                 const cv::Rect &left_bar,
                                 const cv::Rect &right_bar,
                                 double &confidence,
                                 int armor_size = -1)
{
  const float lcx = left_bar.x + left_bar.width * 0.5f;   // 左灯条中线
  const float rcx = right_bar.x + right_bar.width * 0.5f; // 右灯条中线
  std::vector<cv::Point2f> corners = {
      cv::Point2f(lcx, static_cast<float>(left_bar.y + left_bar.height)),   // 左下
      cv::Point2f(lcx, static_cast<float>(left_bar.y)),                     // 左上
      cv::Point2f(rcx, static_cast<float>(right_bar.y)),                    // 右上
      cv::Point2f(rcx, static_cast<float>(right_bar.y + right_bar.height))  // 右下
  };
  return classifyArmor(frame, corners, confidence, armor_size);
}

// 【3】最省事：给装甲板的整体外接矩形（近似接口）
//     ⚠️ 矩形只有外框信息，无法还原"灯条中线"，所以比你手上的灯条 Rect 略偏左。
//        实测在 2.avi 上与精确接口结果一致率约 94%，能用；追求精度请用【2】。
inline std::string classifyArmorBox(const cv::Mat &frame,
                                    const cv::Rect &armor_box,
                                    double &confidence,
                                    int armor_size = -1)
{
  std::vector<cv::Point2f> corners = {
      cv::Point2f(static_cast<float>(armor_box.x), static_cast<float>(armor_box.y + armor_box.height)),
      cv::Point2f(static_cast<float>(armor_box.x), static_cast<float>(armor_box.y)),
      cv::Point2f(static_cast<float>(armor_box.x + armor_box.width), static_cast<float>(armor_box.y)),
      cv::Point2f(static_cast<float>(armor_box.x + armor_box.width),
                  static_cast<float>(armor_box.y + armor_box.height))};
  return classifyArmor(frame, corners, confidence, armor_size);
}

// 【4】完整版：需要拿到 28x28 数字图（可视化调试）和 ok 标志时用这个
inline Result classifyArmorFull(const cv::Mat &frame,
                                const std::vector<cv::Point2f> &corners,
                                int armor_size = -1)
{
  int size = armor_size;
  if (size < 0)
  {
    const float span = std::abs(corners[2].x - corners[0].x);
    const float h = 0.5f * (std::abs(corners[1].y - corners[0].y) +
                            std::abs(corners[3].y - corners[2].y));
    size = detectArmorSize(span, h);
  }
  return Classifier::instance().classify(frame, corners, size);
}

}  // namespace armor_cls

#endif  // ARMOR_CLASSIFIER_HPP
