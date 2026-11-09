#include <iostream>
#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <cmath>
#include <rclcpp/rclcpp.hpp>
#include "armor_re/msg/detect.hpp"


#include <sys/ipc.h>
#include <sys/shm.h>


key_t key = ftok("/tmp/shm", 65);
int shmid = shmget(key, 0, 0);
double* arr = (double*)shmat(shmid, nullptr, 0);

struct lightbar{
  int x, y, height;
  int width = 0;          // 新增：灯条宽度。分类装甲板编号时要用（原代码把它丢掉了）
  bool operator<(const lightbar &a)const{
    return x < a.x;
  }
};
struct armor{
  lightbar leftbar, rightbar;
  std::string number;         // 新增：识别出的编号 "1"~"4"/outpost/guard/base；空串=未识别
  double confidence = 0.0;    // 新增：softmax 置信度
};
armor _call_back(armor_re::msg::Detect message) {
    armor a;
    lightbar l, r;
    l.height = message.lh;
    l.x = message.lx;
    l.y = message.ly;
    r.height = message.rh;
    r.x = message.rx;
    r.y = message.ry;
    l.width = r.width = 0;
    a.leftbar = l; a.leftbar = r;
    a.number = "0";  a.confidence = 0;
    return a;
}
int lx, ly, rx, ry, lh, rh, la, ra;
bool new_data;
bool first_data=true;
rclcpp::Time last_time;
double dt;
class myNode:public rclcpp::Node{
    public:
        myNode(const std::string& node_name):Node(node_name){
            reader = this -> create_subscription<armor_re::msg::Detect>("armor_location", 10, std::bind(&myNode::call_back, this, std::placeholders::_1));
        }
    private:
        rclcpp::Subscription<armor_re::msg::Detect>::SharedPtr reader;
        void call_back(armor_re::msg::Detect message) {
            new_data = true;
            if (first_data)
                last_time = message.header.stamp;
            first_data = false;    
            lx = message.lx;
            ly = message.ly;
            lh = message.lh;
            la = message.la;
            rx = message.rx;
            ry = message.ry;
            rh = message.rh;
            ra = message.ra;
            dt = 0 - (last_time - message.header.stamp).seconds();
            last_time = message.header.stamp;
        }
        
};
std::shared_ptr<myNode> node;

cv::Mat cameraMatrix = (cv::Mat_<double>(3, 3) <<
    2399.583734143874, -0.114556032386470, 688.7026376077414, 
    0, 2398.597221612343, 533.4234008238891, 
    0, 0, 1);

cv::Mat distCoeffs = (cv::Mat_<double>(1, 5) << 0.002592988512527, 0.048045410117226, 0.293333651684761,
 0.0001643665962815789, 0.0001925868649099913);

 
Eigen::MatrixXd jacobianH(Eigen::VectorXd x) {
  Eigen::MatrixXd hd(3, 7);
  hd << 1, 0, 0, 0, x(6)*cos(x(4)), 0, sin(x(4)), 
  0, 1, 0, 0, -x(6)*sin(x(4)), 0, cos(x(4)), 
  0, 0, 0, 0, 1, 0, 0;
  return hd;
}
  
Eigen::VectorXd h(Eigen::VectorXd x) {
  Eigen::VectorXd hx(3);
  hx << x(0)+sin(x(4))*x(6), x(1)+cos(x(4))*x(6), x(4);
  return hx;
}

std::vector<std::vector<armor>> amlist;

double get_angel(const Eigen::VectorXd& a, const Eigen::VectorXd& b) {
    double record_score = 0;
    int index = 0;
    for (int i=0; i<=3; i++) { // index 偏转过几个直角
        Eigen::VectorXd predict_armor(3);
        predict_armor << b(0)+cos(a(2)+i*1.57)*b(6), b(1)+sin(a(2)+i*1.57)*b(6), a(2)+i*1.57;
        double score_sum = std::pow((predict_armor(0)-a(0)), 2) + std::pow((predict_armor(1)-a(1)), 2);
        if (i == 0 || score_sum < record_score) {
            index = i;
            record_score = score_sum;
        }
    }
    //return a(2);
    return a(2) + index*1.57 + ((int)(b(4)/3.14))*3.14;
}
void armorEKFupdata() {
  
  Eigen::MatrixXd Q = Eigen::MatrixXd::Zero(7, 7);
  Q.diagonal() << 0.01, 0.01, 0.1, 0.1, 0.01, 0.1, 0.0000;
  Eigen::MatrixXd I = Eigen::MatrixXd::Identity(7, 7);
  Eigen::MatrixXd R = Eigen::Matrix3d::Zero(3, 3);
  R.diagonal() << 0.01, 0.01, 0.100;

  Eigen::MatrixXd F(7, 7);
  F << 1, 0, 1, 0, 0, 0, 0, 
       0, 1, 0, 1, 0, 0, 0,
       0, 0, 1, 0, 0, 0, 0,
       0, 0, 0, 1, 0, 0, 0,
       0, 0, 0, 0, 1, 1, 0, 
       0, 0, 0, 0, 0, 1, 0,
       0, 0, 0, 0, 0, 0, 1;
  //x y vx vy angle Vangle R
  Eigen::VectorXd x(7);
  x << 0, 0, 0, 0, 0, 0, 0.25;
  Eigen::MatrixXd p = Eigen::MatrixXd::Zero(7, 7);
  p.diagonal() << 100, 100, 100, 100, 100, 100, 10;
  
  //for (int count_ = 0; count_ <= 2300; count_++) {
  while (rclcpp::ok() && cv::waitKey(3) != 27) {
    new_data = false;
    rclcpp::spin_some(node);
    if (!new_data)
        continue;
    //std::cin>>lx>>ly>>rx>>ry>>lh>>rh;
        /* code */
    
        F(0, 2) = dt;
        F(1, 3) = dt;
        F(4, 5) = dt;

    cv::Mat tvec, rvec;
    std::vector<cv::Point2f> imagepoints = {
      cv::Point2f(lx-sin(la*3.14/180)*lh/2, ly-lh/2+cos(la*3.14/180)*lh/2), 
      cv::Point2f(lx+sin(la*3.14/180)*lh/2, ly+lh/2-cos(la*3.14/180)*lh/2), 
      cv::Point2f(rx-sin(ra*3.14/180)*rh/2, ry-rh/2+cos(ra*3.14/180)*rh/2), 
      cv::Point2f(rx+sin(ra*3.14/180)*rh/2, ry+rh/2-cos(ra*3.14/180)*rh/2) 
    };
    std::vector<cv::Point3f> worldpoints = {
      cv::Point3f(-0.15, 0.025, 0),
      cv::Point3f(-0.15, -0.025, 0),
      cv::Point3f(0.15, 0.025, 0),
      cv::Point3f(0.15, -0.025, 0)
    };
    cv::solvePnP(worldpoints, imagepoints, cameraMatrix, distCoeffs, rvec, tvec, false, cv::SOLVEPNP_IPPE);
    
    
    Eigen::VectorXd z(3);

    cv::Mat R_;
    cv::Rodrigues(rvec, R_);

    if (R_.at<double>(2,2) != 0)
        z << tvec.at<double>(0, 0), tvec.at<double>(2, 0), std::atan(R_.at<double>(0, 2) / R_.at<double>(2, 2));
    else
        z << tvec.at<double>(0, 0), tvec.at<double>(2, 0), std::atan(R_.at<double>(0, 2) / 0.001);
    
    z(2) = get_angel(z, x);
    std::cerr<<z(0)<<" "<<z(1)<<" "<<z(2)<<std::endl;

  
  
    Eigen::VectorXd x_hat = F * x;
    Eigen::MatrixXd p_hat = F * p * F.transpose() + Q;
  
    Eigen::VectorXd res = z - h(x_hat);
    Eigen::MatrixXd H = jacobianH(x_hat);
    
    Eigen::MatrixXd S = H * p_hat * H.transpose() + R;
    Eigen::MatrixXd K = p_hat * H.transpose() * S.inverse();
  
    x = x_hat + K * res;
    p = (I - K * H) * p_hat;
    //for (int i=0; i<7; i++) {
    //    std::cout<<x(i)<<" ";
    //}
    std::cerr<<"vx: "<<x(2)<<" vy: "<<x(3)<<" w:"<<x(5)<<" r:"<<x(6)<<std::endl;
    //std::cerr<<"vx: "<<x(2)*60<<" vy: "<<x(3)*60<<" w:"<<x(5)*60<<std::endl;
    arr[0] = x(2);
    arr[1] = x(3);
    arr[2] = x(5);
    
    std::cout<<"finish"<<std::endl;
    }
    
}


int main(int argc, char** argv){
    //freopen("/home/hoyosho/test/tmp_ws/armorlist.txt", "r", stdin);
    freopen("111.txt", "w", stdout);
    rclcpp::init(argc, argv);
    node = std::make_shared<myNode>("reader");

    armorEKFupdata();
    shmdt(arr);
    rclcpp::shutdown();
    return 0;
}