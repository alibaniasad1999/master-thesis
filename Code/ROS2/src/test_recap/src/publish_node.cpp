#include <chrono>
#include <memory>
#include <string>

#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/string.hpp"

using namespace std::chrono_literals;

class MinimalPublisher : public rclcpp::Node {
	public: MinimalPublisher(): Node("minimal_pub_node_name"), count_(0) {
		publisher_ = this->create_publisher<std_msgs::msg::String>("topic", 10);
		auto timer_callback = [this]() -> void {
			auto message = std_msgs::msg::String();
			message.data = "Hello World: " + std::to_string(this->count_++);
			RCLCPP_INFO(this->get_logger(), "Publishing: '%s'", message.data.c_str());
			this->publisher_->publish(message);
		};
		timer_ = this->create_wall_timer(500ms, timer_callback);
		}
	private:
		rclcpp::TimerBase::SharedPtr timer_;
		rclcpp::Publisher<std_msgs::msg::String>::SharedPtr publisher_;
		size_t count_;
};

int main(int argc, char * argv[]){
	rclcpp::init(argc, argv);
	rclcpp::spin(std::make_shared<MinimalPublisher>());
	rclcpp::shutdown();
	return 0;
}
	
