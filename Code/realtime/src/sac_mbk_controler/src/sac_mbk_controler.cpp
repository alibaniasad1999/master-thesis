#include "rclcpp/rclcpp.hpp"
#include "mbk_pid_controler_interface/srv/control_command.hpp"

class ControllerServiceNode : public rclcpp::Node
{
public:
    ControllerServiceNode()
        : Node("controller_service_node"), Kp(10.0), Kd(1.0), Ki(1.0), integral_error(0.0), setpoint(10.0)
    {
        // Create the service server
        service_ = this->create_service<mbk_pid_controler_interface::srv::ControlCommand>(
            "compute_control_force", std::bind(&ControllerServiceNode::compute_control_callback, this, std::placeholders::_1, std::placeholders::_2));

        RCLCPP_INFO(this->get_logger(), "Controller Service Node has been started and is ready to compute control forces.");
    }

private:
    void compute_control_callback(
        const std::shared_ptr<mbk_pid_controler_interface::srv::ControlCommand::Request> request,
        std::shared_ptr<mbk_pid_controler_interface::srv::ControlCommand::Response> response)
    {
        // Extract the state from the request
        double current_position = request->position;
        double current_velocity = request->velocity;

        RCLCPP_INFO(this->get_logger(), "Received state -> Position: %f, Velocity: %f", current_position, current_velocity);

        // Compute the error
        double error = setpoint - current_position;

        // Compute the integral error
        integral_error += error * 0.01;

        // Compute control force using the PID controller
        double control_force = Kp * error - Kd * current_velocity - Ki * integral_error;

        RCLCPP_INFO(this->get_logger(), "Computed control force: %f", control_force);

        // Set the response
        response->control_force = control_force;
    }

    double Kp, Kd, Ki;
    double integral_error, setpoint;
    rclcpp::Service<mbk_pid_controler_interface::srv::ControlCommand>::SharedPtr service_;
};

int main(int argc, char ** argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<ControllerServiceNode>());
    rclcpp::shutdown();
    return 0;
}
