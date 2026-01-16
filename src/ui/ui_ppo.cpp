/**
 * @file ui_ppo.cpp
 * @brief PPO Training with Web-based Visualization
 * 
 * This provides real-time visualization of PPO training in a web browser.
 * Access at http://localhost:8080 after starting.
 */

#include <boost/beast/core.hpp>
#include <boost/beast/http.hpp>
#include <boost/beast/version.hpp>
#include <boost/asio.hpp>
#include <chrono>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <memory>
#include <string>
#include <boost/beast/websocket.hpp>
#include <filesystem>
#include <fstream>
#include <rl_tools/operations/cpu_mux.h>
#include <learning_to_fly/simulator/operations_cpu.h>
#include <learning_to_fly/simulator/ui.h>
namespace rlt = rl_tools;

#include "../training_ppo.h"

namespace beast = boost::beast;
namespace http = beast::http;
namespace net = boost::asio;
using tcp = boost::asio::ip::tcp;

namespace my_program_state {
    std::size_t request_count() {
        static std::size_t count = 0;
        return ++count;
    }
    std::time_t now() {
        return std::time(0);
    }
}

class websocket_session : public std::enable_shared_from_this<websocket_session> {
    beast::websocket::stream<tcp::socket> ws_;

    using ABLATION_SPEC = learning_to_fly::config::DEFAULT_ABLATION_SPEC;
    using CONFIG = learning_to_fly::config::ppo::PPOConfig<ABLATION_SPEC>;
    using TI = CONFIG::TI;

    learning_to_fly::PPOTrainingState<CONFIG> ts;
    boost::asio::steady_timer timer_;
    std::chrono::time_point<std::chrono::high_resolution_clock> training_start, training_end;
    std::thread t;
    std::vector<std::vector<typename CONFIG::ENVIRONMENT::State>> ongoing_trajectories;
    std::vector<TI> ongoing_drones;
    std::vector<TI> idle_drones;
    TI drone_id_counter = 0;
    using T = CONFIG::T;
    using ENVIRONMENT = typename CONFIG::ENVIRONMENT;
    ENVIRONMENT env;
    typename ENVIRONMENT::State current_state;
    rlt::devices::DefaultCPU device;
    rlt::MatrixDynamic<rlt::matrix::Specification<T, TI, 1, ENVIRONMENT::ACTION_DIM>> action;
    bool training_initialized = false;
    bool paused = false;

public:
    explicit websocket_session(tcp::socket socket) : ws_(std::move(socket)), timer_(ws_.get_executor()) {
        env.parameters = parameters::environment<T, TI, ABLATION_SPEC>::parameters;
        rlt::malloc(device, action);
    }

    template<class Body>
    void run(http::request<Body>&& req) {
        ws_.async_accept(
                req,
                beast::bind_front_handler(
                        &websocket_session::on_accept,
                        shared_from_this()
                )
        );
    }

    void on_accept(beast::error_code ec) {
        if(ec) return;
        do_read();
    }

    void do_read() {
        ws_.async_read(
                buffer_,
                beast::bind_front_handler(
                        &websocket_session::on_read,
                        shared_from_this()
                )
        );
    }

    void on_read(beast::error_code ec, std::size_t bytes_transferred) {
        boost::ignore_unused(bytes_transferred);
        if(ec == beast::websocket::error::closed) return;
        if(ec) {
            std::cerr << "read error: " << ec.message() << "\n";
            return;
        }

        std::string message = beast::buffers_to_string(buffer_.data());
        buffer_.consume(buffer_.size());
        nlohmann::json j = nlohmann::json::parse(message);

        if(j["channel"] == "control") {
            std::string action_str = j["data"]["action"];
            if(action_str == "start") {
                if(!t.joinable()) {
                    t = std::thread([this](){
                        training_start = std::chrono::high_resolution_clock::now();
                        learning_to_fly::ppo_training::init(ts, 0);
                        training_initialized = true;
                        training_end = std::chrono::high_resolution_clock::now();
                    });
                }
            }
            if(action_str == "pause") {
                paused = true;
            }
            if(action_str == "resume") {
                paused = false;
            }
        }

        do_read();
    }

    void start_training_loop() {
        if(t.joinable()) {
            t.join();
        }
        do_training_step();
    }

    void do_training_step() {
        timer_.expires_after(std::chrono::milliseconds(1));
        timer_.async_wait([this](beast::error_code ec) {
            if(ec) return;

            if(training_initialized && !paused) {
                training_start = std::chrono::high_resolution_clock::now();
                
                // Run PPO step
                if(ts.step < CONFIG::STEP_LIMIT) {
                    learning_to_fly::ppo_training::step(ts);
                }
                
                training_end = std::chrono::high_resolution_clock::now();
            }

            // Send state updates to browser
            send_state_updates();
            
            do_training_step();
        });
    }

    void send_state_updates() {
        if(!training_initialized) return;
        
        // Send current environment states for visualization
        using UI = rlt::rl::environments::multirotor::UI<ENVIRONMENT>;
        UI ui;
        
        // Get state from current episode if available
        if(!ts.current_episode.empty()) {
            current_state = ts.current_episode.back();
        }
        
        rlt::set_all(device, action, 0);
        
        try {
            ws_.write(net::buffer(rlt::rl::environments::multirotor::state_message(device, ui, current_state, action).dump()));
        } catch(...) {}
    }

private:
    beast::flat_buffer buffer_;
};

class http_connection : public std::enable_shared_from_this<http_connection> {
public:
    http_connection(tcp::socket socket) : socket_(std::move(socket)) {}

    void start() {
        read_request();
        check_deadline();
    }

private:
    tcp::socket socket_;
    beast::flat_buffer buffer_{8192};
    http::request<http::string_body> request_;
    http::response<http::dynamic_body> response_;
    net::steady_timer deadline_{socket_.get_executor(), std::chrono::seconds(60)};

    void read_request() {
        auto self = shared_from_this();
        http::async_read(socket_, buffer_, request_,
            [self](beast::error_code ec, std::size_t bytes_transferred) {
                boost::ignore_unused(bytes_transferred);
                if(!ec) self->process_request();
            });
    }

    void process_request() {
        response_.version(request_.version());
        response_.keep_alive(false);

        if(beast::websocket::is_upgrade(request_)) {
            std::make_shared<websocket_session>(std::move(socket_))->run(std::move(request_));
            return;
        }

        switch(request_.method()) {
            case http::verb::get:
                response_.result(http::status::ok);
                response_.set(http::field::server, "Learning to Fly PPO");
                create_response();
                break;
            default:
                response_.result(http::status::bad_request);
                response_.set(http::field::content_type, "text/plain");
                beast::ostream(response_.body()) << "Invalid request-method '" << std::string(request_.method_string()) << "'";
                break;
        }
        write_response();
    }

    void create_response() {
        std::string target = std::string(request_.target());
        if(target == "/") target = "/index.html";
        
        std::filesystem::path static_path = std::filesystem::path(__FILE__).parent_path() / "static" / target.substr(1);
        
        if(std::filesystem::exists(static_path)) {
            std::ifstream file(static_path);
            std::stringstream ss;
            ss << file.rdbuf();
            
            std::string ext = static_path.extension().string();
            if(ext == ".html") response_.set(http::field::content_type, "text/html");
            else if(ext == ".js") response_.set(http::field::content_type, "application/javascript");
            else if(ext == ".css") response_.set(http::field::content_type, "text/css");
            
            beast::ostream(response_.body()) << ss.str();
        } else {
            response_.result(http::status::not_found);
            response_.set(http::field::content_type, "text/plain");
            beast::ostream(response_.body()) << "File not found: " << target;
        }
    }

    void write_response() {
        auto self = shared_from_this();
        response_.content_length(response_.body().size());
        http::async_write(socket_, response_,
            [self](beast::error_code ec, std::size_t) {
                self->socket_.shutdown(tcp::socket::shutdown_send, ec);
                self->deadline_.cancel();
            });
    }

    void check_deadline() {
        auto self = shared_from_this();
        deadline_.async_wait([self](beast::error_code ec) {
            if(!ec) self->socket_.close(ec);
        });
    }
};

void http_server(tcp::acceptor& acceptor, tcp::socket& socket) {
    acceptor.async_accept(socket, [&](beast::error_code ec) {
        if(!ec) std::make_shared<http_connection>(std::move(socket))->start();
        http_server(acceptor, socket);
    });
}

int main(int argc, char* argv[]) {
    try {
        unsigned short port = 8080;
        if(argc > 1) port = static_cast<unsigned short>(std::atoi(argv[1]));

        auto const address = net::ip::make_address("0.0.0.0");
        net::io_context ioc{1};
        tcp::acceptor acceptor{ioc, {address, port}};
        tcp::socket socket{ioc};

        std::cout << "================================================\n";
        std::cout << "Learning to Fly - PPO Training with UI\n";
        std::cout << "================================================\n\n";
        std::cout << "Open http://localhost:" << port << " in your browser\n";
        std::cout << "to visualize PPO training in real-time.\n\n";

        http_server(acceptor, socket);
        ioc.run();
    } catch(std::exception const& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
