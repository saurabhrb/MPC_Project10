#include <math.h>
#include <uWS/uWS.h>
#include <chrono>
#include <iostream>
#include <thread>
#include <vector>
#include "Eigen-3.3/Eigen/Core"
#include "Eigen-3.3/Eigen/QR"
#include "MPC.h"
#include "json.hpp"

// for convenience
using json = nlohmann::json;

// For converting back and forth between radians and degrees.
constexpr double pi() { return M_PI; }
double deg2rad(double x) { return x * pi() / 180; }
double rad2deg(double x) { return x * 180 / pi(); }

// Format Eigen log output for printing vectors and matrices:
//   precision=4, flags=0, coeffSeparator=", ", rowSeparator="\n",
//   rowPrefix="[", rowSuffix="]"
Eigen::IOFormat Clean(4, 0, ", ", "\n", "[", "]");

std::string hasData(std::string s) {
  auto found_null = s.find("null");
  auto b1 = s.find_first_of("[");
  auto b2 = s.rfind("}]");
  if (found_null != std::string::npos) {
    return "";
  } else if (b1 != std::string::npos && b2 != std::string::npos) {
    return s.substr(b1, b2 - b1 + 2);
  }
  return "";
}

/**
 * Evaluate a polynomial defined by its coefficients at point x.
 */
double polyeval(Eigen::VectorXd coeffs, double x) {
  double result = 0.0;
  for (int i = 0; i < coeffs.size(); i++) {
    result += coeffs[i] * pow(x, i);
  }
  return result;
}

// Fit a polynomial.
// Adapted from
// https://github.com/JuliaMath/Polynomials.jl/blob/master/src/Polynomials.jl#L676-L716
Eigen::VectorXd polyfit(Eigen::VectorXd xvals, Eigen::VectorXd yvals,
                        int order, Eigen::ArrayXd weights) {
  assert(xvals.size() == yvals.size());
  assert(order >= 1 && order <= xvals.size() - 1);

  //std::cout << "\nx:\n" << xvals.format(Clean) << std::endl;
  //std::cout << "\ny:\n" << yvals.format(Clean) << std::endl;

  // A is the Vandermonde matrix for xvals, [1, x, x^2, x^3, ...]
  Eigen::MatrixXd A(xvals.size(), order + 1);
  for (int i = 0; i < xvals.size(); i++) {
    A(i, 0) = 1.0;
  }
  for (int j = 0; j < xvals.size(); j++) {
    for (int i = 0; i < order; i++) {
      A(j, i + 1) = A(j, i) * xvals(j);
    }
  }
  
  // Apply weighting by element-wise multiplication of left and right sides
  // of the equation A(x) * [coeffs] = y.  For Eigen to do element-wise
  // multiplication, the .array() form needs to be used.
  A.array().colwise() *= weights;
  yvals.array() *= weights;

  auto Qr = A.householderQr();
  auto coeffs = Qr.solve(yvals);
  return coeffs;
}

/**
 * Interpolate additional points in the beginning sections of the waypoint
 * vector to prepare for polyfitting the car's reference line in order to
 * get a closer fit on the sections that are closer to the car's current
 * position.  This helps prevent the polyfit from curving away from the car
 * and jumping around when the set of waypoints changes as the car is driving.
 *
 * Returns an Eigen vector of the interpolated waypoints with the beginning
 * 'n_sections' # of sections having 'n_interp_pts' # of points.
 */
Eigen::VectorXd waypoint_interp(Eigen::VectorXd waypts, int n_interp_pts,
                                int n_sections) {
  
  // Need at least 2 points per section (start and end point)
  if (n_interp_pts < 2) { n_interp_pts = 2; }
  
  // # of interp pts for calcs = minus 1 to avoid overlapping endpoints
  const int n_ipts_wk = n_interp_pts - 1;

  // New total pts = original # of pts - (1 pt for each section)
  //                 + (number of new interp pts) * (# of interp sections)
  const long n_tot_ipts = waypts.size() - n_sections + (n_ipts_wk * n_sections);
  
  Eigen::VectorXd waypts_interp(n_tot_ipts);
  for (int i = 0; i < waypts.size(); ++i) {
    if (i < n_sections) {
      // interpolate new points
      double m = (waypts(i+1) - waypts(i)) / n_ipts_wk; // interpolated slope
      for (int j = 0; j < n_ipts_wk; ++j) {
        waypts_interp[i*n_ipts_wk + j] = waypts(i) + (j * m); // interp point
      }
    }
    else {
      // pass through original points
      waypts_interp[(i-n_sections) + (n_ipts_wk*n_sections)] = waypts(i);
    }
  }
  return waypts_interp;
}

/**
 * Main loop to process measurements received from Udacity simulator via
 * uWebSocket messages.  After receiving current vehicle x,y position, nearest
 * waypoints, heading angle, speed, steering angle, and throttle, process it
 * using an MPC controller and send resulting control steering angle and
 * throttle values back to the simulator to drive around the track.
 */
int main() {
  uWS::Hub h;
  
  // Create MPC instance and initialize loop counter
  MPC mpc;
  long int n_loop = 0;
  
  // Set debug logging decimal precision
  //std::cout << std::fixed;
  //std::cout << std::setprecision(6);
  
  /**
   * Loop on communication message with simulator
   */
  h.onMessage([&mpc, &n_loop](uWS::WebSocket<uWS::SERVER> ws, char *data,
                              size_t length, uWS::OpCode opCode) {
    
    // Store time at start of processing received data for latency estimation
    std::chrono::high_resolution_clock::time_point t_start =
                                      std::chrono::high_resolution_clock::now();
    
    // "42" at the start of the message means there's a websocket message event.
    // The 4 signifies a websocket message
    // The 2 signifies a websocket event
    std::string sdata = std::string(data).substr(0, length);
    //cout<<"-------------\n";
   // cout<<"RECEIVED: \n\n";
    //cout << sdata << endl;
    //cout<<"-------------\n";
    if (sdata.size() > 2 && sdata[0] == '4' && sdata[1] == '2') {
      std::string s = hasData(sdata);
      if (s != "") {
        auto j = json::parse(s);
        std::string event = j[0].get<std::string>();
        if (event == "telemetry") {
          // j[1] is the data JSON object
          
          std::vector<double> ptsx = j[1]["ptsx"]; // waypoint global x coords
          std::vector<double> ptsy = j[1]["ptsy"]; // waypoint global y coords
          double px = j[1]["x"]; // car's global x coord
          double py = j[1]["y"]; // car's global y coord
          double psi = j[1]["psi"]; // car's global heading angle (rad)
          double v = j[1]["speed"]; // current speed (mph)
          v *= (1609.34 / 3600); // convert v from mph to m/s

          double steer_value = j[1]["steering_angle"]; // current steer (rad)
          steer_value *= -1.0; // flip direction from sim to match motion eqn's
          
          double throttle_value = j[1]["throttle"]; // current throttle [-1, 1]
          
          n_loop += 1; // increment loop counter for data logging reference

          // Debug log actual vs predicted position stored from last cycle
          /*
          std::cout << "err px: " << (px - mpc.px_pred)
                    << ", py: " << (py - mpc.py_pred)
                    << ", psi: " << (psi - mpc.psi_pred)
                    << ", v: " << (v - mpc.v_pred)
                    << ", latency: " << mpc.ave_latency_ms << std::endl;
          */
      
	//for csv values

	cout<<n_loop<<","<<psi<<","<<px<<","<<py<<","<<steer_value<<","<<throttle_value<<","<<v<<","<<mpc.ave_latency_ms_<<",";
	cout<<mpc.psi_pred_<<","<<mpc.px_pred_<<","<<mpc.py_pred_<<","<<mpc.v_pred_<<",";

          // Adjust predicted veh position to be after estimated latency time
          // using motion equations:
          //   x = x + v * cos(psi) * dt
          //   y = y + v * sin(psi) * dt
          //   psi = psi + v / Lf * delta * dt
          //   v = v + a * dt, use throttle ~ a
          const double latency = mpc.ave_latency_ms_ / 1000;
          const double v_new = v + throttle_value * latency;
          const double v_ave = (v + v_new) / 2;
          const double psi_new = psi + v_ave / mpc.Lf_ * steer_value * latency;
          const double psi_ave = (psi + psi_new) / 2;
          mpc.px_pred_ = px + v_ave * cos(psi_ave) * latency;
          mpc.py_pred_ = py + v_ave * sin(psi_ave) * latency;
          mpc.psi_pred_ = psi_new;
          mpc.v_pred_ = v_new;
          
          // Convert waypoints from global to vehicle coordinates at
          // vehicle's predicted position after latency
          Eigen::VectorXd ptsx_veh(ptsx.size());
          Eigen::VectorXd ptsy_veh(ptsy.size());
          for (int i = 0; i < int(ptsx.size()); ++i) {
            ptsx_veh(i) = (ptsx[i] - mpc.px_pred_) * cos(mpc.psi_pred_)
                          + (ptsy[i] - mpc.py_pred_) * sin(mpc.psi_pred_);
            
            ptsy_veh(i) = -(ptsx[i] - mpc.px_pred_) * sin(mpc.psi_pred_)
                           + (ptsy[i] - mpc.py_pred_) * cos(mpc.psi_pred_);
          }
          
          //std::cout << "\nptsx_veh:\n" << ptsx_veh.format(Clean) << std::endl;
          //std::cout << "\nptsy_veh:\n" << ptsy_veh.format(Clean) << std::endl;
          
          // Interpolate additional waypoints between first two sections for
          // tighter polyfit close to vehicle's position to prevent reference
          // line from jumping around when new waypoints are processed
          constexpr int n_interp_pts = 3; // interp section to have 3 points
          constexpr int n_interp_sections = 2; // interp for first 2 sections
          
          auto ptsx_interp = waypoint_interp(ptsx_veh, n_interp_pts,
                                             n_interp_sections);
          
          auto ptsy_interp = waypoint_interp(ptsy_veh, n_interp_pts,
                                             n_interp_sections);
          
          //std::cout << "\nptsx_interp:\n" << ptsx_interp.format(Clean) << std::endl;
          //std::cout << "\nptsy_interp:\n" << ptsy_interp.format(Clean) << std::endl;
  
          // Apply heavier polyfit weighting for interpolated sections.
          Eigen::ArrayXd weights = Eigen::ArrayXd::Ones(ptsx_interp.size());
          for (int i = 0; i < (n_interp_pts*n_interp_sections-1); ++i) {
            weights(i) = 5;
          }
         
          //std::cout << "\nweights:\n" << weights.format(Clean) << std::endl;
          
          // Fit weighted interpolated waypoints to get 3rd order poly coeffs
          constexpr int n_poly_order = 3;
          auto waypt_coeffs = polyfit(ptsx_interp, ptsy_interp,
                                      n_poly_order, weights);
          
          // Set current state predicted after latency adjustment and
          // conversion to vehicle coordinates (becomes x=0, y=0, psi=0
          // and dt = 0 since calculating state at current time)
          Eigen::VectorXd cur_state(6);
          
          // epsi = (psi - psides) + (v * delta / Lf * dt)
          //      = (0 - psides) + (0) = -psides
          // psides = atan(f'(x)) = atan(f'(0))
          //        = atan(coeffs[1] + 2*coeffs[2]*x + 3*coeffs[3]*x^2)
          //        = atan(coeffs[1] + 0 + 0) = atan(coeffs[1])
          //   -> epsi = -psides = -atan(coeffs[1])
          double epsi = -atan(waypt_coeffs(1));
          
          // cte = (y - f(x)) + (v * sin(epsi) * dt)
          //     = (0 - f(0)) + (0) = -f(0)
          // f(x) = coeffs[0] + coeffs[1]*x + coeffs[2]*x^2 + coeffs[3]*x^3
          // f(0) = coeffs[0] + 0 + 0 + 0 = coeffs[0]
          //   -> cte = -f(0) = -coeffs[0]
          double cte = -waypt_coeffs(0);
          
          // Debug log output data for reference
	//cout<<"-------------\n";
    	//cout<<"SENT: \n\n";
    	/*std::cout << "n: " << n_loop << ", px: " << px << ", py: " << py
                    << ", psi: " << psi << ", v: " << v
                    << ", steer: " << steer_value
                    << ", throttle: " << throttle_value
                    << ", cte: " << cte << ", epsi: " << epsi
                    << ", latency: " << mpc.ave_latency_ms_ << std::endl;
          
    cout<<"-------------\n";
*/
	//cout<<steer_value<<","<<throttle_value<<          
          // Populate current state vector (x=0, y=0, psi=0, v, cte, epsi)
          cur_state << 0, 0, 0, mpc.v_pred_, cte, epsi;
          
          // Solve MPC optimizer with current state and waypoint coefficients
          auto mpc_result = mpc.Solve(cur_state, waypt_coeffs);
          
          // Set control for steering and throttle to MPC 1st actuator step
          // result.  NOTE: Remember to divide by deg2rad(25) before you send
          // the steering value back, otherwise the values will be in between
          // [-deg2rad(25), deg2rad(25)] instead of [-1, 1] and flip sign.
          json msgJson;
          msgJson["steering_angle"] = -mpc_result[0] / deg2rad(25);
          msgJson["throttle"] = mpc_result[1];

          // Display the MPC predicted trajectory (GREEN line)
          msgJson["mpc_x"] = mpc.mpc_path_x_;
          msgJson["mpc_y"] = mpc.mpc_path_y_;

	cout<<msgJson["steering_angle"]<<","<<msgJson["throttle"]<<","<<cte<<","<<epsi<<","<<mpc.ave_latency_ms_<<"\n";
          // Display the waypoints/reference line (YELLOW line)
          //   (copy waypoint Eigen vectors 'ptsx_veh', 'ptsy_veh' back to
          //    standard vectors 'next_x_vals', 'next_y_vals')
          std::vector<double> next_x_vals(ptsx_veh.data(),
                                          ptsx_veh.data() + ptsx_veh.size());

          std::vector<double> next_y_vals(ptsy_veh.data(),
                                          ptsy_veh.data() + ptsy_veh.size());
          
          msgJson["next_x"] = next_x_vals;
          msgJson["next_y"] = next_y_vals;

          // Package message to send to simulator
          auto msg = "42[\"steer\"," + msgJson.dump() + "]";
//cout<<msg<<"\n";
//            cout<<"-------------\n";
          // Latency
          // The purpose is to mimic real driving conditions where
          // the car does actuate the commands instantly.
          //
          // Feel free to play around with this value but should be to drive
          // around the track with 100ms latency.
          //
          // NOTE: REMEMBER TO SET THIS TO 100 MILLISECONDS BEFORE
          // SUBMITTING.
          std::this_thread::sleep_for(std::chrono::milliseconds(100));
          
          // Estimate actual ave latency (processing time + actuator latency)
          auto cur_latency_ms =
                  std::chrono::duration_cast<std::chrono::milliseconds>
                  (std::chrono::high_resolution_clock::now() - t_start).count();
          
          // Smooth stored latency with exponential moving average
          constexpr int n_sm = 3;
          mpc.ave_latency_ms_ = mpc.ave_latency_ms_ * (n_sm - 1)/n_sm
                                 + cur_latency_ms * 1/n_sm;
          
          // Send message to simulator
          ws.send(msg.data(), msg.length(), uWS::OpCode::TEXT);
        }
      } else {
        // Manual driving
        std::string msg = "42[\"manual\",{}]";
        ws.send(msg.data(), msg.length(), uWS::OpCode::TEXT);
      }
    }
  });

  // We don't need this since we're not using HTTP but if it's removed the
  // program doesn't compile :-(
  h.onHttpRequest([](uWS::HttpResponse *res, uWS::HttpRequest req, char *data,
                     size_t, size_t) {
    const std::string s = "<h1>Hello world!</h1>";
    if (req.getUrl().valueLength == 1) {
      res->end(s.data(), s.length());
    }
    else {
      // I guess this should be done more gracefully?
      res->end(nullptr, 0);
    }
  });

  h.onConnection([&h, &n_loop](uWS::WebSocket<uWS::SERVER> ws, uWS::HttpRequest req) {
    std::cout << "Connected!!!" << std::endl;
	n_loop = 0;
cout<<"psi,px,py,steer,throttle,v,ms_old,psi_pred,px_pred,py_pred,v_pred,steer_new,throttle_new,cte,epsi,ms_new\n";
  });

  h.onDisconnection([&h, &n_loop](uWS::WebSocket<uWS::SERVER> ws, int code,
                         char *message, size_t length) {
    //ws.close();
	n_loop = 0;
    std::cout << "Disconnected" << std::endl;

  });

  int port = 4567;
  if (h.listen(port)) {
    std::cout << "Listening to port " << port << std::endl;
  }
  else {
    std::cerr << "Failed to listen to port" << std::endl;
    return -1;
  }
  h.run();
}
