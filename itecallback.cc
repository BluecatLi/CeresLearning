#include "ceres/ceres.h"
#include "glog/logging.h"
using ceres::AutoDiffCostFunction;
using ceres::CostFunction;
using ceres::Problem;
using ceres::Solve;
using ceres::Solver;
using namespace std;
struct CURVE_FITTING_COST
{
	CURVE_FITTING_COST(double x, double y):_x(x),_y(y){}
	template <typename T>
	bool operator()(const T* const a,T* residual)const
	{
		residual[0]=_y-(0.02*(_x+a[0])*(_x+a[0])*(_x+a[0])-0.3*(_x+a[0])*(_x+a[0])+a[1]);
		return true;
	}
	const double _x, _y;
};

class LoggingCallback : public ceres::IterationCallback {
 public:
  explicit LoggingCallback(bool log_to_stdout,double* a)
      : log_to_stdout_(log_to_stdout),a(a) {}

  ~LoggingCallback() {}

  ceres::CallbackReturnType operator()(const ceres::IterationSummary& summary) {
    if(log_to_stdout_){
	  cout<<"a:"<<a[0]<<endl;
   	  cout<<"b:"<<a[1]<<endl;}
    return ceres::SOLVER_CONTINUE;
  }

 private:
  const bool log_to_stdout_;
  double* a;
};


int main() {
  // The variable to solve for with its initial value. It will be
  // mutated in place by the solver.
  double a[2] = {-5.2,3.7};
  double a_initial[2]={-5.2,3.7};
  vector<double> x_data,y_data;
  for(int i=0;i<1000;i++)
  {
	  double x=i/100.0;
	  x_data.push_back(x);
	  y_data.push_back(0.02*x*x*x-0.3*x*x+1.0);
  }


  // Build the problem.
  Problem problem;
  for(int i=0;i<1000;i++)
  {
  problem.AddResidualBlock(
	  new AutoDiffCostFunction<CURVE_FITTING_COST,1,2>(
		  new CURVE_FITTING_COST(x_data[i],y_data[i])
		  ),
	  nullptr,
	  a
	  );
  }
  // Set up the only cost function (also known as residual). This uses
  // auto-differentiation to obtain the derivative (jacobian).
  // Run the solver!
  Solver::Options options;
 // double* address =problem.GetParameterBlock;
 // options.linear_solver_type=ceres::DENSE_QR;
  options.minimizer_progress_to_stdout = true;
  options.update_state_every_iteration = true;
  LoggingCallback* lc = new LoggingCallback(true,a);
  options.callbacks.push_back(lc);
  Solver::Summary summary;
  Solve(options, &problem, &summary);
  std::cout << summary.BriefReport() << "\n";
  std::cout << "a : " << a_initial[0] << " -> " << a[0] << "\n";
  std::cout << "b : " << a_initial[1] << " -> " << a[1] << "\n";
  return 0;
}

