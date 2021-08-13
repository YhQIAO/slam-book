//
// Created by qyh on 2021/8/14.
// the official ceres tutorial
//

#include <iostream>
#include <ceres/ceres.h>

using namespace std;

// demo: find the minimum value of function
// f(x) = 1/2 * (10-x)**2

// The first step is to write a functor
// that will evaluate this the function
// f(x) = 10-x

struct CostFunctor {
    template<typename T>
    bool operator()(const T* const x, T* redidual) const {
        redidual[0] = 10.0-x[0];
        return true;
    }
};

int main(int argc, char** argv) {
    double initial_x = 5.0;
    double x = initial_x;

    ceres::Problem problem;
    ceres::CostFunction* costFunction =
            new ceres::AutoDiffCostFunction<
                    CostFunctor,1,1>(
                            new CostFunctor
                            );
    problem.AddResidualBlock(costFunction, nullptr, &x);

    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_QR;
    options.minimizer_progress_to_stdout = true;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    cout << summary.BriefReport() << endl;
    cout << "x: " << initial_x << "->" << x << endl;
    return 0;

}