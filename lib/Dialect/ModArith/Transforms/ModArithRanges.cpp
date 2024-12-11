#include "lib/Dialect/ModArith/Transforms/ModArithRanges.h"

#include "lib/Dialect/ModArith/IR/ModArithOps.h"

#include "mlir/include/mlir/Analysis/DataFlow/DeadCodeAnalysis.h"  // from @llvm-project
#include "mlir/include/mlir/Analysis/DataFlow/IntegerRangeAnalysis.h"  // from @llvm-project
#include "mlir/include/mlir/Analysis/DataFlowFramework.h"  // from @llvm-project
#include "mlir/include/mlir/IR/MLIRContext.h"   // from @llvm-project
#include "mlir/include/mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/include/mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project

#define DEBUG_TYPE "mod-arith-ranges"

namespace mlir {
namespace heir {
namespace mod_arith {

#define GEN_PASS_DEF_MODARITHRANGES
#include "lib/Dialect/ModArith/Transforms/Passes.h.inc"

struct ModArithRanges : impl::ModArithRangesBase<ModArithRanges> {
  using ModArithRangesBase::ModArithRangesBase;

  void runOnOperation() override {
    Operation *module = getOperation();

    DataFlowSolver solver;
    solver.load<dataflow::DeadCodeAnalysis>();
    solver.load<dataflow::IntegerRangeAnalysis>();
    if (failed(solver.initializeAndRun(module)))
        signalPassFailure();

    auto result = module->walk([&](Operation *op) {
      return WalkResult::advance();
    });
  }
};

}  // namespace mod_arith
}  // namespace heir
}  // namespace mlir
